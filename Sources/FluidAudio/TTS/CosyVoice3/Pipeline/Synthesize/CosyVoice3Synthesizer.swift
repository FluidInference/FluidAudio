@preconcurrency import CoreML
import Foundation

/// Top-level synthesizer orchestrating prefill → decode loop → Flow → HiFT.
///
/// Mirrors `verify/test_coreml_e2e_fp16.py::main()` in Python. Each stage is
/// implemented as a method on this type, keeping the state (KV cache, running
/// decoded list) local to a single synthesis call.
///
/// Decode is **stateless** with an external KV cache. Prefill emits
/// `kv_k` / `kv_v` of shape `[24, 1, 2, 768, 64]` fp32; decode accepts those
/// same tensors as inputs and returns updated `kv_k_out` / `kv_v_out` at
/// the same shape/dtype. We round-trip the cache once per step (≈18 MB
/// total) and bind the previous step's outputs as the next step's inputs.
/// No `MLState` dependency — runs on macOS 14 / iOS 17.
public actor CosyVoice3Synthesizer {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "CosyVoice3Synthesizer")

    private let models: CosyVoice3Models
    private let embeddings: CosyVoice3SpeechEmbeddings

    /// Set to `false` once `LLM-Decode-M768-fp16` rejects pre-allocated
    /// `outputBackings` (model exported without explicit MultiArray
    /// shape/dtype constraints on its `kv_k_out` / `kv_v_out` /
    /// `speech_logits` outputs). Latched off so we don't throw + catch on
    /// every one of ~163 AR decode steps per phrase. Same pattern as
    /// `MagpieKvCache.useOutputBackings`.
    private var useOutputBackings: Bool = true

    /// One-shot flag for "fast path engaged" log message; only emitted on
    /// the first successful `outputBackings` prediction so we don't spam.
    private var loggedFastPath: Bool = false

    public init(models: CosyVoice3Models, embeddings: CosyVoice3SpeechEmbeddings) {
        self.models = models
        self.embeddings = embeddings
    }

    /// Entry point for the Phase 1 parity harness.
    public func synthesize(
        fixture: CosyVoice3FrontendFixture,
        options: CosyVoice3ParityOptions
    ) async throws -> CosyVoice3SynthesisResult {

        let nPrompt = fixture.promptSpeechIds.count
        let roomForNew = CosyVoice3Constants.flowTotalTokens - nPrompt
        guard roomForNew > 0 else {
            throw CosyVoice3Error.sequenceTooLong(nPrompt)
        }
        let maxNew: Int = {
            if let cap = options.maxNewTokens, cap > 0 { return min(cap, roomForNew) }
            return roomForNew
        }()

        // Sampler. Parity harness seeds the Python-recorded decode stream.
        let sampler = CosyVoice3RasSampler(seed: options.seed)
        if options.replayDecodedTokens {
            sampler.seedTokens(fixture.decodedTokens)
        }

        // 1) Prefill (returns kv_k / kv_v as fp32 outputs)
        let tPrefill = Date()
        let (prefillLogits, initialKvK, initialKvV) = try await runPrefill(fixture: fixture)
        let prefillSec = Date().timeIntervalSince(tPrefill)

        // External KV cache with **double-buffered outputBackings**: prefill's
        // `kv_k` / `kv_v` (shape `[24, 1, 2, 768, 64]` fp32, ~9 MB each) feed
        // the first decode step. Subsequent steps rotate between two
        // pre-allocated buffer pairs (A/B) bound as the model's
        // `kv_k_out` / `kv_v_out` outputs. Same pattern as
        // `MagpieKvCache.swapBackings()` — eliminates ~36 MB of host
        // alloc/dealloc per decode step (×163 steps ≈ 5.9 GB churn per
        // phrase). `speech_logits` is also pre-bound so we avoid a fresh
        // 27 KB allocation each step. CoreML rejects this when the model
        // was exported without explicit MultiArray shape/dtype constraints
        // on its outputs; in that case we latch `useOutputBackings = false`
        // and fall back to per-step allocation for the rest of the run.
        let kvShape: [NSNumber] = [
            NSNumber(value: CosyVoice3Constants.numLayers),
            1,
            NSNumber(value: CosyVoice3Constants.kvHeads),
            NSNumber(value: CosyVoice3Constants.kvMaxLength),
            NSNumber(value: CosyVoice3Constants.headDim),
        ]
        let kvKBackA = try MLMultiArray(shape: kvShape, dataType: .float32)
        let kvVBackA = try MLMultiArray(shape: kvShape, dataType: .float32)
        let kvKBackB = try MLMultiArray(shape: kvShape, dataType: .float32)
        let kvVBackB = try MLMultiArray(shape: kvShape, dataType: .float32)
        let logitsBacking = try MLMultiArray(
            shape: [1, 1, NSNumber(value: CosyVoice3Constants.speechVocab)],
            dataType: .float32)

        // Pointer-rotation triple. `frontKvK/V` are read by the next step;
        // `backKvK/V` receive the next step's writes; `spareKvK/V` are the
        // pre-allocated set ready to become `back` after rotation. Initial
        // `front` is the prefill output; we don't reuse those buffers as
        // `spare`/`back` — once decode step 1 finishes, `front` becomes A
        // (just-written), `back` becomes B (next write target), `spare`
        // becomes A's previous contents (which we drop, since prefill
        // output is single-use).
        var frontKvK: MLMultiArray = initialKvK
        var frontKvV: MLMultiArray = initialKvV
        var backKvK: MLMultiArray = kvKBackA
        var backKvV: MLMultiArray = kvVBackA
        var spareKvK: MLMultiArray = kvKBackB
        var spareKvV: MLMultiArray = kvVBackB

        // Reusable per-step inputs for decode. `curLenArr` is mutated in place
        // each step; `inputsEmbedsArr` is overwritten by memcpy per step.
        let curLenArr = try MLMultiArray(shape: [1], dataType: .int32)
        let inputsEmbedsArr = try MLMultiArray(
            shape: [1, 1, NSNumber(value: CosyVoice3Constants.embedDim)],
            dataType: .float32)

        // Logits scratch reused across all decode steps. The hot loop
        // memcpy's into this from `logitsBacking` (or strided-gathers from a
        // freshly-allocated array on the slow path).
        var logitsScratch = [Float](
            repeating: 0, count: CosyVoice3Constants.speechVocab)

        // First token from prefill tail logits.
        var decoded: [Int32] = []
        let firstLogits = sliceLastStepLogits(
            from: prefillLogits,
            tPre: fixture.tPre,
            vocab: CosyVoice3Constants.speechVocab)
        var topId = sampler.sample(logits: firstLogits, decodedSoFar: decoded)
        if CosyVoice3Constants.stopRange.contains(topId) {
            // Prefill emitted EOS at step 0 — the LLM signaled "no speech".
            // Bail out instead of feeding the stop-token embedding into the
            // decode loop (which would accumulate semantically meaningless
            // tokens into `decoded`).
            logger.info("First token \(topId) is a stop token; no speech generated")
            throw CosyVoice3Error.predictionFailed("LLM produced no speech tokens")
        }
        decoded.append(topId)

        // 2) Decode loop (stateless, external cache, double-buffered backings)
        var curLen = fixture.tPre
        var decodeSteps = 0
        var hitEos = false
        let tDecode = Date()
        for step in 1..<maxNew {
            try embeddings.copyEmbedding(tokenId: topId, into: inputsEmbedsArr)
            curLenArr[0] = NSNumber(value: Int32(curLen))
            try runDecode(
                inputsEmbeds: inputsEmbedsArr,
                curLen: curLenArr,
                frontKvK: frontKvK,
                frontKvV: frontKvV,
                backKvK: backKvK,
                backKvV: backKvV,
                logitsBacking: logitsBacking,
                logits: &logitsScratch)
            topId = sampler.sample(logits: logitsScratch, decodedSoFar: decoded)
            curLen += 1
            decodeSteps += 1
            if CosyVoice3Constants.stopRange.contains(topId) {
                logger.info("EOS at step \(step) (token=\(topId))")
                hitEos = true
                break
            }
            decoded.append(topId)

            // Rotate buffers: `back` (just-written) becomes new `front`;
            // `spare` becomes new `back`; old `front` becomes new `spare`
            // (will be overwritten next step). On step 1 the old `front` is
            // the prefill output — drops to `spare` and gets overwritten on
            // step 3, which is harmless (we never read it again).
            let prevFrontK = frontKvK
            let prevFrontV = frontKvV
            frontKvK = backKvK
            frontKvV = backKvV
            backKvK = spareKvK
            backKvV = spareKvV
            spareKvK = prevFrontK
            spareKvV = prevFrontV
        }
        let decodeSec = Date().timeIntervalSince(tDecode)
        guard !decoded.isEmpty else {
            throw CosyVoice3Error.predictionFailed("LLM produced no speech tokens")
        }

        // Truncation signal: AR loop exhausted its decode budget without
        // observing an EOS token in `stopRange` (6_561…6_760). The 250-token
        // cap is structural — it's the fixed `[1, 250]` shape of the Flow
        // model's `token_total` input (`CosyVoice3Constants.flowTotalTokens`),
        // not a synthesizer-side soft limit. With ~40 ms of audio per token
        // (`tokenMelRatio=2 × hiftSamplesPerFrame=480 / sampleRate=24_000`),
        // a prompt taking ~`nPrompt` tokens leaves `(250 - nPrompt) × 0.04 s`
        // of generated audio — i.e. long phrases truncate mid-utterance.
        //
        // Surface this as a `.warning` so callers running long input get a
        // console signal instead of silent truncation. Lifting the cap
        // requires re-exporting Flow with a larger `token_total` shape; for
        // now, splitting input at clause boundaries (， / 。) is the
        // workaround.
        if !hitEos {
            let producedSec =
                Double(decoded.count)
                * Double(CosyVoice3Constants.tokenMelRatio)
                * Double(CosyVoice3Constants.hiftSamplesPerFrame)
                / Double(CosyVoice3Constants.sampleRate)
            logger.warning(
                "LLM-Decode budget exhausted: \(decoded.count) generated tokens "
                    + "/ \(maxNew) cap (no EOS observed). "
                    + "Output truncated at ~"
                    + String(format: "%.1f", producedSec)
                    + "s of audio. The 250-token Flow input is a structural cap; "
                    + "split long phrases at clause boundaries (， 。) to work around."
            )
        }

        // 3) Flow
        let nNew = decoded.count
        let tFlow = Date()
        let mel = try await runFlow(
            promptSpeechIds: fixture.promptSpeechIds,
            decodedTokens: decoded,
            promptMel: fixture.promptMel,
            promptMelFrames: fixture.promptMelFrames,
            spkEmbedding: fixture.spkEmbedding)
        let flowSec = Date().timeIntervalSince(tFlow)

        // 4) Slice mel to new portion + HiFT
        let numPromptMel = mel.numPromptMel
        let newMelStart = numPromptMel
        let newMelFrames = nNew * CosyVoice3Constants.tokenMelRatio
        let tHift = Date()
        let audio = try await runHiFT(
            fullMel: mel.mel,
            newMelStart: newMelStart,
            newMelFrames: newMelFrames)
        let hiftSec = Date().timeIntervalSince(tHift)

        // Emit stage timings via the shared logger for RTFx benchmarking.
        let decodeTps = decodeSteps > 0 ? Double(decodeSteps) / decodeSec : 0
        logger.info(
            String(
                format:
                    "STAGES prefill=%.3fs decode=%.3fs(%d steps, %.2f tok/s) flow=%.3fs hift=%.3fs",
                prefillSec, decodeSec, decodeSteps, decodeTps, flowSec, hiftSec))

        return CosyVoice3SynthesisResult(
            samples: audio,
            sampleRate: CosyVoice3Constants.sampleRate,
            generatedTokenCount: nNew,
            decodedTokens: decoded,
            finishedOnEos: hitEos)
    }

    // MARK: - Stages

    private func runPrefill(
        fixture: CosyVoice3FrontendFixture
    ) async throws -> (logits: MLMultiArray, kvK: MLMultiArray, kvV: MLMultiArray) {
        guard fixture.tPre <= CosyVoice3Constants.prefillLength else {
            throw CosyVoice3Error.prefillTooLong(fixture.tPre)
        }
        // Pad lm_input_embeds from [1, tPre, 896] to [1, 256, 896].
        // Strides may be non-compact (e.g. [T*D_padded, D_padded, 1]).
        let embeds = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: CosyVoice3Constants.prefillLength),
                NSNumber(value: CosyVoice3Constants.embedDim),
            ],
            dataType: .float32)
        let embedDim = CosyVoice3Constants.embedDim
        let embedsStrides = embeds.strides.map { $0.intValue }
        let dst = embeds.dataPointer.bindMemory(to: Float.self, capacity: embeds.count)
        let physicalCount = embedsStrides[0] * embeds.shape[0].intValue
        dst.initialize(repeating: 0, count: physicalCount)
        for t in 0..<fixture.tPre {
            for d in 0..<embedDim {
                let srcIdx = t * embedDim + d
                let dstOff = t * embedsStrides[1] + d * embedsStrides[2]
                dst[dstOff] = fixture.lmInputEmbeds[srcIdx]
            }
        }
        let inputLen = try MLMultiArray(shape: [1], dataType: .int32)
        inputLen[0] = NSNumber(value: Int32(fixture.tPre))

        let features: [String: Any] = [
            "inputs_embeds": embeds,
            "input_len": inputLen,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await models.prefill.compatPrediction(
            from: provider, options: MLPredictionOptions())

        guard
            let logits = output.featureValue(for: "speech_logits")?.multiArrayValue,
            let kvK = output.featureValue(for: "kv_k")?.multiArrayValue,
            let kvV = output.featureValue(for: "kv_v")?.multiArrayValue
        else {
            throw CosyVoice3Error.predictionFailed("prefill: missing outputs")
        }
        return (logits, kvK, kvV)
    }

    /// Run one stateless decode step with an external KV cache.
    ///
    /// Inputs match the converted CoreML graph signature:
    /// - `inputs_embeds: fp32 [1, 1, 896]`
    /// - `cur_len: int32 [1]`
    /// - `kv_k: fp32 [24, 1, 2, 768, 64]` (previous step's `kv_k_out`, or
    ///   prefill's `kv_k` for the first decode step)
    /// - `kv_v: fp32 [24, 1, 2, 768, 64]`
    ///
    /// Outputs (when `outputBackings` is accepted, written into the pre-
    /// allocated `backKvK` / `backKvV` / `logitsBacking` buffers in place):
    /// - `speech_logits: fp32 [1, 1, 6761]`
    /// - `kv_k_out: fp32 [24, 1, 2, 768, 64]`
    /// - `kv_v_out: fp32 [24, 1, 2, 768, 64]`
    ///
    /// Falls back to per-step CoreML allocation + memcpy into the pre-
    /// allocated backings if the model rejects `outputBackings` (latches
    /// `useOutputBackings = false` so we don't retry on every step).
    private func runDecode(
        inputsEmbeds: MLMultiArray,
        curLen: MLMultiArray,
        frontKvK: MLMultiArray,
        frontKvV: MLMultiArray,
        backKvK: MLMultiArray,
        backKvV: MLMultiArray,
        logitsBacking: MLMultiArray,
        logits: inout [Float]
    ) throws {
        let features: [String: Any] = [
            "inputs_embeds": inputsEmbeds,
            "cur_len": curLen,
            "kv_k": frontKvK,
            "kv_v": frontKvV,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)

        var fastPathSucceeded = false
        if useOutputBackings {
            let opts = MLPredictionOptions()
            opts.outputBackings = [
                "kv_k_out": backKvK,
                "kv_v_out": backKvV,
                "speech_logits": logitsBacking,
            ]
            do {
                _ = try models.decode.prediction(from: provider, options: opts)
                Self.readLogits(from: logitsBacking, into: &logits)
                if !loggedFastPath {
                    logger.info(
                        "LLM-Decode outputBackings accepted; double-buffered "
                            + "AR loop active")
                    loggedFastPath = true
                }
                fastPathSucceeded = true
            } catch {
                // CoreML refused our pre-allocated backings — typically
                // because `LLM-Decode-M768-fp16.mlpackage` was exported
                // without explicit MultiArray shape/dtype constraints on
                // its outputs. Latch the flag off so we don't throw + catch
                // on every one of ~163 steps for the rest of the corpus.
                // Warning level so it shows in release builds — this is a
                // perf regression worth surfacing to anyone running with a
                // re-exported model.
                useOutputBackings = false
                logger.warning(
                    "LLM-Decode outputBackings rejected "
                        + "(\(error.localizedDescription)); switching to "
                        + "fresh-alloc fallback for the rest of the run")
            }
        }

        if !fastPathSucceeded {
            // Slow path: per-step CoreML allocation, then memcpy outputs
            // into the pre-allocated backings so the front/back rotation
            // protocol still works after this call.
            let output = try models.decode.prediction(from: provider)
            guard
                let logitsArr = output.featureValue(for: "speech_logits")?.multiArrayValue,
                let kvKOutArr = output.featureValue(for: "kv_k_out")?.multiArrayValue,
                let kvVOutArr = output.featureValue(for: "kv_v_out")?.multiArrayValue
            else {
                throw CosyVoice3Error.predictionFailed(
                    "decode: missing speech_logits / kv_k_out / kv_v_out")
            }
            try Self.copyKvOutput(kvKOutArr, into: backKvK, name: "kv_k_out")
            try Self.copyKvOutput(kvVOutArr, into: backKvV, name: "kv_v_out")
            Self.readLogits(from: logitsArr, into: &logits)
        }
    }

    /// Read a `[1, 1, 6761]` fp32 logits MLMultiArray into `dst`. Honors the
    /// last-dim stride (CoreML may emit non-compact strides on aligned
    /// allocations) — uses `memcpy` when stride==1, strided gather otherwise.
    private static func readLogits(from arr: MLMultiArray, into dst: inout [Float]) {
        let count = CosyVoice3Constants.speechVocab
        let strides = arr.strides.map { $0.intValue }
        let vocabStride = strides.last ?? 1
        let base = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
        if vocabStride == 1 {
            dst.withUnsafeMutableBytes { rawDst in
                guard let dstPtr = rawDst.baseAddress else { return }
                memcpy(dstPtr, base, count * MemoryLayout<Float>.size)
            }
        } else {
            for i in 0..<count { dst[i] = base[i * vocabStride] }
        }
    }

    /// Copy a CoreML-allocated `kv_k_out` / `kv_v_out` MLMultiArray into our
    /// pre-allocated backing array. Used on the `outputBackings`-rejected
    /// fallback path so the front/back rotation protocol stays consistent.
    private static func copyKvOutput(
        _ src: MLMultiArray,
        into dst: MLMultiArray,
        name: String
    ) throws {
        guard src.dataType == dst.dataType else {
            throw CosyVoice3Error.predictionFailed(
                "decode \(name): dtype mismatch \(src.dataType.rawValue) vs \(dst.dataType.rawValue)")
        }
        guard src.count == dst.count else {
            throw CosyVoice3Error.predictionFailed(
                "decode \(name): count mismatch \(src.count) vs \(dst.count)")
        }
        // KV outputs are fp32. With contiguous strides (the default for
        // freshly-allocated CoreML outputs in this graph) memcpy is safe.
        let bytes = src.count * MemoryLayout<Float>.size
        memcpy(dst.dataPointer, src.dataPointer, bytes)
    }

    private func runFlow(
        promptSpeechIds: [Int32],
        decodedTokens: [Int32],
        promptMel: [Float],
        promptMelFrames: Int,
        spkEmbedding: [Float]
    ) async throws -> (mel: MLMultiArray, numPromptMel: Int) {
        let N = CosyVoice3Constants.flowTotalTokens
        let nPrompt = promptSpeechIds.count
        let nNew = decodedTokens.count
        let nTotal = nPrompt + nNew
        guard nTotal <= N else {
            throw CosyVoice3Error.sequenceTooLong(nTotal)
        }
        // token_total: [1, 250] int32, zero-padded. Respect strides.
        let tokenTotal = try MLMultiArray(
            shape: [1, NSNumber(value: N)],
            dataType: .int32)
        let ttStrides = tokenTotal.strides.map { $0.intValue }
        let ttPtr = tokenTotal.dataPointer.bindMemory(to: Int32.self, capacity: tokenTotal.count)
        let ttPhysical = ttStrides[0] * tokenTotal.shape[0].intValue
        ttPtr.initialize(repeating: 0, count: ttPhysical)
        for i in 0..<nPrompt { ttPtr[i * ttStrides[1]] = promptSpeechIds[i] }
        for i in 0..<nNew { ttPtr[(nPrompt + i) * ttStrides[1]] = decodedTokens[i] }

        // num_prompt_tokens: [1] int32
        let numPromptTokens = try MLMultiArray(shape: [1], dataType: .int32)
        numPromptTokens[0] = NSNumber(value: Int32(nPrompt))

        // prompt_feat: [1, 500, 80] fp32, zero-padded along axis 1. Respect strides.
        let hiftFrames = CosyVoice3Constants.hiftMaxFrames
        let melBins = CosyVoice3Constants.melBins
        let promptFeat = try MLMultiArray(
            shape: [
                1, NSNumber(value: hiftFrames), NSNumber(value: melBins),
            ],
            dataType: .float32)
        let pfStrides = promptFeat.strides.map { $0.intValue }
        let pfPtr = promptFeat.dataPointer.bindMemory(to: Float.self, capacity: promptFeat.count)
        let pfPhysical = pfStrides[0] * promptFeat.shape[0].intValue
        pfPtr.initialize(repeating: 0, count: pfPhysical)
        let copyFrames = min(promptMelFrames, hiftFrames)
        for f in 0..<copyFrames {
            for b in 0..<melBins {
                let srcIdx = f * melBins + b
                let dstOff = f * pfStrides[1] + b * pfStrides[2]
                pfPtr[dstOff] = promptMel[srcIdx]
            }
        }

        // embedding: [1, 192] fp32. Respect strides.
        let embedding = try MLMultiArray(
            shape: [1, NSNumber(value: CosyVoice3Constants.speakerEmbeddingDim)],
            dataType: .float32)
        let eStrides = embedding.strides.map { $0.intValue }
        let ePtr = embedding.dataPointer.bindMemory(to: Float.self, capacity: embedding.count)
        let ePhysical = eStrides[0] * embedding.shape[0].intValue
        ePtr.initialize(repeating: 0, count: ePhysical)
        for i in 0..<spkEmbedding.count { ePtr[i * eStrides[1]] = spkEmbedding[i] }

        let features: [String: Any] = [
            "token_total": tokenTotal,
            "num_prompt_tokens": numPromptTokens,
            "prompt_feat": promptFeat,
            "embedding": embedding,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await models.flow.compatPrediction(
            from: provider, options: MLPredictionOptions())

        guard
            let mel = output.featureValue(for: "mel")?.multiArrayValue,
            let nPromptMelArr = output.featureValue(for: "num_prompt_mel")?.multiArrayValue
        else {
            throw CosyVoice3Error.predictionFailed("flow: missing outputs")
        }
        let nPromptMel = nPromptMelArr[0].intValue
        return (mel, nPromptMel)
    }

    private func runHiFT(
        fullMel: MLMultiArray,
        newMelStart: Int,
        newMelFrames: Int
    ) async throws -> [Float] {
        // fullMel logical shape = [1, 80, 500]. Physical strides may be
        // non-compact (e.g. [40960, 512, 1]) — use logical indexing.
        // Dtype depends on the Flow variant: the ANE-port Flow emits fp16 to
        // keep the graph fp16 end-to-end; the prior cpuAndGPU Flow emits fp32.
        // HiFT's `mel` input is always fp32 at the CoreML I/O boundary.
        let hiftFrames = CosyVoice3Constants.hiftMaxFrames
        let melBins = CosyVoice3Constants.melBins
        // fullMel logical shape = [1, 80, totalMelFrames]. Clamp the valid
        // window to the remaining frames after `newMelStart` so a slightly
        // off `num_prompt_mel` from the Flow model can never cause an
        // out-of-bounds read at `srcBase[newMelStart + f]`.
        let totalMelFrames = fullMel.shape.count >= 3 ? fullMel.shape[2].intValue : hiftFrames
        guard newMelStart >= 0 && newMelStart <= totalMelFrames else {
            throw CosyVoice3Error.invalidShape(
                "runHiFT: newMelStart=\(newMelStart) out of range [0, \(totalMelFrames)]")
        }
        let availableFrames = max(0, totalMelFrames - newMelStart)
        let validFrames = min(newMelFrames, hiftFrames, availableFrames)

        let melInput = try MLMultiArray(
            shape: [1, NSNumber(value: melBins), NSNumber(value: hiftFrames)],
            dataType: .float32)
        // melInput strides may also be non-compact — use logical indexing.
        let melInputStrides = melInput.strides.map { $0.intValue }
        let dstBase = melInput.dataPointer.bindMemory(to: Float.self, capacity: melInput.count)
        // Zero-fill entire physical extent (handles padded strides).
        let totalPhysical = melInputStrides[0] * melInput.shape[0].intValue
        dstBase.initialize(repeating: 0, count: totalPhysical)

        let srcStrides = fullMel.strides.map { $0.intValue }
        // fullMel logical: [1, 80, 500]; copy new slice → melInput [1, 80, 500].
        // Branch on src dtype so the fp16 ANE-port Flow output doesn't get
        // reinterpreted as fp32 (would read past end of buffer → SIGSEGV).
        switch fullMel.dataType {
        case .float16:
            let srcBase = fullMel.dataPointer.bindMemory(
                to: Float16.self, capacity: fullMel.count)
            for b in 0..<melBins {
                for f in 0..<validFrames {
                    let srcOff = b * srcStrides[1] + (newMelStart + f) * srcStrides[2]
                    let dstOff = b * melInputStrides[1] + f * melInputStrides[2]
                    dstBase[dstOff] = Float(srcBase[srcOff])
                }
            }
        case .float32:
            let srcBase = fullMel.dataPointer.bindMemory(
                to: Float.self, capacity: fullMel.count)
            for b in 0..<melBins {
                for f in 0..<validFrames {
                    let srcOff = b * srcStrides[1] + (newMelStart + f) * srcStrides[2]
                    let dstOff = b * melInputStrides[1] + f * melInputStrides[2]
                    dstBase[dstOff] = srcBase[srcOff]
                }
            }
        default:
            throw CosyVoice3Error.predictionFailed(
                "runHiFT: unexpected Flow mel dtype \(fullMel.dataType.rawValue) (expected fp16 or fp32)"
            )
        }

        let numValid = try MLMultiArray(shape: [1], dataType: .int32)
        numValid[0] = NSNumber(value: Int32(validFrames))

        let features: [String: Any] = [
            "mel": melInput,
            "num_valid_frames": numValid,
        ]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try await models.hift.compatPrediction(
            from: provider, options: MLPredictionOptions())

        guard
            let audioArr = output.featureValue(for: "audio")?.multiArrayValue,
            let audioLenArr = output.featureValue(for: "audio_length_samples")?.multiArrayValue
        else {
            throw CosyVoice3Error.predictionFailed("hift: missing outputs")
        }
        let audioLen = audioLenArr[0].intValue
        var out = [Float](repeating: 0, count: audioLen)
        // audio logical shape = [1, 240000]; honor strides.
        let audioStrides = audioArr.strides.map { $0.intValue }
        let aBase = audioArr.dataPointer.bindMemory(to: Float.self, capacity: audioArr.count)
        for i in 0..<audioLen {
            out[i] = aBase[i * audioStrides[1]]
        }
        return out
    }

    // MARK: - Helpers

    /// Extracts logits for the last real prefill position (`tPre - 1`).
    /// Prefill output logical shape is `[1, 256, 6761]` fp32; strides may be
    /// non-compact.
    private func sliceLastStepLogits(
        from logits: MLMultiArray,
        tPre: Int,
        vocab: Int
    ) -> [Float] {
        let strides = logits.strides.map { $0.intValue }
        // shape = [1, T, V]; row (time) stride is strides[1], vocab stride is strides[2].
        let rowStride = strides[1]
        let vocabStride = strides[2]
        let ptr = logits.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
        let base = (tPre - 1) * rowStride
        var out = [Float](repeating: 0, count: vocab)
        for i in 0..<vocab { out[i] = ptr[base + i * vocabStride] }
        return out
    }
}
