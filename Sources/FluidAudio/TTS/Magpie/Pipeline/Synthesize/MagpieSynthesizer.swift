import Accelerate
@preconcurrency import CoreML
import Foundation

/// Orchestrates one Magpie synthesis call end-to-end.
///
/// Pipeline (mirroring `generate_coreml.generate`):
///   1. Tokenize text → padded ids (256) + mask.
///   2. `text_encoder.predict` → encoderOutput (1, 256, 768).
///   3. (CFG) make zero-context encoder pair.
///   4. Prefill: 110 step-by-step `decoder_step` calls with speaker embedding rows.
///   5. AR loop (≤ 500 steps):
///        embed current 8 codes → `decoder_step` → LT sample → new codes.
///   6. NanoCodec decode → fp32 PCM 22 kHz.
///   7. Peak-normalize to 0.9 when `options.peakNormalize`.
public actor MagpieSynthesizer {

    private let logger = AppLogger(category: "MagpieSynthesizer")

    private let store: MagpieModelStore
    private let tokenizer: MagpieTokenizer

    public init(store: MagpieModelStore, tokenizer: MagpieTokenizer) {
        self.store = store
        self.tokenizer = tokenizer
    }

    /// Synthesize from plain text (honors `|...|` IPA override per `options`).
    public func synthesize(
        text: String, speaker: MagpieSpeaker, language: MagpieLanguage,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let tokenized = try await tokenizer.tokenize(text, language: language, options: options)
        return try await synthesize(tokenized: tokenized, speaker: speaker, options: options)
    }

    /// Synthesize from pre-tokenized phoneme ids.
    public func synthesize(
        phonemes: MagpiePhonemeTokens, speaker: MagpieSpeaker,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let tokenized = try await tokenizer.pad(phonemes: phonemes)
        return try await synthesize(tokenized: tokenized, speaker: speaker, options: options)
    }

    // MARK: - Core

    private func synthesize(
        tokenized: MagpieTokenizedText, speaker: MagpieSpeaker,
        options: MagpieSynthesisOptions
    ) async throws -> MagpieSynthesisResult {
        let constants = try await store.constants()
        let ltWeights = try await store.localTransformer()
        let textEncoder = try await store.textEncoder()
        let decoderStep = try await store.decoderStep()
        let nanocodecModel = try await store.nanocodecDecoder()

        let dModel = constants.config.dModel
        let maxTextLen = MagpieConstants.maxTextLength
        let numCodebooks = constants.config.numCodebooks
        let audioBosId = constants.config.audioBosId
        let audioEosId = constants.config.audioEosId
        let speakerContextLength = constants.config.speakerContextLength

        let speakerIndex = speaker.rawValue
        guard speakerIndex >= 0 && speakerIndex < constants.speakerEmbeddings.count else {
            throw MagpieError.invalidSpeakerIndex(speakerIndex)
        }

        // 1. text_encoder
        let textEncoderStart = Date()
        let encResult = try runTextEncoder(
            tokenized: tokenized, maxTextLen: maxTextLen, model: textEncoder)
        let encoderOutput = encResult.encoderOutput
        let encoderMask = encResult.encoderMask
        let textEncoderSeconds = Date().timeIntervalSince(textEncoderStart)
        logger.info(
            "text_encoder done in \(String(format: "%.0f", textEncoderSeconds * 1000))ms")

        let useCfg = options.cfgScale != 1.0
        let uncond: (encoderOutput: MLMultiArray, encoderMask: MLMultiArray)?
        if useCfg {
            uncond = try MagpiePrefill.makeUnconditional(
                encoderOutputShape: encoderOutput.shape, maxTextLen: maxTextLen)
        } else {
            uncond = nil
        }

        // 2. KV caches (conditional + optional unconditional).
        let condCache = try MagpieKvCache(
            numLayers: constants.config.numDecoderLayers,
            maxCacheLength: constants.config.maxCacheLength,
            numHeads: constants.config.numHeads,
            headDim: constants.config.headDim)
        let uncondCache: MagpieKvCache? =
            useCfg
            ? try MagpieKvCache(
                numLayers: constants.config.numDecoderLayers,
                maxCacheLength: constants.config.maxCacheLength,
                numHeads: constants.config.numHeads,
                headDim: constants.config.headDim)
            : nil

        // 3. Prefill (fast batched path when decoder_prefill is available).
        let prefill = MagpiePrefill(decoderStep: decoderStep)
        let hasPrefill = await store.hasDecoderPrefill()
        let prefillStart = Date()
        if hasPrefill {
            let decoderPrefill = try await store.decoderPrefill()
            try prefill.prefillFast(
                decoderPrefill: decoderPrefill,
                speakerEmbedding: constants.speakerEmbeddings[speakerIndex],
                speakerContextLength: speakerContextLength,
                dModel: dModel,
                encoderOutput: encoderOutput,
                encoderMask: encoderMask,
                cache: condCache)
            if let uncondTensors = uncond, let uncondCache = uncondCache {
                let zeroSpeaker = Swift.Array<Float>(
                    repeating: 0, count: speakerContextLength * dModel)
                try prefill.prefillFast(
                    decoderPrefill: decoderPrefill,
                    speakerEmbedding: zeroSpeaker,
                    speakerContextLength: speakerContextLength,
                    dModel: dModel,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache)
            }
        } else {
            try prefill.prefill(
                speakerEmbedding: constants.speakerEmbeddings[speakerIndex],
                speakerContextLength: speakerContextLength,
                dModel: dModel,
                encoderOutput: encoderOutput,
                encoderMask: encoderMask,
                cache: condCache)
            if let uncondTensors = uncond, let uncondCache = uncondCache {
                let zeroSpeaker = Swift.Array<Float>(
                    repeating: 0, count: speakerContextLength * dModel)
                try prefill.prefill(
                    speakerEmbedding: zeroSpeaker,
                    speakerContextLength: speakerContextLength,
                    dModel: dModel,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache)
            }
        }
        let prefillElapsed = Date().timeIntervalSince(prefillStart)
        logger.info(
            "Prefill done in \(String(format: "%.2f", prefillElapsed))s "
                + "(\(hasPrefill ? "fast batched" : "slow loop"))")

        // 4. AR loop.
        let sampler = MagpieLocalSampler(
            localTransformer: MagpieLocalTransformer(weights: ltWeights),
            audioEmbeddings: constants.audioEmbeddings)

        var currentCodes = Swift.Array<Int32>(repeating: audioBosId, count: numCodebooks)
        var allFrames: [[Int32]] = []
        var finishedOnEos = false

        let rng = MagpieSamplerRng(seed: options.seed)

        // Allocate audio_embed buffer once; refill in-place each step (vDSP).
        let audioEmbed = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dModel)], dataType: .float32)

        // Pre-allocate the decoder_hidden output backing once. CoreML writes
        // straight into this array each step; we then read it fp16 → fp32 via
        // vImage. Shape: [1, 1, dModel] fp16 (per decoder_step.mlmodelc).
        let condHiddenBacking = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dModel)], dataType: .float16)
        condHiddenBacking.zeroFillFloat16()
        let uncondHiddenBacking: MLMultiArray? =
            useCfg
            ? {
                let arr = try? MLMultiArray(
                    shape: [1, 1, NSNumber(value: dModel)], dataType: .float16)
                arr?.zeroFillFloat16()
                return arr
            }() : nil

        let arLoopStart = Date()
        var decoderStepNanos: UInt64 = 0
        var samplerNanos: UInt64 = 0

        for step in 0..<options.maxSteps {
            try fillAudioEmbed(
                audioEmbed, codes: currentCodes,
                tables: constants.audioEmbeddings, dModel: dModel)

            let dsStart = DispatchTime.now()
            let condHidden = try runDecoderStep(
                audioEmbed: audioEmbed,
                encoderOutput: encoderOutput, encoderMask: encoderMask,
                cache: condCache, hiddenBacking: condHiddenBacking,
                dModel: dModel, model: decoderStep)

            let uncondHidden: [Float]?
            if useCfg, let uncondTensors = uncond, let uncondCache = uncondCache,
                let uncondBacking = uncondHiddenBacking
            {
                uncondHidden = try runDecoderStep(
                    audioEmbed: audioEmbed,
                    encoderOutput: uncondTensors.encoderOutput,
                    encoderMask: uncondTensors.encoderMask,
                    cache: uncondCache, hiddenBacking: uncondBacking,
                    dModel: dModel, model: decoderStep)
            } else {
                uncondHidden = nil
            }
            decoderStepNanos &+= DispatchTime.now().uptimeNanoseconds &- dsStart.uptimeNanoseconds

            let forbidEos = step < options.minFrames
            let smpStart = DispatchTime.now()
            let next = sampler.sample(
                decoderHidden: condHidden,
                uncondDecoderHidden: uncondHidden,
                forbidEos: forbidEos,
                options: options,
                rng: rng)
            samplerNanos &+= DispatchTime.now().uptimeNanoseconds &- smpStart.uptimeNanoseconds

            let isEos = next.contains(audioEosId)
            if isEos && step >= options.minFrames {
                finishedOnEos = true
                logger.info("EOS at step \(step)")
                break
            }
            allFrames.append(next)
            currentCodes = next
        }

        let numFrames = allFrames.count
        guard numFrames > 0 else {
            throw MagpieError.inferenceFailed(
                stage: "synthesize", underlying: "no audio frames generated")
        }
        let arLoopElapsed = Date().timeIntervalSince(arLoopStart)
        if arLoopElapsed > 0 {
            let dsMs = Double(decoderStepNanos) / 1_000_000.0
            let smpMs = Double(samplerNanos) / 1_000_000.0
            logger.info(
                "AR loop: \(numFrames) frames in "
                    + "\(String(format: "%.2f", arLoopElapsed))s "
                    + "(\(String(format: "%.1f", Double(numFrames) / arLoopElapsed)) fps) "
                    + "decoder=\(String(format: "%.0f", dsMs))ms "
                    + "(\(String(format: "%.1f", dsMs / Double(numFrames)))ms/step) "
                    + "sampler=\(String(format: "%.0f", smpMs))ms "
                    + "(\(String(format: "%.1f", smpMs / Double(numFrames)))ms/step)")
        }

        // 5. NanoCodec decode: reshape (numFrames × numCodebooks) into
        //    per-codebook rows.
        var codebookRows = Swift.Array(
            repeating: Swift.Array<Int32>(repeating: 0, count: numFrames),
            count: numCodebooks)
        for t in 0..<numFrames {
            let row = allFrames[t]
            for cb in 0..<numCodebooks {
                codebookRows[cb][t] = row[cb]
            }
        }
        let nanocodec = MagpieNanocodec(
            model: nanocodecModel, numCodebooks: numCodebooks)
        let nanocodecStart = Date()
        var samples = try nanocodec.decode(frames: codebookRows)
        let nanocodecSeconds = Date().timeIntervalSince(nanocodecStart)
        logger.info(
            "nanocodec done in \(String(format: "%.0f", nanocodecSeconds * 1000))ms")

        // 6. Peak normalize to 0.9.
        if options.peakNormalize {
            var peak: Float = 0
            for s in samples where abs(s) > peak { peak = abs(s) }
            if peak > 0 {
                let scale = MagpieConstants.peakTarget / peak
                for i in 0..<samples.count { samples[i] *= scale }
            }
        }

        let timings = MagpieSynthesisTimings(
            textEncoderSeconds: textEncoderSeconds,
            prefillSeconds: prefillElapsed,
            arLoopSeconds: arLoopElapsed,
            decoderStepSeconds: Double(decoderStepNanos) / 1_000_000_000.0,
            samplerSeconds: Double(samplerNanos) / 1_000_000_000.0,
            nanocodecSeconds: nanocodecSeconds)

        return MagpieSynthesisResult(
            samples: samples,
            sampleRate: MagpieConstants.audioSampleRate,
            codeCount: numFrames,
            finishedOnEos: finishedOnEos,
            timings: timings)
    }

    // MARK: - Model runners

    private func runTextEncoder(
        tokenized: MagpieTokenizedText, maxTextLen: Int, model: MLModel
    ) throws -> (encoderOutput: MLMultiArray, encoderMask: MLMultiArray) {
        let tokenArr = try MLMultiArray(
            shape: [1, NSNumber(value: maxTextLen)], dataType: .int32)
        tokenArr.withUnsafeMutableBytes { ptr, _ in
            let base = ptr.bindMemory(to: Int32.self).baseAddress!
            for i in 0..<maxTextLen { base[i] = tokenized.paddedIds[i] }
        }
        let maskArr = try MLMultiArray(
            shape: [1, NSNumber(value: maxTextLen)], dataType: .float32)
        maskArr.withUnsafeMutableBytes { ptr, _ in
            let base = ptr.bindMemory(to: Float.self).baseAddress!
            for i in 0..<maxTextLen { base[i] = tokenized.mask[i] }
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "text_tokens": MLFeatureValue(multiArray: tokenArr),
            "text_mask": MLFeatureValue(multiArray: maskArr),
        ])
        let out = try model.prediction(from: provider)
        guard let encoderOutput = out.featureValue(for: "encoder_output")?.multiArrayValue else {
            throw MagpieError.inferenceFailed(
                stage: "text_encoder", underlying: "missing encoder_output key")
        }
        return (encoderOutput, maskArr)
    }

    private func runDecoderStep(
        audioEmbed: MLMultiArray,
        encoderOutput: MLMultiArray,
        encoderMask: MLMultiArray,
        cache: MagpieKvCache,
        hiddenBacking: MLMultiArray,
        dModel: Int,
        model: MLModel
    ) throws -> [Float] {
        var inputs: [String: MLMultiArray] = [
            "audio_embed": audioEmbed,
            "encoder_output": encoderOutput,
            "encoder_mask": encoderMask,
        ]
        cache.addInputs(to: &inputs)
        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })

        // Bind every output to a pre-allocated MLMultiArray so CoreML writes
        // in place instead of allocating ~18.9 MB of fresh fp16 buffers per
        // step. The cache provides 24 K/V + 12 position back-buffers, the
        // synthesizer provides the 1 hidden buffer. After the call,
        // `swapBackings` promotes back→front for the next step's inputs.
        var backings: [String: Any] = [:]
        cache.addOutputBackings(to: &backings)
        backings[MagpieKvCache.decoderHiddenKey] = hiddenBacking
        let predOpts = MLPredictionOptions()
        predOpts.outputBackings = backings

        _ = try model.prediction(from: provider, options: predOpts)
        cache.swapBackings()

        // Hidden state lives in `hiddenBacking` after the call. Convert fp16
        // → fp32 via vImage into a fresh [Float] result buffer (the sampler
        // wants `[Float]`).
        let dim = dModel  // hiddenBacking shape = [1, 1, dModel]
        var result = Swift.Array<Float>(repeating: 0, count: dim)
        hiddenBacking.withUnsafeBytes { raw in
            guard let src = raw.baseAddress else { return }
            var srcBuffer = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: src),
                height: 1, width: vImagePixelCount(dim), rowBytes: dim * 2)
            result.withUnsafeMutableBufferPointer { dst in
                var dstBuffer = vImage_Buffer(
                    data: dst.baseAddress, height: 1,
                    width: vImagePixelCount(dim), rowBytes: dim * 4)
                _ = vImageConvert_Planar16FtoPlanarF(&srcBuffer, &dstBuffer, 0)
            }
        }
        return result
    }

    /// In-place mean-of-codebook-embeddings into a pre-allocated MLMultiArray.
    /// Replaces the per-step alloc + manual loop with vDSP primitives:
    ///   - `vDSP_vclr` zeros the buffer.
    ///   - `vDSP_vadd` accumulates each codebook's embedding row.
    ///   - `vDSP_vsmul` applies the `1 / numCodebooks` scale.
    private func fillAudioEmbed(
        _ arr: MLMultiArray, codes: [Int32], tables: [[Float]], dModel: Int
    ) throws {
        arr.withUnsafeMutableBytes { ptr, _ in
            guard let base = ptr.bindMemory(to: Float.self).baseAddress else { return }
            vDSP_vclr(base, 1, vDSP_Length(dModel))
            let numCodebooks = codes.count
            for cb in 0..<numCodebooks {
                let row = Int(codes[cb])
                tables[cb].withUnsafeBufferPointer { tablePtr in
                    guard let tableBase = tablePtr.baseAddress else { return }
                    let rowPtr = tableBase.advanced(by: row * dModel)
                    vDSP_vadd(base, 1, rowPtr, 1, base, 1, vDSP_Length(dModel))
                }
            }
            var inv = 1.0 / Float(numCodebooks)
            vDSP_vsmul(base, 1, &inv, base, 1, vDSP_Length(dModel))
        }
    }

}
