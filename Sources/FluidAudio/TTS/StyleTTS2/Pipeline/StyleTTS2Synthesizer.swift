@preconcurrency import CoreML
import Foundation

/// Orchestrates the 7-stage StyleTTS2-ANE CoreML chain.
///
/// Mirrors `mobius/.../scripts/ane/_styletts2_ane_lib.py` step-for-step:
///
///   1. PLBert        tokens [1,T_tok] → bert_dur [1,T_tok,768]
///   2. PostBert      bert_dur, tokens, ref_s[1,256] →
///                    t_en[1,512,T_tok], d[1,T_tok,640],
///                    pred_dur_log[1,T_tok,50], fixed_embedding[1,T_tok,768]
///   3. Alignment     pred_dur[1,T_tok], d, t_en →
///                    en[1,640,MAX_T_A], asr[1,512,MAX_T_A]   (zero-padded)
///   4. DiffusionStep x_noisy[1,1,256], sigma[1],
///                    embedding[1,512,768], features[1,256] → denoised[1,1,256]
///                    Looped 11× by the ADPM2 sampler in Swift
///                    (5 midpoint × 2 + 1 final).
///   5. Prosody       en[1,640,MAX_T_A], s_pros[1,128] → F0[1,MAX_T_A], N[1,MAX_T_A]
///   6. Noise (fp32)  F0[1,MAX_T_A] → sine_waves[1,T_audio,harm+1], uv[1,MAX_T_A,1]
///   7. Vocoder       asr, F0, N, ref_s_mixed[1,256], sine_waves → audio[1,T_audio]
///
/// Stages 4-7 are *fixed* shape (T_a padded to MAX_T_A=2000); stages 1-3 are
/// `RangeDim(2..512)` over T_tok. The host pads tokens / acoustic frames to
/// the bucket and trims the audio output to `T_a * 2 * UPSAMPLE_SCALE` after
/// Vocoder.
///
/// fp32/fp16 boundaries follow the export-script choices:
///   * Stages 1,2,3,4,5,7 — fp16 (ANE-resident)
///   * Stage 6 (Noise)    — fp32 (SineGen phase precision)
///
/// The ADPM2 + Karras sampler is reused verbatim from `StyleTTS2Sampler` —
/// it's model-agnostic and takes a closure for the actual denoise call.
public struct StyleTTS2Synthesizer {

    /// Synthesis options. Defaults match the upstream Python reference.
    public struct Options: Sendable {
        public var diffusionSteps: Int
        public var alpha: Float
        public var beta: Float
        public var randomSeed: UInt64?

        public init(
            diffusionSteps: Int = StyleTTS2Constants.defaultDiffusionSteps,
            alpha: Float = StyleTTS2Constants.defaultAlpha,
            beta: Float = StyleTTS2Constants.defaultBeta,
            randomSeed: UInt64? = nil
        ) {
            self.diffusionSteps = diffusionSteps
            self.alpha = alpha
            self.beta = beta
            self.randomSeed = randomSeed
        }
    }

    /// One-shot synthesis.
    public static func synthesize(
        ids: [Int32],
        voice: StyleTTS2VoiceStyle,
        options: Options,
        store: StyleTTS2ModelStore
    ) async throws -> StyleTTS2SynthesisResult {

        // Prepend pad token (id 0) per upstream contract.
        var paddedIds: [Int32] = [Int32(StyleTTS2Constants.padTokenId)]
        paddedIds.append(contentsOf: ids)
        let tTok = paddedIds.count
        guard tTok >= 2, tTok <= StyleTTS2Constants.maxInputTokens else {
            throw StyleTTS2Error.phonemeSequenceTooLong(tTok)
        }

        var timings = StyleTTS2StageTimings()

        // Build base tensors (tokens + style) reused by multiple stages.
        let tokensArr = try KokoroAneArrays.int32Array(shape: [1, tTok], from: paddedIds)
        let styleFullArr = try KokoroAneArrays.float16Array(
            shape: [1, StyleTTS2Constants.refStyleDim],
            from: voice.concatenated
        )

        // ── 1: PLBert ─────────────────────────────────────────────────
        let plBertModel = try await store.model(for: .plBert)
        let plBertOut = try await predict(
            stage: .plBert,
            model: plBertModel,
            inputs: ["tokens": tokensArr],
            timing: &timings.plBert
        )
        let bertDur = try rebuild16(plBertOut, key: "bert_dur", stage: .plBert)

        // ── 2: PostBert ───────────────────────────────────────────────
        let postBertModel = try await store.model(for: .postBert)
        let postBertOut = try await predict(
            stage: .postBert,
            model: postBertModel,
            inputs: [
                "bert_dur": bertDur,
                "tokens": tokensArr,
                "style": styleFullArr,
            ],
            timing: &timings.postBert
        )
        let tEn = try rebuild16(postBertOut, key: "t_en", stage: .postBert)
        let dArr = try rebuild16(postBertOut, key: "d", stage: .postBert)
        let predDurLog = try outputArray(postBertOut, key: "pred_dur_log", stage: .postBert)

        // Compute durations + T_a from pred_dur_log
        // (= round(sigmoid(logit).sum(-1)).clamp(min=1)).
        let durations = try computeDurations(arr: predDurLog, tTok: tTok)
        let tA = durations.reduce(0, +)
        guard tA >= 1 else {
            throw StyleTTS2Error.inputProcessingFailed("predicted T_a was 0")
        }
        if tA > StyleTTS2Constants.maxAcousticFrames {
            throw StyleTTS2Error.acousticFramesExceedCap(
                have: tA, cap: StyleTTS2Constants.maxAcousticFrames)
        }
        let predDurArr = try KokoroAneArrays.int32Array(
            shape: [1, tTok], from: durations.map(Int32.init))

        // ── 3: ADPM2 sampler over DiffusionStep ───────────────────────
        // Build embedding[1,512,768] padded from bert_dur. Static shape
        // mandated by the Stage 4 export.
        let embeddingPadded = try padBertDurToFixedEmbedding(bertDur: bertDur, tTok: tTok)
        let featuresFp16 = styleFullArr  // [1, 256] — full ref_s

        let noise = generateGaussianNoise(
            count: StyleTTS2Constants.refStyleDim, seed: options.randomSeed)
        let sPred = try await runDiffusionSampler(
            noise: noise,
            embedding: embeddingPadded,
            features: featuresFp16,
            steps: options.diffusionSteps,
            store: store,
            timing: &timings.diffusionStep
        )

        // ── 4: Style mix ──────────────────────────────────────────────
        // On-disk ref_s.bin layout (yl4579 convention, per
        // `mobius/.../scripts/06_dump_ref_s.py`):
        //   [0:128]   = style_encoder(mel)      →  s_acou   (Vocoder)
        //   [128:256] = predictor_encoder(mel)  →  s_pros   (Prosody, PostBert)
        //
        // FluidAudio's `StyleTTS2VoiceStyle.acoustic = .prefix(128)` and
        // `.prosody = .suffix(128)` accessors line up directly with the disk
        // layout. The diffusion sampler outputs in the same layout, so the
        // first half of `sPred` is the acoustic prediction and the second
        // half is the prosody prediction — matches the legacy 4-graph
        // backend exactly.
        let acousticOriginal = Array(voice.acoustic)  // [0:128] = s_acou
        let prosodyOriginal = Array(voice.prosody)  // [128:256] = s_pros
        let acousticPred = Array(sPred[0..<StyleTTS2Constants.styleDim])
        let prosodyPred = Array(
            sPred[StyleTTS2Constants.styleDim..<StyleTTS2Constants.refStyleDim])
        // alpha governs the acoustic blend, beta the prosody blend — same
        // weight assignment as the legacy backend.
        let acousticMix = mix(a: acousticPred, b: acousticOriginal, alpha: options.alpha)
        let prosodyMix = mix(a: prosodyPred, b: prosodyOriginal, alpha: options.beta)

        let prosodyMixFp16 = try KokoroAneArrays.float16Array(
            shape: [1, StyleTTS2Constants.styleDim], from: prosodyMix)
        let acousticMixFp16 = try KokoroAneArrays.float16Array(
            shape: [1, StyleTTS2Constants.styleDim], from: acousticMix)

        // ── 5: Alignment ──────────────────────────────────────────────
        let alignModel = try await store.model(for: .alignment)
        let alignOut = try await predict(
            stage: .alignment,
            model: alignModel,
            inputs: [
                "pred_dur": predDurArr,
                "d": dArr,
                "t_en": tEn,
            ],
            timing: &timings.alignment
        )
        let enArr = try rebuild16(alignOut, key: "en", stage: .alignment)
        let asrArr = try rebuild16(alignOut, key: "asr", stage: .alignment)

        // ── 6: Prosody ────────────────────────────────────────────────
        let prosodyModel = try await store.model(for: .prosody)
        let prosOut = try await predict(
            stage: .prosody,
            model: prosodyModel,
            inputs: [
                "en": enArr,
                "s": prosodyMixFp16,  // [1, 128] = s_pros (second half of ref_s)
            ],
            timing: &timings.prosody
        )
        let f0Raw = try outputArray(prosOut, key: "F0", stage: .prosody)
        let f0Shape = f0Raw.shape.map(\.intValue)
        let nRaw = try outputArray(prosOut, key: "N", stage: .prosody)
        let nShape = nRaw.shape.map(\.intValue)

        // ── 7: Noise (fp32 boundary) ──────────────────────────────────
        let f0F32 = try KokoroAneArrays.float32Array(shape: f0Shape, from: f0Raw)
        let noiseModel = try await store.model(for: .noise)
        let noiseOut = try await predict(
            stage: .noise,
            model: noiseModel,
            inputs: ["F0_curve": f0F32],
            timing: &timings.noise
        )

        // ── 8: Vocoder (fp16 boundary) — RangeDim, active-T_a ─────────
        // The Vocoder graph is RangeDim on T_a (default ≤ MAX_T_A) and the
        // upstream stages (Alignment, Prosody, Noise) are zero-padded to
        // MAX_T_A. We must slice every input to active T_a before predict;
        // otherwise the conv-transpose receptive field leaks across the
        // active/zero boundary and corrupts the audio (log-mel cos drops to
        // ~0.74 vs the PyTorch reference). Mirrors the Python
        // 99_e2e_validate.py active-T_a slicing exactly.
        let tAFrames = tA * 2  // F0/N are at MAX_T_A * 2
        let tAudio = tA * 2 * StyleTTS2Constants.upsampleScale  // sine_waves frames

        let asrSliced = try KokoroAneArrays.sliceTrailingTimeFp16(
            asrArr,
            channels: StyleTTS2Constants.hiddenDim,
            oldT: StyleTTS2Constants.maxAcousticFrames,
            newT: tA
        )

        // Skip the redundant fp16→fp16 memcpy when Prosody already
        // emitted fp16 (the common case). Slicing reads from the source
        // directly, so reusing `f0Raw`/`nRaw` is safe and saves one
        // MAX_T_A-sized allocation per call.
        let f0F16Source: MLMultiArray =
            f0Raw.dataType == .float16
            ? f0Raw : try KokoroAneArrays.float16Array(shape: f0Shape, from: f0Raw)
        let nF16Source: MLMultiArray =
            nRaw.dataType == .float16
            ? nRaw : try KokoroAneArrays.float16Array(shape: nShape, from: nRaw)
        let f0F16 = try KokoroAneArrays.sliceLeadingTimeFp16(
            f0F16Source, newShape: [1, tAFrames])
        let nF16 = try KokoroAneArrays.sliceLeadingTimeFp16(
            nF16Source, newShape: [1, tAFrames])

        let sineWavesF16Full = try rebuild16(noiseOut, key: "sine_waves", stage: .noise)
        let sineShapeFull = sineWavesF16Full.shape.map(\.intValue)
        // The downstream slice expects 3D `[1, T_audio, harm+1]`; reject
        // anything else loudly instead of falling through with a wrong
        // `harm+1` from the `?? 1` fallback.
        guard sineShapeFull.count == 3, let harmPlus1 = sineShapeFull.last else {
            throw StyleTTS2Error.unexpectedOutputShape(
                stage: StyleTTS2Stage.noise.rawValue,
                expected: "[1, T_audio, harm+1]",
                got: "\(sineShapeFull)"
            )
        }
        let sineWavesF16 = try KokoroAneArrays.sliceLeadingTimeFp16(
            sineWavesF16Full, newShape: [1, tAudio, harmPlus1])

        let vocoderModel = try await store.model(for: .vocoder)
        let vocOut = try await predict(
            stage: .vocoder,
            model: vocoderModel,
            inputs: [
                "asr": asrSliced,
                "F0_curve": f0F16,
                "N": nF16,
                "s": acousticMixFp16,  // [1, 128] = s_acou (first half of ref_s)
                "sine_waves": sineWavesF16,
            ],
            timing: &timings.vocoder
        )
        let audioArr = try outputArray(vocOut, key: "audio", stage: .vocoder)
        let samples = KokoroAneArrays.readFloats(audioArr)

        return StyleTTS2SynthesisResult(
            samples: samples,
            sampleRate: StyleTTS2Constants.sampleRate,
            encoderTokens: tTok,
            acousticFrames: tA,
            timings: timings
        )
    }

    // MARK: - Diffusion sampler

    private static func runDiffusionSampler(
        noise: [Float],
        embedding: MLMultiArray,
        features: MLMultiArray,
        steps: Int,
        store: StyleTTS2ModelStore,
        timing: inout Double
    ) async throws -> [Float] {
        let model = try await store.model(for: .diffusionStep)

        // Cumulative wall-clock across all 11 (5×2+1) invocations.
        let t0 = DispatchTime.now()

        let denoise: StyleTTS2Sampler.DenoiseStep = { x, sigma in
            // Each step is a full ANE round-trip; check cancellation
            // at the top so the 11-call loop doesn't keep grinding
            // after the caller has bailed.
            try Task.checkCancellation()
            let xArr = try KokoroAneArrays.float16Array(
                shape: [1, 1, StyleTTS2Constants.refStyleDim], from: x)
            let sigmaArr = try KokoroAneArrays.float16Array(
                shape: [1], from: [sigma])
            let provider = try MLDictionaryFeatureProvider(dictionary: [
                "x_noisy": MLFeatureValue(multiArray: xArr),
                "sigma": MLFeatureValue(multiArray: sigmaArr),
                "embedding": MLFeatureValue(multiArray: embedding),
                "features": MLFeatureValue(multiArray: features),
            ])
            let prediction: MLFeatureProvider
            do {
                prediction = try await model.prediction(from: provider)
            } catch {
                throw StyleTTS2Error.predictionFailed(
                    stage: StyleTTS2Stage.diffusionStep.rawValue, underlying: error)
            }
            guard let denoisedArr = prediction.featureValue(for: "denoised")?.multiArrayValue
            else {
                throw StyleTTS2Error.unexpectedOutputShape(
                    stage: StyleTTS2Stage.diffusionStep.rawValue,
                    expected: "MLMultiArray for 'denoised'",
                    got: "nil"
                )
            }
            return Array(
                KokoroAneArrays.readFloats(denoisedArr).prefix(
                    StyleTTS2Constants.refStyleDim))
        }

        let result = try await StyleTTS2Sampler.adpm2Sample(
            steps: steps, noise: noise, denoise: denoise)

        timing =
            Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds)
            / 1_000_000.0
        return result
    }

    // MARK: - Helpers

    /// Pad a `[1, T_tok, 768]` bert_dur into the static `[1, 512, 768]`
    /// embedding required by the Stage 4 DiffusionStep export. The trailing
    /// `512 - T_tok` rows stay zero.
    private static func padBertDurToFixedEmbedding(
        bertDur: MLMultiArray, tTok: Int
    ) throws -> MLMultiArray {
        let bertDim = 768
        let tEmbMax = StyleTTS2Constants.maxInputTokens  // 512
        // Validate the source layout before computing memcpy offsets — a
        // shape drift in PLBert would otherwise produce silently garbled
        // output rather than a localised failure.
        let srcShape = bertDur.shape.map(\.intValue)
        guard srcShape == [1, tTok, bertDim] else {
            throw StyleTTS2Error.unexpectedOutputShape(
                stage: StyleTTS2Stage.plBert.rawValue,
                expected: "[1, \(tTok), \(bertDim)]",
                got: "\(srcShape)"
            )
        }
        let dst = try MLMultiArray(
            shape: [1, NSNumber(value: tEmbMax), NSNumber(value: bertDim)],
            dataType: .float16)
        // Zero-fill explicitly (CoreML doesn't guarantee zero-init).
        let dstU16 = dst.dataPointer.bindMemory(to: UInt16.self, capacity: dst.count)
        dstU16.initialize(repeating: 0, count: dst.count)

        // Copy first T_tok rows from bert_dur. Source dtype is fp16 — same
        // storage width as dst — so byte-level memcpy works for that stripe.
        let copyRows = min(tTok, tEmbMax)
        let stride = bertDim  // both arrays row-major.
        let bytesPerRow = stride * MemoryLayout<UInt16>.size

        if bertDur.dataType == .float16 {
            let srcU16 = bertDur.dataPointer.bindMemory(to: UInt16.self, capacity: bertDur.count)
            for r in 0..<copyRows {
                memcpy(
                    dstU16.advanced(by: r * stride),
                    srcU16.advanced(by: r * stride),
                    bytesPerRow)
            }
        } else if bertDur.dataType == .float32 {
            // Fall back to per-element fp32 → fp16 conversion.
            let srcF32 = bertDur.dataPointer.bindMemory(to: Float.self, capacity: bertDur.count)
            var tmp = [Float](repeating: 0, count: copyRows * stride)
            for r in 0..<copyRows {
                for c in 0..<stride {
                    tmp[r * stride + c] = srcF32[r * stride + c]
                }
            }
            let srcArr = try KokoroAneArrays.float16Array(
                shape: [1, copyRows, stride], from: tmp)
            let srcU16 = srcArr.dataPointer.bindMemory(to: UInt16.self, capacity: srcArr.count)
            for r in 0..<copyRows {
                memcpy(
                    dstU16.advanced(by: r * stride),
                    srcU16.advanced(by: r * stride),
                    bytesPerRow)
            }
        } else {
            throw StyleTTS2Error.unexpectedOutputShape(
                stage: StyleTTS2Stage.plBert.rawValue,
                expected: "float16 or float32 bert_dur",
                got: "\(bertDur.dataType)"
            )
        }
        return dst
    }

    /// Compute per-token durations from `pred_dur_log[1, T_tok, 50]`.
    /// `pred_dur = round(sigmoid(logit).sum(-1)).clamp(min=1)`.
    private static func computeDurations(arr: MLMultiArray, tTok: Int) throws -> [Int] {
        let dimDur = 50
        let count = arr.count
        let total = tTok * dimDur
        // Throw at the source on shape mismatch instead of returning all
        // zeros and surfacing the failure later as a confusing
        // "predicted T_a was 0" — the real cause is a PostBert shape drift.
        guard count >= total else {
            throw StyleTTS2Error.unexpectedOutputShape(
                stage: StyleTTS2Stage.postBert.rawValue,
                expected: "pred_dur_log [1, \(tTok), \(dimDur)] (\(total) elements)",
                got: "count=\(count)"
            )
        }

        // Read logits as Float regardless of fp16/fp32 storage.
        var out = [Int](repeating: 0, count: tTok)
        let floats = KokoroAneArrays.readFloats(arr)
        for i in 0..<tTok {
            var sum: Float = 0
            for k in 0..<dimDur {
                let logit = floats[i * dimDur + k]
                sum += sigmoid(logit)
            }
            out[i] = max(1, Int(sum.rounded()))
        }
        return out
    }

    /// Numerically stable sigmoid. The naive `1 / (1 + exp(-x))` form
    /// underflows to zero and loses precision for very negative `x`
    /// (`expf` overflows around `-88`). Splitting the branch keeps both
    /// tails exact even if the model emits saturated logits.
    private static func sigmoid(_ x: Float) -> Float {
        if x >= 0 {
            return 1.0 / (1.0 + exp(-x))
        }
        let ex = exp(x)
        return ex / (1.0 + ex)
    }

    private static func mix(a: [Float], b: [Float], alpha: Float) -> [Float] {
        precondition(a.count == b.count)
        var out = [Float](repeating: 0, count: a.count)
        for i in 0..<a.count {
            out[i] = alpha * a[i] + (1.0 - alpha) * b[i]
        }
        return out
    }

    /// Box-Muller Gaussian noise, deterministic when `seed` is non-nil.
    private static func generateGaussianNoise(count: Int, seed: UInt64?) -> [Float] {
        var values = [Float](repeating: 0, count: count)
        if var rng = seed.map({ StyleTTS2SplitMix64(seed: $0) }) {
            fillBoxMuller(into: &values) { rng.nextUnitFloat() }
        } else {
            var sys = SystemRandomNumberGenerator()
            fillBoxMuller(into: &values) { Float.random(in: 0..<1, using: &sys) }
        }
        return values
    }

    private static func fillBoxMuller(into values: inout [Float], generator: () -> Float) {
        let count = values.count
        var i = 0
        while i < count {
            let u1 = max(generator(), Float.leastNormalMagnitude)
            let u2 = generator()
            let r = sqrt(-2.0 * log(u1))
            let theta = 2.0 * Float.pi * u2
            values[i] = r * cos(theta)
            if i + 1 < count {
                values[i + 1] = r * sin(theta)
            }
            i += 2
        }
    }

    // MARK: - CoreML plumbing

    private static func predict(
        stage: StyleTTS2Stage,
        model: MLModel,
        inputs: [String: MLMultiArray],
        timing: inout Double
    ) async throws -> MLFeatureProvider {
        // Cooperatively cancel between stages — long syntheses (large
        // T_a, many diffusion calls) can't preempt CoreML's
        // `prediction(from:)` itself, but checking at the stage
        // boundary makes the orchestrator drop out promptly when an
        // interactive caller cancels.
        try Task.checkCancellation()
        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let start = DispatchTime.now()
        do {
            let out = try await model.prediction(from: provider)
            timing =
                Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds)
                / 1_000_000.0
            return out
        } catch {
            throw StyleTTS2Error.predictionFailed(
                stage: stage.rawValue, underlying: error)
        }
    }

    private static func outputArray(
        _ provider: MLFeatureProvider, key: String, stage: StyleTTS2Stage
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw StyleTTS2Error.unexpectedOutputShape(
                stage: stage.rawValue,
                expected: "MLMultiArray for '\(key)'",
                got: "nil"
            )
        }
        return value
    }

    private static func rebuild16(
        _ provider: MLFeatureProvider, key: String, stage: StyleTTS2Stage
    ) throws -> MLMultiArray {
        let arr = try outputArray(provider, key: key, stage: stage)
        return try KokoroAneArrays.float16Array(shape: arr.shape.map(\.intValue), from: arr)
    }
}

// MARK: - Tiny SplitMix64 (deterministic noise)

/// Tiny SplitMix64 PRNG — used only to seed Gaussian noise. Deterministic
/// for a given seed so synthesis is reproducible.
private struct StyleTTS2SplitMix64 {
    private var state: UInt64

    init(seed: UInt64) {
        self.state = seed
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z &>> 31)
    }

    mutating func nextUnitFloat() -> Float {
        let bits = next() >> 40
        return Float(bits) / Float(1 << 24)
    }
}
