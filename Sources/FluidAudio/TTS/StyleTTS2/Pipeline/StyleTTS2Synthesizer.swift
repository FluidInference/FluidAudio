@preconcurrency import CoreML
import Foundation

/// End-to-end StyleTTS2 synthesizer driving the 4-stage diffusion pipeline.
///
/// Mirrors `mobius-styletts2/scripts/99b_e2e_coreml.py` step-for-step:
///
///   1. `text_predictor` (per token bucket)
///        in:  tokens[1, B_tok], style[1, 128]
///        out: t_en[1, 512, B_tok], d[1, B_tok, 640],
///             pred_dur_log[1, B_tok, 50], bert_dur[1, B_tok, 768]
///   2. ADPM2 sampler over `diffusion_step_512` (B=512 always)
///        in:  x_noisy[1,1,256], sigma[1], embedding[1,512,768], features[1,256]
///        out: denoised[1,1,256]
///   3. Style mix:
///        ref = α·s_pred[:,:128] + (1-α)·ref_s[:,:128]   // acoustic (decoder)
///        s   = β·s_pred[:,128:] + (1-β)·ref_s[:,128:]   // prosody  (f0n_energy)
///   4. Hard alignment from durations + gather-style matmul → en, asr.
///      HiFi-GAN right-shift by 1 frame.
///   5. `f0n_energy` (dynamic; padded to mel bucket)
///        in:  en[1,640,B_mel], s[1,128]
///        out: F0[1, 2*B_mel], N[1, 2*B_mel]
///   6. `decoder` (per mel bucket)
///        in:  asr[1,512,B_mel], F0_curve[1,2*B_mel], N[1,2*B_mel], s[1,128]
///        out: waveform[B_mel*600]
///   7. Trim to `T_mel*600` then drop the trailing 50 samples (decoder edge).
///
/// `synthesize` returns a fully-formed 24 kHz mono 16-bit PCM WAV blob.
public actor StyleTTS2Synthesizer {

    private let logger = AppLogger(category: "StyleTTS2Synthesizer")
    private let modelStore: StyleTTS2ModelStore

    public init(modelStore: StyleTTS2ModelStore) {
        self.modelStore = modelStore
    }

    /// Synthesis options. Defaults match the upstream Python reference.
    public struct Options: Sendable {
        public var diffusionSteps: Int
        public var alpha: Float
        public var beta: Float
        public var randomSeed: UInt64?

        public init(
            diffusionSteps: Int = StyleTTS2Constants.defaultDiffusionSteps,
            alpha: Float = 0.3,
            beta: Float = 0.7,
            randomSeed: UInt64? = nil
        ) {
            self.diffusionSteps = diffusionSteps
            self.alpha = alpha
            self.beta = beta
            self.randomSeed = randomSeed
        }
    }

    /// Synthesize from already-encoded token ids and a precomputed voice style.
    ///
    /// Caller is responsible for phonemization + vocab encoding (use
    /// `StyleTTS2Manager.tokenize`). The leading pad token (id 0) is inserted
    /// here per the upstream contract.
    public func synthesize(
        ids: [Int32],
        voice: StyleTTS2VoiceStyle,
        options: Options = Options()
    ) async throws -> Data {
        let samples = try await synthesizeSamples(ids: ids, voice: voice, options: options)
        return try AudioWAV.data(
            from: samples, sampleRate: Double(StyleTTS2Constants.audioSampleRate))
    }

    /// Same as `synthesize` but returns raw fp32 PCM samples.
    public func synthesizeSamples(
        ids: [Int32],
        voice: StyleTTS2VoiceStyle,
        options: Options = Options()
    ) async throws -> [Float] {

        // ---- Frontend: prepend pad, pick token bucket, pad. ----
        var paddedIds: [Int32] = [Int32(StyleTTS2Constants.padTokenId)]
        paddedIds.append(contentsOf: ids)
        let tTok = paddedIds.count
        let bTok = modelStore.textPredictorBucket(forTokenLength: tTok)
        guard tTok <= bTok else {
            throw StyleTTS2Error.processingFailed(
                "input length \(tTok) exceeds largest text_predictor bucket \(bTok)")
        }

        // ---- Stage A: text_predictor ----
        let aOut = try await runTextPredictor(
            tokens: paddedIds,
            tokenBucket: bTok,
            voice: voice
        )
        // Slice T_tok-prefix views.
        let tEn = aOut.tEn  // shape (512, T_tok), row-major
        let dFull = aOut.dFull  // shape (T_tok, 640)
        let predDurLog = aOut.predDurLog  // shape (T_tok, 50)
        let bertDur = aOut.bertDur  // shape (T_tok, 768)

        // ---- Durations + T_mel ----
        let durations = computeDurations(predDurLog: predDurLog, tTok: tTok)
        let tMel = durations.reduce(0, +)
        guard tMel >= 1 else {
            throw StyleTTS2Error.processingFailed("predicted T_mel was 0")
        }
        let bMel = modelStore.decoderBucket(forMelFrames: tMel)
        guard tMel <= bMel else {
            throw StyleTTS2Error.processingFailed(
                "T_mel \(tMel) exceeds largest decoder bucket \(bMel)")
        }

        // ---- Stage B: ADPM2 sampler over diffusion_step_512 ----
        let noise = generateGaussianNoise(
            count: StyleTTS2Constants.refStyleDim, seed: options.randomSeed)
        let sPred = try await runDiffusionSampler(
            noise: noise,
            bertDur: bertDur,
            tTok: tTok,
            voice: voice,
            steps: options.diffusionSteps
        )

        // ---- Style mix ----
        let acousticOriginal = Array(voice.acoustic)  // ref_s[:,:128]
        let prosodyOriginal = Array(voice.prosody)  // ref_s[:,128:]
        let acousticPred = Array(sPred[0..<StyleTTS2Constants.styleDim])
        let prosodyPred = Array(sPred[StyleTTS2Constants.styleDim..<StyleTTS2Constants.refStyleDim])
        let acousticMix = mix(a: acousticPred, b: acousticOriginal, alpha: options.alpha)
        let prosodyMix = mix(a: prosodyPred, b: prosodyOriginal, alpha: options.beta)

        // ---- Alignment + gather ----
        let tokenIndex = buildTokenIndex(durations: durations, tMel: tMel)
        var en = gatherEn(dFull: dFull, tokenIndex: tokenIndex, tMel: tMel)
        var asr = gatherAsr(tEn: tEn, tokenIndex: tokenIndex, tMel: tMel)

        // HiFi-GAN right-shift by 1 frame on both en and asr.
        rightShift(channels: 640, frames: tMel, buffer: &en)
        rightShift(channels: 512, frames: tMel, buffer: &asr)

        // ---- Stage C: f0n_energy ----
        let f0nOut = try await runF0nEnergy(
            en: en, tMel: tMel, melBucket: bMel, prosodyMix: prosodyMix)
        let f0 = f0nOut.f0  // length 2*T_mel
        let n = f0nOut.n  // length 2*T_mel

        // ---- Stage D: decoder ----
        let waveform = try await runDecoder(
            asr: asr,
            f0: f0,
            n: n,
            acousticMix: acousticMix,
            tMel: tMel,
            melBucket: bMel
        )

        // Trim: take T_mel*600 samples then drop the trailing 50 (decoder edge).
        let lastSample = min(waveform.count, tMel * 600)
        let trimmed: [Float]
        if lastSample > 50 {
            trimmed = Array(waveform[0..<(lastSample - 50)])
        } else {
            trimmed = Array(waveform[0..<lastSample])
        }
        logger.info(
            "Synthesized \(tTok) tokens → T_mel=\(tMel) → \(trimmed.count) samples "
                + "(\(String(format: "%.2f", Double(trimmed.count) / 24_000.0))s)")
        return trimmed
    }

    // MARK: - Stage A: text_predictor

    private struct TextPredictorOutput {
        let tEn: [Float]  // (512, T_tok), row-major
        let dFull: [Float]  // (T_tok, 640)
        let predDurLog: [Float]  // (T_tok, 50)
        let bertDur: [Float]  // (T_tok, 768)
    }

    private func runTextPredictor(
        tokens: [Int32],
        tokenBucket: Int,
        voice: StyleTTS2VoiceStyle
    ) async throws -> TextPredictorOutput {
        let tTok = tokens.count
        let model = try await modelStore.textPredictor(bucket: tokenBucket)

        // tokens: (1, B_tok) int32, padded with 0.
        let tokensArr = try MLMultiArray(
            shape: [1, NSNumber(value: tokenBucket)], dataType: .int32)
        let tokensPtr = tokensArr.dataPointer.bindMemory(to: Int32.self, capacity: tokensArr.count)
        tokensPtr.initialize(repeating: 0, count: tokensArr.count)
        for i in 0..<tTok {
            tokensPtr[i] = tokens[i]
        }

        // style: (1, 128) f32 — acoustic half of ref_s.
        let styleArr = try MLMultiArray(
            shape: [1, NSNumber(value: StyleTTS2Constants.styleDim)],
            dataType: .float32)
        let stylePtr = styleArr.dataPointer.bindMemory(to: Float.self, capacity: styleArr.count)
        for (i, v) in voice.acoustic.enumerated() {
            stylePtr[i] = v
        }

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: tokensArr),
            "style": MLFeatureValue(multiArray: styleArr),
        ])
        let prediction = try await model.prediction(from: inputs)

        guard let tEnArr = prediction.featureValue(for: "t_en")?.multiArrayValue,
            let dArr = prediction.featureValue(for: "d")?.multiArrayValue,
            let dlArr = prediction.featureValue(for: "pred_dur_log")?.multiArrayValue,
            let bArr = prediction.featureValue(for: "bert_dur")?.multiArrayValue
        else {
            throw StyleTTS2Error.processingFailed(
                "text_predictor: missing one of t_en/d/pred_dur_log/bert_dur outputs")
        }

        // t_en is (1, 512, B_tok). Take the first T_tok columns.
        let tEn = sliceFirstAxis2D(
            arr: tEnArr, leading: 512, trailing: tokenBucket, take: tTok, sliceDim: .trailing)
        // d is (1, B_tok, 640). Take the first T_tok rows.
        let dFull = sliceFirstAxis2D(
            arr: dArr, leading: tokenBucket, trailing: 640, take: tTok, sliceDim: .leading)
        let predDurLog = sliceFirstAxis2D(
            arr: dlArr, leading: tokenBucket, trailing: 50, take: tTok, sliceDim: .leading)
        let bertDur = sliceFirstAxis2D(
            arr: bArr, leading: tokenBucket, trailing: 768, take: tTok, sliceDim: .leading)

        return TextPredictorOutput(
            tEn: tEn, dFull: dFull, predDurLog: predDurLog, bertDur: bertDur)
    }

    private enum SliceDim { case leading, trailing }

    /// Slice an MLMultiArray of shape `(1, leading, trailing)` to the first
    /// `take` entries along either the leading or trailing axis. Returns a
    /// flat row-major `[Float]`.
    ///
    /// Reads via `dataPointer` instead of `arr[idx].floatValue` and avoids
    /// `arr.strides` entirely — both trigger
    /// `E5RT: tensor_buffer has known strides while the model has
    /// FlexibleShapeInfo` on `text_predictor`'s flex-shape outputs. CoreML
    /// emits dense row-major buffers, so for shape `(1, leading, trailing)`
    /// the flat index is simply `r * trailing + c`.
    private func sliceFirstAxis2D(
        arr: MLMultiArray,
        leading: Int,
        trailing: Int,
        take: Int,
        sliceDim: SliceDim
    ) -> [Float] {
        let outCount: Int
        switch sliceDim {
        case .leading: outCount = take * trailing
        case .trailing: outCount = leading * take
        }
        var out = [Float](repeating: 0, count: outCount)

        func fill(_ get: (Int) -> Float) {
            switch sliceDim {
            case .leading:
                // Result shape: (take, trailing).
                for r in 0..<take {
                    for c in 0..<trailing {
                        out[r * trailing + c] = get(r * trailing + c)
                    }
                }
            case .trailing:
                // Result shape: (leading, take).
                for r in 0..<leading {
                    for c in 0..<take {
                        out[r * take + c] = get(r * trailing + c)
                    }
                }
            }
        }

        let count = arr.count
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            fill { p[$0] }
        case .float16:
            let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: count)
            fill { Float(p[$0]) }
        case .double:
            let p = arr.dataPointer.bindMemory(to: Double.self, capacity: count)
            fill { Float(p[$0]) }
        default:
            // Fallback re-introduces the FlexibleShapeInfo trip wire, but
            // we don't expect text_predictor to emit anything other than
            // fp16/fp32.
            fill { arr[$0].floatValue }
        }
        return out
    }

    // MARK: - Durations

    /// `pred_dur = round(sigmoid(pred_dur_log).sum(-1)).clamp(min=1)`.
    private func computeDurations(predDurLog: [Float], tTok: Int) -> [Int] {
        var durations = [Int](repeating: 0, count: tTok)
        let dimDur = 50
        for i in 0..<tTok {
            var sum: Float = 0
            for k in 0..<dimDur {
                let logit = predDurLog[i * dimDur + k]
                sum += sigmoid(logit)
            }
            durations[i] = max(1, Int(sum.rounded()))
        }
        return durations
    }

    private func sigmoid(_ x: Float) -> Float {
        return 1.0 / (1.0 + exp(-x))
    }

    // MARK: - Stage B: diffusion sampler

    private func runDiffusionSampler(
        noise: [Float],
        bertDur: [Float],  // (T_tok, 768)
        tTok: Int,
        voice: StyleTTS2VoiceStyle,
        steps: Int
    ) async throws -> [Float] {

        let model = try await modelStore.diffusionStep()

        // Build embedding: (1, 512, 768). Pad bert_dur from (T_tok, 768) to
        // (512, 768). The diffusion model is shipped with B=512 only.
        let embeddingDim = 768
        let diffBucket = StyleTTS2Constants.diffusionBucket
        let embeddingArr = try MLMultiArray(
            shape: [1, NSNumber(value: diffBucket), NSNumber(value: embeddingDim)],
            dataType: .float32)
        let embPtr = embeddingArr.dataPointer.bindMemory(
            to: Float.self, capacity: embeddingArr.count)
        embPtr.initialize(repeating: 0, count: embeddingArr.count)
        for r in 0..<tTok {
            for c in 0..<embeddingDim {
                embPtr[r * embeddingDim + c] = bertDur[r * embeddingDim + c]
            }
        }
        let embeddingFeature = MLFeatureValue(multiArray: embeddingArr)

        // Build features: (1, 256) — full ref_s.
        let featuresArr = try MLMultiArray(
            shape: [1, NSNumber(value: StyleTTS2Constants.refStyleDim)],
            dataType: .float32)
        let featPtr = featuresArr.dataPointer.bindMemory(
            to: Float.self, capacity: featuresArr.count)
        for (i, v) in voice.concatenated.enumerated() {
            featPtr[i] = v
        }
        let featuresFeature = MLFeatureValue(multiArray: featuresArr)

        // Closure: (x_noisy[256], sigma) → denoised[256].
        let denoise: StyleTTS2Sampler.DenoiseStep = { x, sigma in
            // x_noisy: (1, 1, 256) f32
            let xArr = try MLMultiArray(
                shape: [1, 1, NSNumber(value: StyleTTS2Constants.refStyleDim)],
                dataType: .float32)
            let xPtr = xArr.dataPointer.bindMemory(to: Float.self, capacity: xArr.count)
            for (i, v) in x.enumerated() {
                xPtr[i] = v
            }
            // sigma: (1,) f32
            let sigmaArr = try MLMultiArray(shape: [1], dataType: .float32)
            sigmaArr.dataPointer.bindMemory(to: Float.self, capacity: 1)[0] = sigma

            let inputs = try MLDictionaryFeatureProvider(dictionary: [
                "x_noisy": MLFeatureValue(multiArray: xArr),
                "sigma": MLFeatureValue(multiArray: sigmaArr),
                "embedding": embeddingFeature,
                "features": featuresFeature,
            ])
            let prediction = try await model.prediction(from: inputs)
            guard let denoisedArr = prediction.featureValue(for: "denoised")?.multiArrayValue
            else {
                throw StyleTTS2Error.processingFailed("diffusion_step: missing denoised output")
            }
            // `denoised` ships as Float16 per the model schema.
            return await self.readMLMultiArrayPrefix(
                arr: denoisedArr, count: StyleTTS2Constants.refStyleDim)
        }

        return try await StyleTTS2Sampler.adpm2Sample(
            steps: steps, noise: noise, denoise: denoise)
    }

    // MARK: - Style mix

    private func mix(a: [Float], b: [Float], alpha: Float) -> [Float] {
        precondition(a.count == b.count)
        var out = [Float](repeating: 0, count: a.count)
        for i in 0..<a.count {
            out[i] = alpha * a[i] + (1.0 - alpha) * b[i]
        }
        return out
    }

    // MARK: - Alignment + gather

    /// Build a `tokenIndex[t_mel] = t_tok` lookup from per-token durations.
    /// Equivalent to the one-hot `aln[T_tok, T_mel]` matrix the Python ref
    /// constructs, but as a flat lookup (avoids materializing the full
    /// matrix).
    private func buildTokenIndex(durations: [Int], tMel: Int) -> [Int] {
        var idx = [Int](repeating: 0, count: tMel)
        var cursor = 0
        for (t, n) in durations.enumerated() {
            for k in 0..<n where cursor + k < tMel {
                idx[cursor + k] = t
            }
            cursor += n
        }
        return idx
    }

    /// `en[c, t_mel] = d_full[token_index[t_mel], c]`. Shape (640, T_mel),
    /// row-major.
    private func gatherEn(dFull: [Float], tokenIndex: [Int], tMel: Int) -> [Float] {
        let channels = 640
        var en = [Float](repeating: 0, count: channels * tMel)
        for t in 0..<tMel {
            let tok = tokenIndex[t]
            for c in 0..<channels {
                en[c * tMel + t] = dFull[tok * channels + c]
            }
        }
        return en
    }

    /// `asr[c, t_mel] = t_en[c, token_index[t_mel]]`. Shape (512, T_mel),
    /// row-major.
    private func gatherAsr(tEn: [Float], tokenIndex: [Int], tMel: Int) -> [Float] {
        let channels = 512
        let tTok = tEn.count / channels
        var asr = [Float](repeating: 0, count: channels * tMel)
        for t in 0..<tMel {
            let tok = tokenIndex[t]
            for c in 0..<channels {
                asr[c * tMel + t] = tEn[c * tTok + tok]
            }
        }
        return asr
    }

    /// HiFi-GAN convention: shift right by 1 frame and duplicate frame 0.
    /// `out[c, 0] = in[c, 0]; out[c, t] = in[c, t-1]` for t >= 1.
    private func rightShift(channels: Int, frames: Int, buffer: inout [Float]) {
        guard frames >= 2 else { return }
        // Shift in place from right to left.
        for c in 0..<channels {
            let base = c * frames
            for t in stride(from: frames - 1, through: 1, by: -1) {
                buffer[base + t] = buffer[base + t - 1]
            }
            // buffer[base + 0] is unchanged → duplicate of original frame 0.
        }
    }

    // MARK: - Stage C: f0n_energy

    private struct F0NOutput {
        let f0: [Float]
        let n: [Float]
    }

    private func runF0nEnergy(
        en: [Float], tMel: Int, melBucket: Int, prosodyMix: [Float]
    ) async throws -> F0NOutput {
        let model = try await modelStore.f0nEnergy()
        let channels = 640

        // en_pad: (1, 640, 4096) — always use the largest enumerated shape.
        // f0n_energy ships with enumeratedShapes; using only the default
        // shape sidesteps E5RT's "tensor_buffer has known strides while
        // the model has FlexibleShapeInfo" check that fires when the
        // chosen enumeration differs from the default.
        let enBucket = StyleTTS2Constants.decoderBuckets.last ?? 4096
        let enArr = try MLMultiArray(
            shape: [
                1, NSNumber(value: channels), NSNumber(value: enBucket),
            ],
            dataType: .float32)
        let enPtr = enArr.dataPointer.bindMemory(to: Float.self, capacity: enArr.count)
        enPtr.initialize(repeating: 0, count: enArr.count)
        for c in 0..<channels {
            for t in 0..<tMel {
                enPtr[c * enBucket + t] = en[c * tMel + t]
            }
        }

        // s: (1, 128) — prosody half (post-mix).
        let sArr = try MLMultiArray(
            shape: [1, NSNumber(value: StyleTTS2Constants.styleDim)],
            dataType: .float32)
        let sPtr = sArr.dataPointer.bindMemory(to: Float.self, capacity: sArr.count)
        for (i, v) in prosodyMix.enumerated() {
            sPtr[i] = v
        }

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "en": MLFeatureValue(multiArray: enArr),
            "s": MLFeatureValue(multiArray: sArr),
        ])
        let prediction = try await model.prediction(from: inputs)
        guard let f0Arr = prediction.featureValue(for: "F0")?.multiArrayValue,
            let nArr = prediction.featureValue(for: "N")?.multiArrayValue
        else {
            throw StyleTTS2Error.processingFailed("f0n_energy: missing F0/N output")
        }

        // F0 and N are (1, 2*B_mel). Slice to first 2*T_mel.
        // Storage dtype is Float16 per the model schema.
        let outLen = 2 * tMel
        let f0 = readMLMultiArrayPrefix(arr: f0Arr, count: outLen)
        let n = readMLMultiArrayPrefix(arr: nArr, count: outLen)
        return F0NOutput(f0: f0, n: n)
    }

    /// Read the first `count` scalars from an MLMultiArray, regardless of
    /// whether the storage dtype is Float32 or Float16.
    private func readMLMultiArrayPrefix(arr: MLMultiArray, count: Int) -> [Float] {
        let n = min(count, arr.count)
        var out = [Float](repeating: 0, count: n)
        switch arr.dataType {
        case .float32:
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: arr.count)
            for i in 0..<n {
                out[i] = p[i]
            }
        case .float16:
            let p = arr.dataPointer.bindMemory(to: Float16.self, capacity: arr.count)
            for i in 0..<n {
                out[i] = Float(p[i])
            }
        case .double:
            let p = arr.dataPointer.bindMemory(to: Double.self, capacity: arr.count)
            for i in 0..<n {
                out[i] = Float(p[i])
            }
        default:
            for i in 0..<n {
                out[i] = arr[i].floatValue
            }
        }
        return out
    }

    // MARK: - Stage D: decoder

    private func runDecoder(
        asr: [Float],
        f0: [Float],
        n: [Float],
        acousticMix: [Float],
        tMel: Int,
        melBucket: Int
    ) async throws -> [Float] {
        let model = try await modelStore.decoder(bucket: melBucket)

        // asr_pad: (1, 512, B_mel)
        let asrArr = try MLMultiArray(
            shape: [1, 512, NSNumber(value: melBucket)], dataType: .float32)
        let asrPtr = asrArr.dataPointer.bindMemory(to: Float.self, capacity: asrArr.count)
        asrPtr.initialize(repeating: 0, count: asrArr.count)
        for c in 0..<512 {
            for t in 0..<tMel {
                asrPtr[c * melBucket + t] = asr[c * tMel + t]
            }
        }

        // F0_curve: (1, 2*B_mel)
        let f0Arr = try MLMultiArray(
            shape: [1, NSNumber(value: 2 * melBucket)], dataType: .float32)
        let f0Ptr = f0Arr.dataPointer.bindMemory(to: Float.self, capacity: f0Arr.count)
        f0Ptr.initialize(repeating: 0, count: f0Arr.count)
        for i in 0..<f0.count {
            f0Ptr[i] = f0[i]
        }

        // N: (1, 2*B_mel)
        let nArr = try MLMultiArray(
            shape: [1, NSNumber(value: 2 * melBucket)], dataType: .float32)
        let nPtr = nArr.dataPointer.bindMemory(to: Float.self, capacity: nArr.count)
        nPtr.initialize(repeating: 0, count: nArr.count)
        for i in 0..<n.count {
            nPtr[i] = n[i]
        }

        // s: (1, 128) — acoustic mix.
        let sArr = try MLMultiArray(
            shape: [1, NSNumber(value: StyleTTS2Constants.styleDim)], dataType: .float32)
        let sPtr = sArr.dataPointer.bindMemory(to: Float.self, capacity: sArr.count)
        for (i, v) in acousticMix.enumerated() {
            sPtr[i] = v
        }

        let inputs = try MLDictionaryFeatureProvider(dictionary: [
            "asr": MLFeatureValue(multiArray: asrArr),
            "F0_curve": MLFeatureValue(multiArray: f0Arr),
            "N": MLFeatureValue(multiArray: nArr),
            "s": MLFeatureValue(multiArray: sArr),
        ])
        let prediction = try await model.prediction(from: inputs)
        guard let waveArr = prediction.featureValue(for: "waveform")?.multiArrayValue else {
            throw StyleTTS2Error.processingFailed("decoder: missing waveform output")
        }
        let count = waveArr.count
        let wPtr = waveArr.dataPointer.bindMemory(to: Float.self, capacity: count)
        var waveform = [Float](repeating: 0, count: count)
        for i in 0..<count {
            waveform[i] = wPtr[i]
        }
        return waveform
    }

    // MARK: - Noise generator

    /// Box-Muller Gaussian noise. Deterministic when `seed` is non-nil
    /// (SplitMix64 PRNG → uniform → Box-Muller); otherwise uses
    /// `SystemRandomNumberGenerator`.
    private func generateGaussianNoise(count: Int, seed: UInt64?) -> [Float] {
        var values = [Float](repeating: 0, count: count)
        if var rng = seed.map({ SplitMix64(seed: $0) }) {
            fillBoxMuller(into: &values, generator: { rng.nextUnitFloat() })
        } else {
            var sys = SystemRandomNumberGenerator()
            fillBoxMuller(
                into: &values,
                generator: {
                    Float.random(in: 0..<1, using: &sys)
                })
        }
        return values
    }

    private func fillBoxMuller(into values: inout [Float], generator: () -> Float) {
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
}

/// Tiny SplitMix64 PRNG. Used purely to seed Gaussian noise — not for
/// cryptographic or statistical applications. State is `Sendable` because
/// it's used inside an actor.
private struct SplitMix64 {
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

    /// Uniform float in [0, 1).
    mutating func nextUnitFloat() -> Float {
        // Take the top 24 bits → exact representation in fp32 mantissa.
        let bits = next() >> 40
        return Float(bits) / Float(1 << 24)
    }
}
