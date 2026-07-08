import Accelerate
@preconcurrency import CoreML
import Foundation

/// One LuxTTS synthesis run: 48 kHz mono samples plus the frame accounting
/// used by callers to report durations.
public struct LuxTtsSynthesisResult: Sendable {
    /// Generated waveform, 48 kHz mono Float32 in [-1, 1].
    public let samples: [Float]
    /// Always `LuxTtsConstants.outputSampleRate` (48 000).
    public let sampleRate: Int
    /// Prompt conditioning length in 24 kHz mel frames.
    public let promptFrames: Int
    /// Generated mel frames (`featuresLength - promptFrames`).
    public let generatedFrames: Int
    /// Total flow-matching sequence length (prompt + generated frames).
    public let featuresLength: Int
}

/// Drives the LuxTTS (ZipVoice-Distill) CoreML stages end-to-end, mirroring
/// the Python reference host (`coreml/dump_swift_fixtures.py::synth_coreml`):
///
///   1. `TextEncoder(tokens, padding_mask) → token_embeds`; keep `S + 1`
///      rows (the extra pad-slot row absorbs remainder frames).
///   2. Ratio duration estimate → `featuresLength`, avg-duration expansion
///      of token embeddings to per-frame `text_condition`.
///   3. `speech_condition` = prompt mel (feat-scaled) zero-padded to
///      `featuresLength`; frame `padding_mask` = 1.0 beyond `featuresLength`.
///   4. 4-step anchor-Euler flow matching over seeded N(0, 1) noise via
///      `FmDecoder` (distill guidance_scale baked input, 3.0).
///   5. Generated mel `/ featScale` → fixed-shape Vocos vocoder (282 / 555
///      frame bucket) → 48 kHz waveform, truncated to `(gen - 1) * 512`
///      samples and clipped to [-1, 1].
///   6. If the prompt RMS was below `targetRms`, scale the waveform back
///      down by `promptRms / targetRms` (upstream `rms_norm` contract).
struct LuxTtsSynthesizer {

    private let logger = AppLogger(category: "LuxTtsSynthesizer")
    private let store: LuxTtsModelStore

    init(store: LuxTtsModelStore) {
        self.store = store
    }

    /// Synthesize from pre-tokenized ids and 24 kHz mono prompt audio.
    func synthesize(
        promptTokenIds: [Int],
        textTokenIds: [Int],
        promptAudio24k: [Float],
        speed: Float,
        seed: UInt64
    ) async throws -> LuxTtsSynthesisResult {
        guard !promptTokenIds.isEmpty else {
            throw LuxTtsError.tokenizerFailed("prompt produced no tokens")
        }
        guard !textTokenIds.isEmpty else {
            throw LuxTtsError.tokenizerFailed("text produced no tokens")
        }
        guard !promptAudio24k.isEmpty else {
            throw LuxTtsError.invalidPromptAudio("no samples")
        }
        guard speed > 0 else {
            throw LuxTtsError.invalidPromptAudio("speed must be > 0 (got \(speed))")
        }

        let featDim = LuxTtsConstants.featDim
        let maxTokens = LuxTtsConstants.maxTokens
        let maxFrames = LuxTtsConstants.maxFrames

        // --- Prompt: cap length, rms-norm, mel ---
        let maxPromptSamples = Int(
            LuxTtsConstants.maxPromptSeconds * Double(LuxTtsConstants.melSampleRate))
        var prompt = Array(promptAudio24k.prefix(maxPromptSamples))

        var meanSquare: Float = 0
        vDSP_measqv(prompt, 1, &meanSquare, vDSP_Length(prompt.count))
        let promptRms = sqrt(meanSquare)
        guard promptRms > 0 else {
            throw LuxTtsError.invalidPromptAudio("prompt audio is silent")
        }
        if promptRms < LuxTtsConstants.targetRms {
            var gain = LuxTtsConstants.targetRms / promptRms
            vDSP_vsmul(prompt, 1, &gain, &prompt, 1, vDSP_Length(prompt.count))
        }

        // The extractor holds a non-Sendable vDSP DFT setup; keeping it a
        // per-call local (rather than a stored property) keeps the whole
        // synthesizer `struct` `Sendable`, so this async method can be called
        // across `LuxTtsManager`'s actor boundary without tripping the
        // region-based isolation / sending checker. Its precomputed tables are
        // pulled from a shared cache, so construction is cheap.
        let extractor = LuxTtsMelExtractor()
        let promptMel = extractor.extract(audio: prompt)  // [T][100], unscaled
        let promptFrames = promptMel.count
        guard promptFrames > 0 else {
            throw LuxTtsError.invalidPromptAudio("prompt too short for one mel frame")
        }

        // --- Duration estimate + shape-bucket guards ---
        let cat = promptTokenIds + textTokenIds
        let tokenCount = cat.count
        guard tokenCount + 1 <= maxTokens else {
            throw LuxTtsError.inputTooLong(
                "prompt + text tokens \(tokenCount) + pad slot > \(maxTokens)")
        }

        let featuresLength = LuxTtsSolver.featuresLength(
            promptFrames: promptFrames,
            promptTokenCount: promptTokenIds.count,
            textTokenCount: textTokenIds.count,
            speed: Double(speed))
        guard featuresLength <= maxFrames else {
            throw LuxTtsError.inputTooLong(
                "features length \(featuresLength) > \(maxFrames) frames "
                    + "(shorter text or shorter prompt required)")
        }
        let genFrames = featuresLength - promptFrames
        guard genFrames >= 2 else {
            throw LuxTtsError.tokenizerFailed(
                "estimated only \(genFrames) generated frames — text too short")
        }
        // TODO(phase 2): chunk long inputs across multiple vocoder windows
        // instead of erroring; mel truncation is NOT allowed.
        guard let bucket = LuxTtsConstants.vocoderBuckets.first(where: { $0 >= genFrames }) else {
            throw LuxTtsError.inputTooLong(
                "generated frames \(genFrames) exceed the largest vocoder bucket "
                    + "(\(LuxTtsConstants.vocoderBuckets.max() ?? 0))")
        }

        // --- Stage 1: text encoder ---
        let tokens = try makeArray(shape: [1, maxTokens], dataType: .int32)
        let tokenMask = try makeArray(shape: [1, maxTokens], dataType: .float32)
        tokens.withUnsafeMutableBufferPointer(ofType: Int32.self) { buf, _ in
            for i in 0..<tokenCount { buf[i] = Int32(cat[i]) }
        }
        tokenMask.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            // 1.0 = pad, from position S on (the extra pad-slot row at S is
            // masked in attention but its embedding row is still consumed by
            // the remainder frames of the expansion below).
            for i in tokenCount..<maxTokens { buf[i] = 1.0 }
        }

        let encoderOut = try await predict(
            stage: "TextEncoder",
            model: store.textEncoder(),
            inputs: [
                "tokens": MLFeatureValue(multiArray: tokens),
                "padding_mask": MLFeatureValue(multiArray: tokenMask),
            ])
        guard let embedsArray = encoderOut.featureValue(for: "token_embeds")?.multiArrayValue else {
            throw LuxTtsError.inferenceFailed(stage: "TextEncoder", underlying: "no token_embeds")
        }
        // Compact copy: CoreML output rows are stride-padded (e.g. 100 → 112).
        let embeds = try copyRows(
            from: embedsArray, rowCount: tokenCount + 1, rowLength: featDim,
            stage: "TextEncoder")

        // --- Stage 2: expansion + conditions (fixed 1 × 1024 × 100) ---
        let textCondition = try makeArray(shape: [1, maxFrames, featDim], dataType: .float32)
        let speechCondition = try makeArray(shape: [1, maxFrames, featDim], dataType: .float32)
        let frameMask = try makeArray(shape: [1, maxFrames], dataType: .float32)

        let tokensIndex = try LuxTtsSolver.tokensIndex(
            tokensCount: tokenCount, featuresLength: featuresLength)
        textCondition.withUnsafeMutableBufferPointer(ofType: Float.self) { out, _ in
            for frame in 0..<featuresLength {
                let src = tokensIndex[frame] * featDim
                let dst = frame * featDim
                for d in 0..<featDim { out[dst + d] = embeds[src + d] }
            }
        }
        speechCondition.withUnsafeMutableBufferPointer(ofType: Float.self) { out, _ in
            for frame in 0..<promptFrames {
                let dst = frame * featDim
                for d in 0..<featDim {
                    out[dst + d] = promptMel[frame][d] * LuxTtsConstants.featScale
                }
            }
        }
        frameMask.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for i in featuresLength..<maxFrames { buf[i] = 1.0 }
        }

        // --- Stage 3: anchor-Euler flow matching ---
        let activeCount = featuresLength * featDim
        var noise = StyleTTS2NoiseSource(seed: seed)
        var x = noise.nextGaussianArray(count: activeCount)

        let xArray = try makeArray(shape: [1, maxFrames, featDim], dataType: .float32)
        let tArray = try makeArray(shape: [1], dataType: .float32)
        let guidance = try makeArray(shape: [1], dataType: .float32)
        guidance[0] = NSNumber(value: LuxTtsConstants.guidanceScale)

        let timeSteps = LuxTtsSolver.timeSteps(
            numSteps: LuxTtsConstants.numSteps, tShift: LuxTtsConstants.tShift)
        let decoder = try await store.fmDecoder()

        var x0p = [Float](repeating: 0, count: activeCount)
        var x1p = [Float](repeating: 0, count: activeCount)
        // Reused across all 4 steps (copyRows writes every element each step).
        var v = [Float](repeating: 0, count: activeCount)

        for step in 0..<LuxTtsConstants.numSteps {
            let tCur = Float(timeSteps[step])
            let tNext = Float(timeSteps[step + 1])
            let isLast = step == LuxTtsConstants.numSteps - 1

            tArray[0] = NSNumber(value: tCur)
            xArray.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
                // The tail past activeCount stays zero from makeArray's
                // reset(to: 0); only the active prefix is ever written here.
                x.withUnsafeBufferPointer { src in
                    buf.baseAddress!.update(from: src.baseAddress!, count: activeCount)
                }
            }

            let out = try predict(
                stage: "FmDecoder step \(step)",
                model: decoder,
                inputs: [
                    "t": MLFeatureValue(multiArray: tArray),
                    "x": MLFeatureValue(multiArray: xArray),
                    "text_condition": MLFeatureValue(multiArray: textCondition),
                    "speech_condition": MLFeatureValue(multiArray: speechCondition),
                    "guidance_scale": MLFeatureValue(multiArray: guidance),
                    "padding_mask": MLFeatureValue(multiArray: frameMask),
                ])
            guard let vArray = out.featureValue(for: "v")?.multiArrayValue else {
                throw LuxTtsError.inferenceFailed(
                    stage: "FmDecoder step \(step)", underlying: "no v output")
            }
            try copyRows(
                from: vArray, into: &v, rowCount: featuresLength, rowLength: featDim,
                stage: "FmDecoder step \(step)")

            // x1p = x + (1 - t)·v ; x0p = x - t·v ;
            // x ← (1 - tNext)·x0p + tNext·x1p (final step: x = x1p).
            var oneMinusT = 1.0 - tCur
            vDSP_vsma(v, 1, &oneMinusT, x, 1, &x1p, 1, vDSP_Length(activeCount))
            if isLast {
                x = x1p
            } else {
                var negT = -tCur
                vDSP_vsma(v, 1, &negT, x, 1, &x0p, 1, vDSP_Length(activeCount))
                var w0 = 1.0 - tNext
                var w1 = tNext
                vDSP_vsmsma(x0p, 1, &w0, x1p, 1, &w1, &x, 1, vDSP_Length(activeCount))
            }
        }

        // --- Stage 4: vocoder ---
        let melInput = try makeArray(shape: [1, featDim, bucket], dataType: .float32)
        let logFloor = log(LuxTtsConstants.logMelFloor)
        let invScale = 1.0 / LuxTtsConstants.featScale
        melInput.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
            for m in 0..<featDim {
                let row = m * bucket
                for f in 0..<genFrames {
                    buf[row + f] = x[(promptFrames + f) * featDim + m] * invScale
                }
                for f in genFrames..<bucket { buf[row + f] = logFloor }
            }
        }

        let vocoderOut = try await predict(
            stage: "Vocoder\(bucket)",
            model: store.vocoder(bucket: bucket),
            inputs: ["mel": MLFeatureValue(multiArray: melInput)])
        guard let audioArray = vocoderOut.featureValue(for: "audio")?.multiArrayValue else {
            throw LuxTtsError.inferenceFailed(
                stage: "Vocoder\(bucket)", underlying: "no audio output")
        }

        let outputSamples = audioArray.shape.last?.intValue ?? audioArray.count
        let sampleCount = min((genFrames - 1) * LuxTtsConstants.hop48k, outputSamples)
        var samples = try copyRows(
            from: audioArray, rowCount: 1, rowLength: sampleCount,
            stage: "Vocoder\(bucket)")
        var lo: Float = -1.0
        var hi: Float = 1.0
        vDSP_vclip(samples, 1, &lo, &hi, &samples, 1, vDSP_Length(sampleCount))

        // Match the original prompt loudness when the prompt was boosted.
        if promptRms < LuxTtsConstants.targetRms {
            var gain = promptRms / LuxTtsConstants.targetRms
            vDSP_vsmul(samples, 1, &gain, &samples, 1, vDSP_Length(sampleCount))
        }

        logger.info(
            "LuxTTS synth: prompt \(promptFrames) frames, generated \(genFrames) frames "
                + "(bucket \(bucket)), \(sampleCount) samples @48kHz")

        return LuxTtsSynthesisResult(
            samples: samples,
            sampleRate: LuxTtsConstants.outputSampleRate,
            promptFrames: promptFrames,
            generatedFrames: genFrames,
            featuresLength: featuresLength)
    }

    // MARK: - Helpers

    /// Copy `rowCount` rows of `rowLength` floats out of a CoreML output
    /// array into a compact buffer. CoreML prediction outputs are often
    /// stride-padded along the row dimension (e.g. 100-float rows padded to
    /// 112 on GPU) — reading the raw buffer contiguously scrambles every row
    /// after the first, so the row stride must be honored.
    private func copyRows(
        from array: MLMultiArray, rowCount: Int, rowLength: Int, stage: String
    ) throws -> [Float] {
        var out = [Float](repeating: 0, count: rowCount * rowLength)
        try copyRows(from: array, into: &out, rowCount: rowCount, rowLength: rowLength, stage: stage)
        return out
    }

    /// `copyRows` variant that writes into a caller-owned buffer (reused
    /// across the flow-matching steps to avoid a per-step allocation). `out`
    /// must hold at least `rowCount * rowLength` floats.
    private func copyRows(
        from array: MLMultiArray, into out: inout [Float],
        rowCount: Int, rowLength: Int, stage: String
    ) throws {
        precondition(out.count >= rowCount * rowLength, "copyRows destination too small")
        let dims = array.shape.count
        guard array.dataType == .float32, dims >= 2,
            array.strides[dims - 1].intValue == 1,
            rowLength <= array.shape[dims - 1].intValue,
            rowCount <= array.shape[dims - 2].intValue
        else {
            throw LuxTtsError.inferenceFailed(
                stage: stage,
                underlying:
                    "unexpected output layout: shape \(array.shape) strides \(array.strides) "
                    + "dtype \(array.dataType.rawValue) for \(rowCount)×\(rowLength) read")
        }
        let rowStride = array.strides[dims - 2].intValue
        array.withUnsafeBufferPointer(ofType: Float.self) { src in
            out.withUnsafeMutableBufferPointer { dst in
                for row in 0..<rowCount {
                    dst.baseAddress!.advanced(by: row * rowLength)
                        .update(from: src.baseAddress!.advanced(by: row * rowStride), count: rowLength)
                }
            }
        }
    }

    private func makeArray(shape: [Int], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        do {
            let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dataType)
            array.reset(to: 0)
            return array
        } catch {
            throw LuxTtsError.inferenceFailed(stage: "allocation", underlying: "\(error)")
        }
    }

    private func predict(
        stage: String, model: MLModel, inputs: [String: MLFeatureValue]
    ) throws -> MLFeatureProvider {
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: inputs)
            return try model.prediction(from: provider)
        } catch {
            throw LuxTtsError.inferenceFailed(stage: stage, underlying: "\(error)")
        }
    }
}
