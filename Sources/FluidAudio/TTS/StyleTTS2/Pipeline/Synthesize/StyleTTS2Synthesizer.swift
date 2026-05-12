@preconcurrency import CoreML
import Foundation

/// 8-stage StyleTTS2 LibriTTS (iteration_3) synthesizer.
///
/// Mirrors the dispatch order + per-stage feed shapes from
/// `mobius/models/tts/styletts2/coreml/inference.py`. Eager glue between
/// stages lives in `StyleTTS2GlueOps`; tensor I/O conversion lives in
/// `StyleTTS2MultiArray`.
public actor StyleTTS2Synthesizer {

    private let logger = AppLogger(category: "StyleTTS2Synthesizer")
    private let store: StyleTTS2ModelStore

    public init(store: StyleTTS2ModelStore) {
        self.store = store
    }

    // MARK: - Public API

    /// Synthesize 24 kHz mono Float32 audio for the given (already encoded
    /// to TextCleaner IDs) phoneme tokens, blending the diffusion-sampled
    /// style with the reference-audio style at α / β.
    ///
    /// - Parameters:
    ///   - tokenIds: TextCleaner IDs (must include the leading 0 pad token).
    ///   - referenceMel: Flat row-major `[1, 1, 80, T_mel]` Float32 mel
    ///     spectrogram from `StyleTTS2MelExtractor`.
    ///   - referenceMelFrames: `T_mel`.
    ///   - alpha: Blend weight for `ref_diff` (default 0.3).
    ///   - beta: Blend weight for `s_diff` (default 0.7).
    ///   - noiseSeed: RNG seed for the fused-sampler aux noises (default 0).
    public func synthesize(
        tokenIds: [Int32],
        referenceMel: [Float],
        referenceMelFrames: Int,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        noiseSeed: UInt64 = 0
    ) async throws -> [Float] {
        let realN = tokenIds.count

        // Choose the bert / sampler bucket. Default T = 57 unless we need
        // a bucketed variant. Note: realN must satisfy `realN ≤ chosenT`.
        let chosenT: Int
        if realN <= StyleTTS2Constants.defaultBertTokens {
            chosenT = StyleTTS2Constants.defaultBertTokens
        } else {
            chosenT = try await store.resolveBucket(for: realN)
        }
        if realN > chosenT {
            throw StyleTTS2Error.textTooLong(tokenCount: realN, maxLength: chosenT)
        }

        // ---- Stage 1: text_encoder (CPU_ONLY, RangeDim T) ----
        let tEnFlat = try await runTextEncoder(tokenIds: tokenIds, realN: realN)
        // tEn shape from the CoreML graph is [1, 512, real_n]. Channel count
        // is fixed at 512 by the architecture.
        let textEncoderChannels = tEnFlat.count / realN

        // ---- Stage 2: bert + bert_encoder (ALL, fixed T = chosenT) ----
        let (bertDur, dEn) = try await runBert(
            tokenIds: tokenIds, realN: realN, paddedT: chosenT)
        // bertDur is `[1, T, 768]`; dEn after slicing is `[1, 512, real_n]`
        // (we slice off the padded positions before feeding duration_predictor).
        let bertDurChannels = bertDur.count / chosenT
        let dEnChannels = dEn.count / realN

        // ---- Stage 3: ref_encoder (CPU_AND_GPU) ----
        let refS = try await runRefEncoder(mel: referenceMel, frames: referenceMelFrames)
        // refS is `[1, 256]` flat Float32.

        // ---- Stage 4: fused_diffusion_sampler (ALL, fixed T = chosenT) ----
        let sPred = try await runFusedSampler(
            bertDur: bertDur, bertDurChannels: bertDurChannels, paddedT: chosenT,
            refS: refS, noiseSeed: noiseSeed,
            tokenCount: realN)
        // sPred is `[1, 1, 256]` flat Float32.

        // ---- α/β blend: split + mix the 256-dim style vector ----
        let (ref128, s128) = StyleTTS2GlueOps.blendStyle(
            sPred256: sPred, refS256: refS, alpha: alpha, beta: beta)

        // ---- Stage 5: duration_predictor (CPU_ONLY, RangeDim T) ----
        let (d, durationLogits) = try await runDurationPredictor(
            dEn: dEn, dEnChannels: dEnChannels, realN: realN, s: s128)

        let durations = try StyleTTS2GlueOps.roundDurations(durationLogits)
        let (alignmentMatrix, totalFrames) = StyleTTS2GlueOps.buildAlignmentMatrix(
            durations: durations)
        logger.info("StyleTTS2 alignment: \(realN) tokens → \(totalFrames) frames")

        // ---- Build `en` and `asr` via alignment matmul + optional shift ----
        // d is `[1, real_n, 640]` row-major → transpose to [1, 640, real_n]
        // before matmul.
        let dChannels = d.count / realN
        let dT = StyleTTS2GlueOps.transposeLast2D(d, rows: realN, cols: dChannels)
        var en = StyleTTS2GlueOps.matmulAligned(
            features: dT, channels: dChannels, realN: realN,
            alignment: alignmentMatrix, totalFrames: totalFrames)
        var asr = StyleTTS2GlueOps.matmulAligned(
            features: tEnFlat, channels: textEncoderChannels, realN: realN,
            alignment: alignmentMatrix, totalFrames: totalFrames)

        if StyleTTS2Constants.applyHifiganAsrShift {
            en = StyleTTS2GlueOps.hifiganShift(en, channels: dChannels, frames: totalFrames)
            asr = StyleTTS2GlueOps.hifiganShift(
                asr, channels: textEncoderChannels, frames: totalFrames)
        }

        // ---- Stage 6: fused_f0n_har_source (CPU_ONLY, fp32) ----
        let (f0Pred, nPred, har) = try await runFusedF0nHarSource(
            en: en, enChannels: dChannels, totalFrames: totalFrames, s: s128)

        // ---- Stage 7: decoder_pre (CPU_AND_NE, fp16) ----
        let xPre = try await runDecoderPre(
            asr: asr, asrChannels: textEncoderChannels, totalFrames: totalFrames,
            f0Pred: f0Pred, nPred: nPred, ref: ref128)

        // ---- Stage 8: decoder_upsample (CPU_ONLY, fp16) ----
        // x_pre is [1, 512, totalFrames * 2] (decoder_pre upsamples once);
        // har_source is [1, 1, har.count].
        let audio = try await runDecoderUpsample(
            xPre: xPre, xPreChannels: 512, xPreFrames: totalFrames * 2,
            ref: ref128, harSource: har)

        // Tail trim — mirrors `np.squeeze(audio_np)[..., :-50]`.
        let trim = min(StyleTTS2Constants.tailTrimSamples, audio.count)
        if trim > 0 {
            return Array(audio.prefix(audio.count - trim))
        }
        return audio
    }

    // MARK: - Per-stage runners

    private func runTextEncoder(tokenIds: [Int32], realN: Int) async throws -> [Float] {
        let model = try await store.textEncoder()

        let tokens = try StyleTTS2MultiArray.makeInt32(
            tokenIds, shape: [1, realN])
        let lengths = try StyleTTS2MultiArray.makeInt32(
            [Int32(realN)], shape: [1])
        // text_mask = length_to_mask(lengths) — all-zero `[1, real_n]` for the
        // RangeDim path (no padding happens, so no positions are masked).
        let textMask = try StyleTTS2MultiArray.makeFloat32(
            [Float](repeating: 0, count: realN), shape: [1, realN])

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: tokens),
            "input_lengths": MLFeatureValue(multiArray: lengths),
            "text_mask": MLFeatureValue(multiArray: textMask),
        ])
        let out = try await predict(model, provider: provider, stage: "text_encoder")
        let outputName = firstOutputName(of: model)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "text_encoder", underlying: "no output value")
        }
        return StyleTTS2MultiArray.extractFloats(arr)
    }

    private func runBert(
        tokenIds: [Int32], realN: Int, paddedT: Int
    ) async throws -> (bertDur: [Float], dEn: [Float]) {
        // Pad tokens with 0s up to paddedT and build attention mask.
        var paddedTokens = tokenIds
        if realN < paddedT {
            paddedTokens.append(contentsOf: [Int32](repeating: 0, count: paddedT - realN))
        }
        var attentionMask = [Int32](repeating: 1, count: realN)
        if realN < paddedT {
            attentionMask.append(contentsOf: [Int32](repeating: 0, count: paddedT - realN))
        }

        let model = try await store.bertModel(forTokenCount: realN)
        let tokensArr = try StyleTTS2MultiArray.makeInt32(
            paddedTokens, shape: [1, paddedT])
        let attnArr = try StyleTTS2MultiArray.makeInt32(
            attentionMask, shape: [1, paddedT])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "tokens": MLFeatureValue(multiArray: tokensArr),
            "attention_mask": MLFeatureValue(multiArray: attnArr),
        ])
        let out = try await predict(model, provider: provider, stage: "bert")

        let outputs = orderedOutputs(of: model)
        guard outputs.count == 2 else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "bert", underlying: "expected 2 outputs, got \(outputs.count)")
        }
        guard
            let bertDurArr = out.featureValue(for: outputs[0])?.multiArrayValue,
            let dEnPaddedArr = out.featureValue(for: outputs[1])?.multiArrayValue
        else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "bert", underlying: "missing output value(s)")
        }
        let bertDur = StyleTTS2MultiArray.extractFloats(bertDurArr)
        let dEnPadded = StyleTTS2MultiArray.extractFloats(dEnPaddedArr)

        // dEnPadded shape `[1, 512, paddedT]`. Slice the trailing
        // padding away to get `[1, 512, real_n]`.
        let dEnShape = StyleTTS2MultiArray.shape(of: dEnPaddedArr)
        guard dEnShape.count == 3, dEnShape[2] == paddedT else {
            throw StyleTTS2Error.invalidTensorShape(
                stage: "bert.d_en", expected: "[1, C, \(paddedT)]", got: "\(dEnShape)")
        }
        let channels = dEnShape[1]
        var dEn = [Float](repeating: 0, count: channels * realN)
        for c in 0..<channels {
            for n in 0..<realN {
                dEn[c * realN + n] = dEnPadded[c * paddedT + n]
            }
        }
        return (bertDur, dEn)
    }

    private func runRefEncoder(mel: [Float], frames: Int) async throws -> [Float] {
        let model = try await store.refEncoder()
        // Mel feed shape: `[1, 1, 80, T_mel]`.
        let melArr = try StyleTTS2MultiArray.makeFloat32(
            mel, shape: [1, 1, StyleTTS2Constants.melNMels, frames])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "mel": MLFeatureValue(multiArray: melArr)
        ])
        let out = try await predict(model, provider: provider, stage: "ref_encoder")
        let outputName = firstOutputName(of: model)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "ref_encoder", underlying: "no output value")
        }
        return StyleTTS2MultiArray.extractFloats(arr)
    }

    private func runFusedSampler(
        bertDur: [Float],
        bertDurChannels: Int,
        paddedT: Int,
        refS: [Float],
        noiseSeed: UInt64,
        tokenCount: Int
    ) async throws -> [Float] {
        let model = try await store.samplerModel(forTokenCount: tokenCount)

        // Pre-draw `noise_init` ([1, 1, 256]) + `noises_aux`
        // ([N-1, 1, 1, 256]) Gaussians.
        var rng = StyleTTS2NoiseSource(seed: noiseSeed)
        let noiseInit = rng.nextGaussianArray(count: StyleTTS2Constants.styleDim)
        let auxStepCount = StyleTTS2Constants.diffusionSteps - 1
        var noisesAux = [Float](
            repeating: 0,
            count: auxStepCount * StyleTTS2Constants.styleDim)
        for s in 0..<auxStepCount {
            let row = rng.nextGaussianArray(count: StyleTTS2Constants.styleDim)
            for i in 0..<StyleTTS2Constants.styleDim {
                noisesAux[s * StyleTTS2Constants.styleDim + i] = row[i]
            }
        }

        let noiseInitArr = try StyleTTS2MultiArray.makeFloat32(
            noiseInit, shape: [1, 1, StyleTTS2Constants.styleDim])
        let noisesAuxArr = try StyleTTS2MultiArray.makeFloat32(
            noisesAux, shape: [auxStepCount, 1, 1, StyleTTS2Constants.styleDim])
        let embeddingArr = try StyleTTS2MultiArray.makeFloat32(
            bertDur, shape: [1, paddedT, bertDurChannels])
        let featuresArr = try StyleTTS2MultiArray.makeFloat32(
            refS, shape: [1, refS.count])

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "noise_init": MLFeatureValue(multiArray: noiseInitArr),
            "noises_aux": MLFeatureValue(multiArray: noisesAuxArr),
            "embedding": MLFeatureValue(multiArray: embeddingArr),
            "features": MLFeatureValue(multiArray: featuresArr),
        ])
        let out = try await predict(model, provider: provider, stage: "fused_diffusion_sampler")
        let outputName = firstOutputName(of: model)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "fused_diffusion_sampler", underlying: "no output value")
        }
        return StyleTTS2MultiArray.extractFloats(arr)
    }

    private func runDurationPredictor(
        dEn: [Float], dEnChannels: Int, realN: Int, s: [Float]
    ) async throws -> (d: [Float], logits: MLMultiArray) {
        let model = try await store.durationPredictor()
        let dEnArr = try StyleTTS2MultiArray.makeFloat32(
            dEn, shape: [1, dEnChannels, realN])
        let sArr = try StyleTTS2MultiArray.makeFloat32(
            s, shape: [1, s.count])
        // text_mask shaped `[1, real_n]`, all zeros (no padding in the
        // RangeDim T path).
        let textMaskArr = try StyleTTS2MultiArray.makeFloat32(
            [Float](repeating: 0, count: realN), shape: [1, realN])

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "d_en": MLFeatureValue(multiArray: dEnArr),
            "s": MLFeatureValue(multiArray: sArr),
            "text_mask": MLFeatureValue(multiArray: textMaskArr),
        ])
        let out = try await predict(model, provider: provider, stage: "duration_predictor")

        let outputs = orderedOutputs(of: model)
        guard outputs.count == 2 else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "duration_predictor",
                underlying: "expected 2 outputs, got \(outputs.count)")
        }
        guard
            let dArr = out.featureValue(for: outputs[0])?.multiArrayValue,
            let logitsArr = out.featureValue(for: outputs[1])?.multiArrayValue
        else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "duration_predictor", underlying: "missing output value(s)")
        }
        return (StyleTTS2MultiArray.extractFloats(dArr), logitsArr)
    }

    private func runFusedF0nHarSource(
        en: [Float], enChannels: Int, totalFrames: Int, s: [Float]
    ) async throws -> (f0: [Float], n: [Float], har: [Float]) {
        let model = try await store.fusedF0nHarSource()
        let enArr = try StyleTTS2MultiArray.makeFloat32(
            en, shape: [1, enChannels, totalFrames])
        let sArr = try StyleTTS2MultiArray.makeFloat32(
            s, shape: [1, s.count])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "en": MLFeatureValue(multiArray: enArr),
            "s": MLFeatureValue(multiArray: sArr),
        ])
        let out = try await predict(model, provider: provider, stage: "fused_f0n_har_source")

        let outputs = orderedOutputs(of: model)
        guard outputs.count == 3 else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "fused_f0n_har_source",
                underlying: "expected 3 outputs, got \(outputs.count)")
        }
        guard
            let f0Arr = out.featureValue(for: outputs[0])?.multiArrayValue,
            let nArr = out.featureValue(for: outputs[1])?.multiArrayValue,
            let harArr = out.featureValue(for: outputs[2])?.multiArrayValue
        else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "fused_f0n_har_source", underlying: "missing output value(s)")
        }
        return (
            StyleTTS2MultiArray.extractFloats(f0Arr),
            StyleTTS2MultiArray.extractFloats(nArr),
            StyleTTS2MultiArray.extractFloats(harArr)
        )
    }

    private func runDecoderPre(
        asr: [Float], asrChannels: Int, totalFrames: Int,
        f0Pred: [Float], nPred: [Float], ref: [Float]
    ) async throws -> [Float] {
        let model = try await store.decoderPre()
        let asrArr = try StyleTTS2MultiArray.makeFloat32(
            asr, shape: [1, asrChannels, totalFrames])
        // f0_pred / n_pred shape: `[1, 2 * real_frames]` from the upstream
        // `f0n_predictor`. The fused stage emits the same flattened
        // representation, so we reshape to whatever the model expects (the
        // mlmodelc carries the canonical shape; we just pass `[1, count]`).
        let f0Arr = try StyleTTS2MultiArray.makeFloat32(
            f0Pred, shape: [1, f0Pred.count])
        let nArr = try StyleTTS2MultiArray.makeFloat32(
            nPred, shape: [1, nPred.count])
        let refArr = try StyleTTS2MultiArray.makeFloat32(
            ref, shape: [1, ref.count])

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "asr": MLFeatureValue(multiArray: asrArr),
            "f0_pred": MLFeatureValue(multiArray: f0Arr),
            "n_pred": MLFeatureValue(multiArray: nArr),
            "ref": MLFeatureValue(multiArray: refArr),
        ])
        let out = try await predict(model, provider: provider, stage: "decoder_pre")
        let outputName = firstOutputName(of: model)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "decoder_pre", underlying: "no output value")
        }
        return StyleTTS2MultiArray.extractFloats(arr)
    }

    private func runDecoderUpsample(
        xPre: [Float], xPreChannels: Int, xPreFrames: Int,
        ref: [Float], harSource: [Float]
    ) async throws -> [Float] {
        let model = try await store.decoderUpsample()
        let xPreArr = try StyleTTS2MultiArray.makeFloat32(
            xPre, shape: [1, xPreChannels, xPreFrames])
        let refArr = try StyleTTS2MultiArray.makeFloat32(
            ref, shape: [1, ref.count])
        let harArr = try StyleTTS2MultiArray.makeFloat32(
            harSource, shape: [1, 1, harSource.count])

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "x_pre": MLFeatureValue(multiArray: xPreArr),
            "ref": MLFeatureValue(multiArray: refArr),
            "har_source": MLFeatureValue(multiArray: harArr),
        ])
        let out = try await predict(model, provider: provider, stage: "decoder_upsample")
        let outputName = firstOutputName(of: model)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw StyleTTS2Error.inferenceFailed(
                stage: "decoder_upsample", underlying: "no output value")
        }
        return StyleTTS2MultiArray.extractFloats(arr)
    }

    // MARK: - Plumbing

    private func predict(
        _ model: MLModel, provider: MLFeatureProvider, stage: String
    ) async throws -> sending MLFeatureProvider {
        do {
            return try await model.prediction(from: provider)
        } catch {
            throw StyleTTS2Error.inferenceFailed(stage: stage, underlying: "\(error)")
        }
    }

    private nonisolated func firstOutputName(of model: MLModel) -> String {
        return model.modelDescription.outputDescriptionsByName.keys.first ?? ""
    }

    /// Output names in the canonical spec order. CoreML preserves output
    /// declaration order, but `outputDescriptionsByName` is a dictionary;
    /// fall back to a stable sort if the spec walk fails.
    private nonisolated func orderedOutputs(of model: MLModel) -> [String] {
        let descMap = model.modelDescription.outputDescriptionsByName
        return descMap.keys.sorted()
    }
}
