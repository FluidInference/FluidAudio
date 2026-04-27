@preconcurrency import CoreML
import Foundation

/// Orchestrates the 7-stage CoreML chain produced by laishere/kokoro-coreml.
///
/// All inputs / outputs follow `convert-coreml.py:run_chain()`. fp16 ↔ fp32
/// conversions are applied at the boundaries identified in
/// `mobius/.../docs/architecture.md`:
///   * Prosody → Noise   : fp16 → fp32
///   * Noise → Vocoder   : fp32 → fp16
///   * Vocoder → Tail    : fp16 → fp32 (and `anchor` is discarded)
public struct KokoroLaiSynthesizer {

    /// One-shot synthesis from already-tokenised input ids + style slices.
    /// Used by `KokoroLaiManager.synthesize(...)` after vocab + voice-pack
    /// resolution.
    public static func synthesize(
        inputIds: [Int32],
        phonemeCount: Int,
        styleS: [Float],
        styleTimbre: [Float],
        speed: Float = Float(KokoroLaiConstants.defaultSpeed),
        store: KokoroLaiModelStore
    ) async throws -> KokoroLaiSynthesisResult {
        precondition(styleS.count == 128, "style_s must be length 128, got \(styleS.count)")
        precondition(
            styleTimbre.count == 128,
            "style_timbre must be length 128, got \(styleTimbre.count)")

        let tEnc = inputIds.count
        var timings = KokoroLaiStageTimings()

        // Build base tensors used by multiple stages.
        let inputIdsArr = try KokoroLaiArrays.int32Array(shape: [1, tEnc], from: inputIds)
        let attnMaskArr = try KokoroLaiArrays.attentionMask(length: tEnc)
        let styleSArr = try KokoroLaiArrays.float16Array(shape: [1, 128], from: styleS)
        let styleTimbreF32 = try KokoroLaiArrays.float32Array(
            shape: [1, 128], from: styleTimbre)
        let styleTimbreF16 = try KokoroLaiArrays.float16Array(
            shape: [1, 128], from: styleTimbre)
        let speedArr = try KokoroLaiArrays.float16Array(shape: [1], from: [speed])

        // ── 1: Albert ────────────────────────────────────────────────
        let albertModel = try await store.model(for: .albert)
        let albertOut = try await predict(
            stage: .albert, model: albertModel,
            inputs: ["input_ids": inputIdsArr, "attention_mask": attnMaskArr],
            timing: &timings.albert
        )
        let bertDur = try KokoroLaiArrays.float16Array(
            shape: try outputShape(albertOut, key: "bert_dur"),
            from: try outputArray(albertOut, key: "bert_dur"))

        // ── 2: PostAlbert ────────────────────────────────────────────
        let postModel = try await store.model(for: .postAlbert)
        let postOut = try await predict(
            stage: .postAlbert, model: postModel,
            inputs: [
                "bert_dur": bertDur,
                "input_ids": inputIdsArr,
                "style_s": styleSArr,
                "speed": speedArr,
                "attention_mask": attnMaskArr,
            ],
            timing: &timings.postAlbert
        )

        // duration → pred_dur (int32, rounded, clamped ≥ 1)
        let duration = try outputArray(postOut, key: "duration")
        let durFloats = KokoroLaiArrays.readFloats(duration)
        let predDur = durFloats.map { d -> Int32 in
            let r = Int32(Float(d).rounded())
            return max(r, 1)
        }
        let tA = predDur.reduce(0) { $0 + Int($1) }
        if tA > KokoroLaiConstants.maxAcousticFrames {
            throw KokoroLaiError.acousticFramesExceedCap(
                have: tA, cap: KokoroLaiConstants.maxAcousticFrames)
        }

        let predDurArr = try KokoroLaiArrays.int32Array(
            shape: [1, predDur.count], from: predDur)
        let dArr = try KokoroLaiArrays.float16Array(
            shape: try outputShape(postOut, key: "d"),
            from: try outputArray(postOut, key: "d"))
        let tEnArr = try KokoroLaiArrays.float16Array(
            shape: try outputShape(postOut, key: "t_en"),
            from: try outputArray(postOut, key: "t_en"))

        // ── 3: Alignment ─────────────────────────────────────────────
        let alignModel = try await store.model(for: .alignment)
        let alignOut = try await predict(
            stage: .alignment, model: alignModel,
            inputs: ["pred_dur": predDurArr, "d": dArr, "t_en": tEnArr],
            timing: &timings.alignment
        )
        let enRaw = try outputArray(alignOut, key: "en")
        let asrRaw = try outputArray(alignOut, key: "asr")

        let enArr = try KokoroLaiArrays.float16Array(
            shape: try outputShape(alignOut, key: "en"), from: enRaw)
        let asrArr = try KokoroLaiArrays.float16Array(
            shape: try outputShape(alignOut, key: "asr"), from: asrRaw)

        // ── 4: Prosody ───────────────────────────────────────────────
        let prosodyModel = try await store.model(for: .prosody)
        let prosOut = try await predict(
            stage: .prosody, model: prosodyModel,
            inputs: ["en": enArr, "style_s": styleSArr],
            timing: &timings.prosody
        )
        let f0Raw = try outputArray(prosOut, key: "F0")
        let nRaw = try outputArray(prosOut, key: "N")

        // ── 5: Noise (fp32 boundary) ─────────────────────────────────
        let f0F32 = try KokoroLaiArrays.float32Array(
            shape: try outputShape(prosOut, key: "F0"), from: f0Raw)
        let noiseModel = try await store.model(for: .noise)
        let noiseOut = try await predict(
            stage: .noise, model: noiseModel,
            inputs: ["F0_curve": f0F32, "style_timbre": styleTimbreF32],
            timing: &timings.noise
        )
        let xs0Raw = try outputArray(noiseOut, key: "x_source_0")
        let xs1Raw = try outputArray(noiseOut, key: "x_source_1")

        // ── 6: Vocoder (fp16 boundary) ───────────────────────────────
        let f0F16 = try KokoroLaiArrays.float16Array(
            shape: try outputShape(prosOut, key: "F0"), from: f0Raw)
        let nF16 = try KokoroLaiArrays.float16Array(
            shape: try outputShape(prosOut, key: "N"), from: nRaw)
        let xs0F16 = try KokoroLaiArrays.float16Array(
            shape: try outputShape(noiseOut, key: "x_source_0"), from: xs0Raw)
        let xs1F16 = try KokoroLaiArrays.float16Array(
            shape: try outputShape(noiseOut, key: "x_source_1"), from: xs1Raw)
        let vocoderModel = try await store.model(for: .vocoder)
        let vocOut = try await predict(
            stage: .vocoder, model: vocoderModel,
            inputs: [
                "asr": asrArr,
                "F0_curve": f0F16,
                "N_pred": nF16,
                "x_source_0": xs0F16,
                "x_source_1": xs1F16,
                "style_timbre": styleTimbreF16,
            ],
            timing: &timings.vocoder
        )
        // Discard "anchor"; use only "x_pre".
        let xPreRaw = try outputArray(vocOut, key: "x_pre")

        // ── 7: Tail (fp32 iSTFT) ─────────────────────────────────────
        let xPreF32 = try KokoroLaiArrays.float32Array(
            shape: try outputShape(vocOut, key: "x_pre"), from: xPreRaw)
        let tailModel = try await store.model(for: .tail)
        let tailOut = try await predict(
            stage: .tail, model: tailModel,
            inputs: ["x_pre": xPreF32],
            timing: &timings.tail
        )
        let audioArr = try outputArray(tailOut, key: "audio")
        let samples = KokoroLaiArrays.readFloats(audioArr)

        return KokoroLaiSynthesisResult(
            samples: samples,
            sampleRate: KokoroLaiConstants.sampleRate,
            encoderTokens: tEnc,
            acousticFrames: tA,
            timings: timings
        )
    }

    // MARK: - Helpers

    private static func predict(
        stage: KokoroLaiStage,
        model: MLModel,
        inputs: [String: MLMultiArray],
        timing: inout Double
    ) async throws -> MLFeatureProvider {
        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputs.mapValues { MLFeatureValue(multiArray: $0) })
        let start = Date()
        do {
            let out = try await model.prediction(from: provider)
            timing = Date().timeIntervalSince(start) * 1000
            return out
        } catch {
            throw KokoroLaiError.predictionFailed(stage: stage.rawValue, underlying: error)
        }
    }

    private static func outputArray(_ provider: MLFeatureProvider, key: String) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw KokoroLaiError.unexpectedOutputShape(
                stage: key, expected: "MLMultiArray for '\(key)'", got: "nil")
        }
        return value
    }

    private static func outputShape(_ provider: MLFeatureProvider, key: String) throws -> [Int] {
        let arr = try outputArray(provider, key: key)
        return arr.shape.map { $0.intValue }
    }
}
