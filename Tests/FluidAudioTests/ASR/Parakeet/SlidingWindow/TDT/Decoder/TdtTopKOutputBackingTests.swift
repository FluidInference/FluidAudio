import CoreML
import XCTest

@testable import FluidAudio

/// The v3 joint backs its `top_k_ids` / `top_k_logits` outputs too, so every joint output is
/// backed (#1168). Backed predictions must give the same decisions as unbacked ones frame
/// after frame, and a decision must keep its top-K values after later predictions reuse the
/// buffers.
///
/// Needs the Parakeet TDT v3 models. Runs when they are already cached or when
/// `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download; otherwise skips.
@available(macOS 14.0, iOS 17.0, *)
final class TdtTopKOutputBackingTests: XCTestCase {

    private func loadModels() async throws -> AsrModels {
        let cached = AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory())
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            cached || allowDownload, "Parakeet v3 models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        return try await AsrModels.downloadAndLoad()
    }

    func testBackingsFollowTheJointsDeclaredTopKOutputs() async throws {
        let models = try await loadModels()
        let backings = try TdtModelInference.topKOutputBackings(for: models.joint)
        let outputs = models.joint.modelDescription.outputDescriptionsByName
        XCTAssertEqual(Set(backings.keys), ["top_k_ids", "top_k_logits"])
        for (name, backing) in backings {
            let constraint = try XCTUnwrap(outputs[name]?.multiArrayConstraint)
            XCTAssertEqual(backing.shape, constraint.shape, name)
            XCTAssertEqual(backing.dataType, constraint.dataType, name)
        }
        // A model without the pair gets none, so joints without top-K are unchanged.
        XCTAssertTrue(try TdtModelInference.topKOutputBackings(for: models.decoder).isEmpty)
    }

    func testBackedTopKDecisionsMatchUnbackedFrameByFrame() async throws {
        let models = try await loadModels()
        let encoderModel = try XCTUnwrap(models.encoder, "v3 runs a separate encoder")
        let asr = AsrManager()
        try await asr.loadModels(models)

        // The encoder output of the first 15 s of a recording.
        let name = "01-validation-request-21.4s.wav"
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil))
        var audio = Array(try AudioConverter().resampleAudioFile(url).prefix(ASRConstants.maxModelSamples))
        audio += [Float](repeating: 0, count: ASRConstants.maxModelSamples - audio.count)
        let signal = try MLMultiArray(shape: [1, NSNumber(value: audio.count)], dataType: .float32)
        audio.withUnsafeBufferPointer { samples in
            signal.dataPointer.bindMemory(to: Float.self, capacity: audio.count)
                .update(from: samples.baseAddress!, count: audio.count)
        }
        let signalLength = try MLMultiArray(shape: [1], dataType: .int32)
        signalLength[0] = NSNumber(value: audio.count)
        let preprocessorInput = try MLDictionaryFeatureProvider(dictionary: [
            "audio_signal": MLFeatureValue(multiArray: signal),
            "audio_length": MLFeatureValue(multiArray: signalLength),
        ])
        let preprocessed = try await models.preprocessor.prediction(from: preprocessorInput)
        var encoderFeatures: [String: MLFeatureValue] = [:]
        for name in encoderModel.modelDescription.inputDescriptionsByName.keys {
            encoderFeatures[name] = try XCTUnwrap(
                preprocessed.featureValue(for: name) ?? preprocessorInput.featureValue(for: name), name)
        }
        let encoded = try await encoderModel.prediction(
            from: try MLDictionaryFeatureProvider(dictionary: encoderFeatures))
        let encoderOutput = try XCTUnwrap(encoded.featureValue(for: "encoder")?.multiArrayValue)
        let encoderLength = try XCTUnwrap(encoded.featureValue(for: "encoder_length")?.multiArrayValue)[0].intValue
        let frames = try EncoderFrameView(encoderOutput: encoderOutput, validLength: encoderLength)

        // The decoder primed with the start-of-sequence token, as each decode begins.
        let inference = TdtModelInference()
        let state = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
        let target = try MLMultiArray(shape: [1, 1], dataType: .int32)
        let targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = 1
        let primed = try inference.runDecoder(
            token: models.version.blankId, state: state, model: models.decoder, targetArray: target,
            targetLengthArray: targetLength)
        let decoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: ASRConstants.decoderHiddenSize), 1], dataType: .float32)
        try inference.normalizeDecoderProjection(
            try XCTUnwrap(primed.output.featureValue(for: "decoder")?.multiArrayValue), into: decoderStep)

        let hidden = models.version.encoderHiddenSize
        let encoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: hidden), 1], dataType: .float32)
        let provider = ReusableJointInputProvider(encoderStep: encoderStep, decoderStep: decoderStep)
        let tokenId = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
        let tokenProb = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        let duration = try MLMultiArray(shape: [1, 1, 1], dataType: .int32)
        let topK = try TdtModelInference.topKOutputBackings(for: models.joint)

        func decide(_ frame: Int, backed: Bool) throws -> TdtJointDecision {
            try inference.runJointPrepared(
                encoderFrames: frames, timeIndex: frame, preparedDecoderStep: decoderStep, model: models.joint,
                encoderStep: encoderStep,
                encoderDestPtr: encoderStep.dataPointer.bindMemory(to: Float.self, capacity: hidden),
                encoderDestStride: encoderStep.strides[1].intValue, inputProvider: provider,
                tokenIdBacking: tokenId, tokenProbBacking: tokenProb, durationBacking: duration,
                topKBackings: backed ? topK : [:], needsTopK: true)
        }

        var kept: [TdtJointDecision] = []
        for frame in 0..<min(encoderLength, 60) {
            let backed = try decide(frame, backed: true)
            let unbacked = try decide(frame, backed: false)
            XCTAssertEqual(backed.token, unbacked.token, "frame \(frame)")
            XCTAssertEqual(backed.probability, unbacked.probability, "frame \(frame)")
            XCTAssertEqual(backed.durationBin, unbacked.durationBin, "frame \(frame)")
            XCTAssertEqual(backed.topKIds, unbacked.topKIds, "frame \(frame)")
            XCTAssertEqual(backed.topKLogits, unbacked.topKLogits, "frame \(frame)")
            kept.append(backed)
        }
        XCTAssertGreaterThan(Set(kept.compactMap { $0.topKIds?.first }).count, 1, "frames should differ")

        // Later predictions reused the buffers; each decision still holds its own values.
        for (frame, decision) in kept.enumerated() {
            let again = try decide(frame, backed: true)
            XCTAssertEqual(decision.topKIds, again.topKIds, "frame \(frame)")
            XCTAssertEqual(decision.topKLogits, again.topKLogits, "frame \(frame)")
        }
    }
}
