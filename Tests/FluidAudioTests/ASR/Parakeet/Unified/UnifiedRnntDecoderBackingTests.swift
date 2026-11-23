import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// `UnifiedRnntDecoder` reuses its inputs and binds its outputs to buffers made once.
/// It must emit exactly what a plain loop does that makes fresh tensors for every
/// call, including when state carries from one `decode` pass to the next.
///
/// Needs the Parakeet Unified decoder and joint in the default model cache; skipped
/// when they have not been downloaded.
final class UnifiedRnntDecoderBackingTests: XCTestCase {

    private let config = UnifiedConfig()

    private var modelDirectory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models", isDirectory: true)
            .appendingPathComponent(config.family.repo.folderName, isDirectory: true)
    }

    private func loadModel(_ file: String, _ units: MLComputeUnits) throws -> MLModel {
        let url = modelDirectory.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip("Parakeet Unified models are not downloaded (\(file))")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    /// Encoder output [1, D, T] for the first 15 s window of a speech fixture.
    private func encoderOutput() throws -> (encoded: MLMultiArray, frames: Int) {
        let fixture = "01-validation-request-21.4s"
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(fixture)", withExtension: "wav")
                ?? Bundle.module.url(forResource: fixture, withExtension: "wav")
        else {
            throw XCTSkip("fixture \(fixture).wav not bundled")
        }
        let samples = try AudioConverter().resampleAudioFile(url)
        let windowSamples = 15 * config.sampleRate
        var window = [Float](repeating: 0, count: windowSamples)
        let validCount = min(samples.count, windowSamples)
        window.replaceSubrange(0..<validCount, with: samples[0..<validCount])
        let extractor = UnifiedMelExtractor(windowSamples: windowSamples, nMels: config.melFeatures)
        let (mel, melLength) = try extractor.features(window: window, validCount: validCount)
        let encoder = try loadModel(config.family.offlineEncoderFile(precision: .int8), .cpuAndNeuralEngine)
        let output = try encoder.prediction(
            from: UnifiedEncoderFeatureProvider(mel: mel, melLength: melLength, promptId: nil))
        guard let encoded = output.featureValue(for: "encoder")?.multiArrayValue,
            let length = output.featureValue(for: "encoder_length")?.multiArrayValue
        else {
            throw XCTSkip("encoder gave no output")
        }
        return (encoded, min(length[0].intValue, encoded.shape[2].intValue))
    }

    private struct ReferenceDecoder {
        let decoder: MLModel
        let joint: MLModel
        let config: UnifiedConfig
        var h: MLMultiArray
        var c: MLMultiArray
        var lastToken: Int32

        init(decoder: MLModel, joint: MLModel, config: UnifiedConfig) throws {
            self.decoder = decoder
            self.joint = joint
            self.config = config
            h = try Self.zeros(config)
            c = try Self.zeros(config)
            lastToken = Int32(config.blankIdx)
        }

        static func zeros(_ config: UnifiedConfig) throws -> MLMultiArray {
            let array = try MLMultiArray(
                shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
                dataType: .float32)
            array.reset(to: 0)
            return array
        }

        func step(_ token: Int32, _ h: MLMultiArray, _ c: MLMultiArray) throws -> MLFeatureProvider {
            let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
            targets[0] = NSNumber(value: token)
            let length = try MLMultiArray(shape: [1], dataType: .int32)
            length[0] = 1
            return try decoder.prediction(
                from: MLDictionaryFeatureProvider(dictionary: [
                    "targets": targets, "target_length": length, "h_in": h, "c_in": c,
                ]))
        }

        /// The decode loop with fresh tensors for every call.
        mutating func decode(_ encoded: MLMultiArray, _ frames: Range<Int>) throws -> [(Int, Int, Float)] {
            var token = lastToken
            var currentH = h
            var currentC = c
            var out = try step(token, currentH, currentC)
            var emissions: [(Int, Int, Float)] = []
            let dim = encoded.shape[1].intValue
            let source = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
            for t in frames {
                let frame = try MLMultiArray(shape: [1, NSNumber(value: dim), 1], dataType: .float32)
                let target = frame.dataPointer.bindMemory(to: Float.self, capacity: dim)
                for d in 0..<dim {
                    target[d] = source[d * encoded.strides[1].intValue + t * encoded.strides[2].intValue]
                }
                for _ in 0..<config.maxSymbolsPerFrame {
                    let result = try joint.prediction(
                        from: MLDictionaryFeatureProvider(dictionary: [
                            "encoder_step": frame,
                            "decoder_step": out.featureValue(for: "decoder")!.multiArrayValue!,
                        ]))
                    let id = result.featureValue(for: "token_id")!.multiArrayValue![0].int32Value
                    if id == Int32(config.blankIdx) { break }
                    let prob = result.featureValue(for: "token_prob")!.multiArrayValue![0].floatValue
                    emissions.append((Int(id), t, prob))
                    token = id
                    currentH = out.featureValue(for: "h_out")!.multiArrayValue!
                    currentC = out.featureValue(for: "c_out")!.multiArrayValue!
                    out = try step(token, currentH, currentC)
                }
            }
            lastToken = token
            h = currentH
            c = currentC
            return emissions
        }
    }

    func testMatchesFreshTensorDecodingAcrossPassesAndReset() throws {
        let decoderModel = try loadModel(config.family.decoderFile, .cpuOnly)
        let jointModel = try loadModel(config.family.jointDecisionFile, .cpuOnly)
        let (encoded, frames) = try encoderOutput()
        let decoder = try UnifiedRnntDecoder(
            decoderModel: decoderModel, jointDecisionModel: jointModel, config: config)

        // Twice, with a reset between, so stale buffers from the first round would show.
        for round in 0..<2 {
            var reference = try ReferenceDecoder(decoder: decoderModel, joint: jointModel, config: config)
            try decoder.reset()
            var compared = 0
            // Two passes, so LSTM state and the last token carry between them.
            for range in [0..<frames / 2, frames / 2..<frames] {
                let expected = try reference.decode(encoded, range)
                let actual = try decoder.decode(encoded: encoded, frameRange: range)
                XCTAssertEqual(actual.map(\.token), expected.map(\.0), "round \(round), frames \(range)")
                XCTAssertEqual(actual.map(\.frame), expected.map(\.1), "round \(round), frames \(range)")
                XCTAssertEqual(
                    actual.map { $0.prob.bitPattern }, expected.map { $0.2.bitPattern },
                    "round \(round), frames \(range)")
                compared += expected.count
            }
            XCTAssertGreaterThan(compared, 20, "too few tokens emitted to compare")
        }
    }
}
