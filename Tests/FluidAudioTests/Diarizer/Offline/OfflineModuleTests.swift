import Accelerate
@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class OfflineDiarizerConfigTests: XCTestCase {

    func testDefaultConfigurationMatchesExpectedValues() throws {
        let config = OfflineDiarizerConfig.default

        XCTAssertEqual(config.clusteringThreshold, 0.6, accuracy: 1e-12)
        XCTAssertEqual(config.Fa, 0.07)
        XCTAssertEqual(config.Fb, 0.8)
        XCTAssertEqual(config.maxVBxIterations, 20)
        XCTAssertTrue(config.embeddingExcludeOverlap)
        XCTAssertEqual(config.samplesPerWindow, 160_000)

        XCTAssertNoThrow(try config.validate())
    }

    func testValidateThrowsForInvalidClusteringThreshold() {
        let config = OfflineDiarizerConfig(clusteringThreshold: 1.5)

        XCTAssertThrowsError(try config.validate()) { error in
            guard case OfflineDiarizationError.invalidConfiguration(let message) = error else {
                XCTFail("Expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("clustering.threshold"))
        }
    }

    func testValidateThrowsForInvalidBatchSize() {
        let config = OfflineDiarizerConfig(embeddingBatchSize: 0)

        XCTAssertThrowsError(try config.validate()) { error in
            guard case OfflineDiarizationError.invalidBatchSize(let message) = error else {
                XCTFail("Expected invalidBatchSize, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("embeddingBatchSize"))
        }
    }

    func testValidateThrowsForInvalidSegmentationMinDurationOn() {
        var config = OfflineDiarizerConfig()
        config.segmentationMinDurationOn = -0.5

        XCTAssertThrowsError(try config.validate()) { error in
            guard case OfflineDiarizationError.invalidConfiguration(let message) = error else {
                XCTFail("Expected invalidConfiguration, got \(error)")
                return
            }
            XCTAssertTrue(message.contains("segmentation.minDurationOn"))
        }
    }
}

@available(macOS 13.0, iOS 16.0, *)
final class OfflineTypesTests: XCTestCase {

    func testErrorDescriptionsAreHumanReadable() {
        XCTAssertEqual(
            OfflineDiarizationError.modelNotLoaded("segmentation").localizedDescription,
            "Model not loaded: segmentation"
        )

        XCTAssertEqual(
            OfflineDiarizationError.noSpeechDetected.localizedDescription,
            "No speech detected in audio"
        )

        XCTAssertEqual(
            OfflineDiarizationError.invalidBatchSize("embedding batch").localizedDescription,
            "Invalid batch size: embedding batch"
        )
    }

    func testSegmentationOutputInitialization() {
        let output = SegmentationOutput(
            logProbs: [[[0.1, 0.9]]],
            numChunks: 1,
            numFrames: 1,
            numSpeakers: 2
        )

        XCTAssertEqual(output.numChunks, 1)
        XCTAssertEqual(output.numFrames, 1)
        XCTAssertEqual(output.numSpeakers, 2)
    }

    func testVBxOutputInitialization() {
        let output = VBxOutput(
            gamma: [[0.6, 0.4]],
            pi: [0.5, 0.5],
            hardClusters: [[0, 1]],
            centroids: [[0.1, 0.2], [0.3, 0.4]],
            numClusters: 2,
            elbos: [1.0, 1.1]
        )

        XCTAssertEqual(output.gamma.count, 1)
        XCTAssertEqual(output.numClusters, 2)
        XCTAssertEqual(output.centroids[1][1], 0.4, accuracy: 1e-6)
    }
}

@available(macOS 13.0, iOS 16.0, *)
final class ModelWarmupTests: XCTestCase {

    func testWarmupSingleInputInvokesPredictionsWithExpectedShape() throws {
        let model = WarmupMockModel()
        let iterations = 3

        let duration = try ModelWarmup.warmup(
            model: model,
            inputName: "audio",
            inputShape: [1, 160],
            iterations: iterations
        )

        XCTAssertGreaterThanOrEqual(duration, 0)
        XCTAssertEqual(model.receivedInputs.count, iterations)

        for invocation in model.receivedInputs {
            let array = invocation["audio"]
            XCTAssertNotNil(array)
            XCTAssertEqual(array?.shape.map { $0.intValue }, [1, 160])
        }
    }

    func testWarmupEmbeddingModelUsesFbankInputsWhenAvailable() throws {
        let model = WarmupMockModel()
        let weightFrames = 64

        try ModelWarmup.warmupEmbeddingModel(model, weightFrames: weightFrames)

        guard let lastInvocation = model.receivedInputs.last else {
            XCTFail("Expected at least one invocation")
            return
        }

        let features = lastInvocation["fbank_features"]
        let weights = lastInvocation["weights"]
        XCTAssertNotNil(features)
        XCTAssertNotNil(weights)

        XCTAssertEqual(features?.shape.map { $0.intValue }, [1, 1, 80, 998])
        XCTAssertEqual(weights?.shape.map { $0.intValue }, [1, weightFrames])
    }

    func testWarmupEmbeddingModelFallsBackToCombinedWhenFbankFails() throws {
        let model = WarmupMockModel()
        model.failureKeys = ["fbank_features"]
        let weightFrames = 32

        try ModelWarmup.warmupEmbeddingModel(model, weightFrames: weightFrames)

        // Expect one invocation: only the successful combined fallback is recorded
        XCTAssertEqual(model.receivedInputs.count, 1)

        guard let lastInvocation = model.receivedInputs.last else {
            XCTFail("Expected fallback invocation")
            return
        }

        XCTAssertNotNil(lastInvocation["audio_and_weights"])
        XCTAssertNil(lastInvocation["fbank_features"])
    }

    // MARK: - Helpers

    private final class WarmupMockModel: MLModel {
        private(set) var receivedInputs: [[String: MLMultiArray]] = []
        var failureKeys: Set<String> = []

        override func prediction(
            from input: MLFeatureProvider,
            options: MLPredictionOptions = MLPredictionOptions()
        ) throws -> MLFeatureProvider {
            for name in input.featureNames {
                if failureKeys.contains(name) {
                    throw MockError.simulatedFailure
                }
            }

            var captured: [String: MLMultiArray] = [:]
            for name in input.featureNames {
                if let array = input.featureValue(for: name)?.multiArrayValue {
                    captured[name] = array
                }
            }
            receivedInputs.append(captured)

            return try MLDictionaryFeatureProvider(dictionary: [
                "output": MLFeatureValue(double: 0.0)
            ])
        }

        private enum MockError: Error {
            case simulatedFailure
        }
    }
}
