import Accelerate
@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class WeightInterpolationTests: XCTestCase {

    func testResampleUsesHalfPixelOffsetMapping() {
        let input: [Float] = [0, 10, 20, 30]
        let result = WeightInterpolation.resample(input, to: 2)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0], 5, accuracy: 1e-5)
        XCTAssertEqual(result[1], 25, accuracy: 1e-5)
    }

    func testResampleMatchesInterpolationCoefficients() {
        let input = (0..<16).map { Float($0) * 0.25 }
        let outputLength = 7

        let direct = WeightInterpolation.resample(input, to: outputLength)
        let coefficients = WeightInterpolation.InterpolationCoefficients(
            inputLength: input.count,
            outputLength: outputLength
        )
        let gathered = coefficients.interpolate(input)

        XCTAssertEqual(direct.count, gathered.count)
        for (lhs, rhs) in zip(direct, gathered) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-5)
        }
    }

    func testResample2DBroadcastsRows() {
        let inputs: [[Float]] = [
            [1, 3, 5, 7],
            [2, 4, 6, 8],
        ]

        let outputs = WeightInterpolation.resample2D(inputs, to: 2)

        XCTAssertEqual(outputs.count, 2)
        XCTAssertEqual(outputs[0][0], 2, accuracy: 1e-5)
        XCTAssertEqual(outputs[0][1], 6, accuracy: 1e-5)
        XCTAssertEqual(outputs[1][0], 3, accuracy: 1e-5)
        XCTAssertEqual(outputs[1][1], 7, accuracy: 1e-5)
    }

    func testZoomFactorProducesExpectedLength() {
        let input = (0..<10).map(Float.init)
        let zoomed = WeightInterpolation.zoom(input, factor: 0.5)

        XCTAssertEqual(zoomed.count, 5)
    }
}

@available(macOS 13.0, iOS 16.0, *)
final class VDSPOperationsTests: XCTestCase {

    func testL2NormalizeProducesUnitVector() {
        let input: [Float] = [3, 4]
        let normalized = VDSPOperations.l2Normalize(input)

        XCTAssertEqual(normalized[0], 0.6, accuracy: 1e-6)
        XCTAssertEqual(normalized[1], 0.8, accuracy: 1e-6)
        XCTAssertEqual(VDSPOperations.dotProduct(normalized, normalized), 1, accuracy: 1e-5)
    }

    func testMatrixVectorMultiplyMatchesManualComputation() {
        let matrix: [[Float]] = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        let vector: [Float] = [7, 8, 9]

        let result = VDSPOperations.matrixVectorMultiply(matrix: matrix, vector: vector)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0], 50, accuracy: 1e-6)
        XCTAssertEqual(result[1], 122, accuracy: 1e-6)
    }

    func testMatrixMultiplyMatchesExpected() {
        let a: [[Float]] = [
            [1, 2],
            [3, 4],
        ]
        let b: [[Float]] = [
            [5, 6, 7],
            [8, 9, 10],
        ]

        let result = VDSPOperations.matrixMultiply(a: a, b: b)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0].count, 3)
        XCTAssertEqual(result[1].count, 3)
        XCTAssertEqual(result[0][0], 21, accuracy: 1e-6)
        XCTAssertEqual(result[0][1], 24, accuracy: 1e-6)
        XCTAssertEqual(result[0][2], 27, accuracy: 1e-6)
        XCTAssertEqual(result[1][0], 47, accuracy: 1e-6)
        XCTAssertEqual(result[1][1], 54, accuracy: 1e-6)
        XCTAssertEqual(result[1][2], 61, accuracy: 1e-6)
    }

    func testLogSumExpMatchesAnalyticalValue() {
        let vector: [Float] = [0.0, 1.0, 2.0]
        let expected = log(exp(0.0) + exp(1.0) + exp(2.0))

        XCTAssertEqual(VDSPOperations.logSumExp(vector), Float(expected), accuracy: 1e-5)
    }

    func testSoftmaxProducesProbabilityDistribution() {
        let vector: [Float] = [1.0, 2.0, 3.0]
        let result = VDSPOperations.softmax(vector)
        let sum = result.reduce(0, +)

        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
        XCTAssertTrue(result[2] > result[1] && result[1] > result[0])
    }

    func testPairwiseEuclideanDistances() {
        let a: [[Float]] = [
            [0, 0],
            [1, 1],
        ]
        let b: [[Float]] = [
            [0, 1],
            [2, 3],
        ]

        let distances = VDSPOperations.pairwiseEuclideanDistances(a: a, b: b)

        XCTAssertEqual(distances.count, 2)
        // a[0]=[0,0] vs b[0]=[0,1]: sqrt(0^2 + 1^2) = 1
        XCTAssertEqual(distances[0][0], 1, accuracy: 1e-6)
        // a[0]=[0,0] vs b[1]=[2,3]: sqrt(4 + 9) = sqrt(13)
        XCTAssertEqual(distances[0][1], Float(sqrt(13)), accuracy: 1e-6)
        // a[1]=[1,1] vs b[0]=[0,1]: sqrt(1 + 0) = 1
        XCTAssertEqual(distances[1][0], 1, accuracy: 1e-6)
        // a[1]=[1,1] vs b[1]=[2,3]: sqrt(1 + 4) = sqrt(5)
        XCTAssertEqual(distances[1][1], Float(sqrt(5)), accuracy: 1e-6)
    }
}

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
