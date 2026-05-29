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

/// Coverage for the per-chunk embedding exposure surface added to surface
/// fine-grained data for downstream cluster-purity correction. Exercises the
/// public API contract (default off, opt-in on) plus the internal mapping
/// helper without requiring a full pipeline run.
@available(macOS 13.0, iOS 16.0, *)
final class ChunkEmbeddingExposureTests: XCTestCase {

    func testExposeChunkEmbeddingsDefaultsToFalse() {
        let config = OfflineDiarizerConfig()
        XCTAssertFalse(
            config.exposeChunkEmbeddings,
            "Per-chunk embedding exposure must be opt-in to avoid imposing the "
                + "memory cost on existing callers."
        )
    }

    func testExposeChunkEmbeddingsCanBeEnabledAndPersisted() {
        var config = OfflineDiarizerConfig()
        XCTAssertFalse(config.exposeChunkEmbeddings)
        config.exposeChunkEmbeddings = true
        XCTAssertTrue(config.exposeChunkEmbeddings)
    }

    func testDiarizationResultChunkEmbeddingsDefaultsToNil() {
        let result = DiarizationResult(segments: [])
        XCTAssertNil(result.chunkEmbeddings)
        XCTAssertNil(result.speakerDatabase)
        XCTAssertNil(result.timings)
        XCTAssertTrue(result.segments.isEmpty)
    }

    func testDiarizationResultPreservesProvidedChunkEmbeddings() {
        let chunk = ChunkEmbedding(
            speakerId: "S1",
            chunkIndex: 0,
            speakerIndex: 0,
            startTimeSeconds: 0.0,
            endTimeSeconds: 1.6,
            embedding256: [Float](repeating: 0.1, count: 256)
            // rho128 omitted — defaults to [], representing "no PLDA available"
        )

        let result = DiarizationResult(
            segments: [],
            speakerDatabase: nil,
            chunkEmbeddings: [chunk],
            timings: nil
        )

        XCTAssertEqual(result.chunkEmbeddings?.count, 1)
        XCTAssertEqual(result.chunkEmbeddings?.first?.speakerId, "S1")
        XCTAssertEqual(result.chunkEmbeddings?.first?.startTimeSeconds, 0.0)
        XCTAssertEqual(result.chunkEmbeddings?.first?.endTimeSeconds, 1.6)
        XCTAssertEqual(result.chunkEmbeddings?.first?.embedding256.count, 256)
        XCTAssertEqual(
            result.chunkEmbeddings?.first?.rho128, [],
            "Default rho128 should be empty when no PLDA payload is provided."
        )
    }

    func testChunkEmbeddingCodableRoundTrip() throws {
        let original = ChunkEmbedding(
            speakerId: "S3",
            chunkIndex: 7,
            speakerIndex: 2,
            startTimeSeconds: 5.5,
            endTimeSeconds: 7.1,
            embedding256: (0..<256).map { Float($0) / 255.0 },
            rho128: (0..<128).map { Double($0) / 127.0 }
        )

        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(ChunkEmbedding.self, from: encoded)

        XCTAssertEqual(decoded.speakerId, original.speakerId)
        XCTAssertEqual(decoded.chunkIndex, original.chunkIndex)
        XCTAssertEqual(decoded.speakerIndex, original.speakerIndex)
        XCTAssertEqual(decoded.startTimeSeconds, original.startTimeSeconds)
        XCTAssertEqual(decoded.endTimeSeconds, original.endTimeSeconds)
        XCTAssertEqual(decoded.embedding256, original.embedding256)
        XCTAssertEqual(decoded.rho128, original.rho128)
    }

    func testChunkEmbeddingFieldsRoundTrip() {
        let embedding = (0..<256).map { Float($0) / 255.0 }
        let rho = (0..<128).map { Double($0) / 127.0 }
        let chunk = ChunkEmbedding(
            speakerId: "S2",
            chunkIndex: 42,
            speakerIndex: 1,
            startTimeSeconds: 12.34,
            endTimeSeconds: 13.94,
            embedding256: embedding,
            rho128: rho
        )

        XCTAssertEqual(chunk.speakerId, "S2")
        XCTAssertEqual(chunk.chunkIndex, 42)
        XCTAssertEqual(chunk.speakerIndex, 1)
        XCTAssertEqual(chunk.startTimeSeconds, 12.34, accuracy: 1e-9)
        XCTAssertEqual(chunk.endTimeSeconds, 13.94, accuracy: 1e-9)
        XCTAssertEqual(chunk.embedding256, embedding)
        XCTAssertEqual(chunk.rho128, rho)
    }

    func testBuildPublicChunkEmbeddingsAssignsSpeakerIdsViaClusterPlusOne() {
        let logger = AppLogger(category: "ChunkEmbeddingExposureTests")
        let timed: [TimedEmbedding] = [
            TimedEmbedding(
                chunkIndex: 0, speakerIndex: 0, startFrame: 0, endFrame: 588,
                frameWeights: [], startTime: 0.0, endTime: 1.6,
                embedding256: [Float](repeating: 0.0, count: 4),
                rho128: []
            ),
            TimedEmbedding(
                chunkIndex: 1, speakerIndex: 0, startFrame: 0, endFrame: 588,
                frameWeights: [], startTime: 1.6, endTime: 3.2,
                embedding256: [Float](repeating: 0.5, count: 4),
                rho128: []
            ),
            TimedEmbedding(
                chunkIndex: 2, speakerIndex: 1, startFrame: 0, endFrame: 588,
                frameWeights: [], startTime: 3.2, endTime: 4.8,
                embedding256: [Float](repeating: 1.0, count: 4),
                rho128: []
            ),
        ]
        let assignments = [0, 2, 1]

        let result = OfflineDiarizerManager.buildPublicChunkEmbeddings(
            timedEmbeddings: timed,
            assignments: assignments,
            logger: logger
        )

        XCTAssertEqual(result.count, 3)
        XCTAssertEqual(result.map { $0.speakerId }, ["S1", "S3", "S2"])
        XCTAssertEqual(result[0].startTimeSeconds, 0.0, accuracy: 1e-9)
        XCTAssertEqual(result[1].startTimeSeconds, 1.6, accuracy: 1e-9)
        XCTAssertEqual(result[2].startTimeSeconds, 3.2, accuracy: 1e-9)
        // Confirm all chunk-level fields propagate
        XCTAssertEqual(result[0].chunkIndex, 0)
        XCTAssertEqual(result[1].chunkIndex, 1)
        XCTAssertEqual(result[2].chunkIndex, 2)
        XCTAssertEqual(result[0].speakerIndex, 0)
        XCTAssertEqual(result[2].speakerIndex, 1)
    }

    func testBuildPublicChunkEmbeddingsReturnsEmptyOnLengthMismatch() {
        let logger = AppLogger(category: "ChunkEmbeddingExposureTests")
        let timed: [TimedEmbedding] = [
            TimedEmbedding(
                chunkIndex: 0, speakerIndex: 0, startFrame: 0, endFrame: 588,
                frameWeights: [], startTime: 0.0, endTime: 1.6,
                embedding256: [], rho128: []
            )
        ]
        // Mismatched: 1 timed embedding but 2 assignments
        let assignments = [0, 0]

        let result = OfflineDiarizerManager.buildPublicChunkEmbeddings(
            timedEmbeddings: timed,
            assignments: assignments,
            logger: logger
        )

        XCTAssertTrue(
            result.isEmpty,
            "Mismatched lengths should produce an empty result rather than a partial mapping."
        )
    }

    func testBuildPublicChunkEmbeddingsHandlesEmptyInput() {
        let logger = AppLogger(category: "ChunkEmbeddingExposureTests")
        let result = OfflineDiarizerManager.buildPublicChunkEmbeddings(
            timedEmbeddings: [],
            assignments: [],
            logger: logger
        )
        XCTAssertTrue(result.isEmpty)
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
