import XCTest

@testable import FluidAudio

/// CI-specific tests that run reliably in GitHub Actions
/// These tests focus on core functionality that doesn't require model downloads
final class CITests: XCTestCase {

    // MARK: - Package Structure Tests

    func testPackageImports() {
        // Test that all public APIs are accessible
        let _ = DiarizerConfig.default
        let _ = DiarizerManager.self
    }

    func testDiarizerCreation() {
        // Test CoreML diarizer creation works
        let manager1 = DiarizerManager()
        let manager2 = DiarizerManager(config: .default)

        XCTAssertFalse(manager1.isAvailable)  // Not initialized
        XCTAssertFalse(manager2.isAvailable)  // Not initialized
    }

    // MARK: - Configuration Tests

    func testDiarizerConfigDefaults() {
        let defaultConfig = DiarizerConfig.default

        XCTAssertEqual(defaultConfig.clusteringThreshold, 0.7, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.minSpeechDuration, 1.0, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.minSilenceGap, 0.5, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.numClusters, -1)
        XCTAssertEqual(defaultConfig.minActiveFramesCount, 10.0, accuracy: 0.01)
        XCTAssertFalse(defaultConfig.debugMode)
    }

    func testDiarizerConfigCustom() {
        let customConfig = DiarizerConfig(
            clusteringThreshold: 0.8,
            minSpeechDuration: 2.0,
            minSilenceGap: 1.0,
            numClusters: 3,
            minActiveFramesCount: 15.0,
            debugMode: true
        )

        XCTAssertEqual(customConfig.clusteringThreshold, 0.8, accuracy: 0.01)
        XCTAssertEqual(customConfig.minSpeechDuration, 2.0, accuracy: 0.01)
        XCTAssertEqual(customConfig.minSilenceGap, 1.0, accuracy: 0.01)
        XCTAssertEqual(customConfig.numClusters, 3)
        XCTAssertEqual(customConfig.minActiveFramesCount, 15.0, accuracy: 0.01)
        XCTAssertTrue(customConfig.debugMode)
    }

    // MARK: - Data Structure Tests

    func testSpeakerSegmentCreation() {
        let embedding: [Float] = [0.1, 0.2, -0.3, 0.4, -0.5]
        let segment = TimedSpeakerSegment(
            speakerId: "Speaker 1",
            embedding: embedding,
            startTimeSeconds: 10.5,
            endTimeSeconds: 25.3,
            qualityScore: 0.95
        )

        XCTAssertEqual(segment.speakerId, "Speaker 1")
        XCTAssertEqual(segment.startTimeSeconds, 10.5, accuracy: 0.01)
        XCTAssertEqual(segment.endTimeSeconds, 25.3, accuracy: 0.01)
        XCTAssertEqual(segment.qualityScore, 0.95, accuracy: 0.01)
        XCTAssertEqual(segment.durationSeconds, 14.8, accuracy: 0.01)
        XCTAssertEqual(segment.embedding.count, 5)
    }

    // Removed testSpeakerEmbeddingCreation - SpeakerEmbedding struct was redundant and removed

    func testAudioValidationResult() {
        let validResult = AudioValidationResult(
            isValid: true,
            durationSeconds: 30.0,
            issues: []
        )

        let invalidResult = AudioValidationResult(
            isValid: false,
            durationSeconds: 0.5,
            issues: ["Audio too short", "Poor quality"]
        )

        XCTAssertTrue(validResult.isValid)
        XCTAssertEqual(validResult.durationSeconds, 30.0)
        XCTAssertTrue(validResult.issues.isEmpty)

        XCTAssertFalse(invalidResult.isValid)
        XCTAssertEqual(invalidResult.durationSeconds, 0.5)
        XCTAssertEqual(invalidResult.issues.count, 2)
    }

    // MARK: - Error Handling Tests

    func testDiarizerErrorCases() {
        // Test that error enum cases exist and can be created
        let notInitializedError = DiarizerError.notInitialized
        let processingError = DiarizerError.processingFailed("Test error")
        let downloadError = DiarizerError.modelDownloadFailed
        let embeddingError = DiarizerError.embeddingExtractionFailed

        // Verify error descriptions exist
        XCTAssertFalse(notInitializedError.localizedDescription.isEmpty)
        XCTAssertFalse(processingError.localizedDescription.isEmpty)
        XCTAssertFalse(downloadError.localizedDescription.isEmpty)
        XCTAssertFalse(embeddingError.localizedDescription.isEmpty)
    }

    // MARK: - Audio Processing Utilities

    func testSyntheticAudioGeneration() {
        // Test that we can generate test audio for validation
        let sampleRate = 16000
        let duration = 1.0  // 1 second
        let frequency: Float = 440.0  // A4 note

        let samples = (0..<Int(Double(sampleRate) * duration)).map { i in
            sin(2.0 * Float.pi * frequency * Float(i) / Float(sampleRate)) * 0.5
        }

        XCTAssertEqual(samples.count, sampleRate)
        XCTAssertGreaterThan(samples.max() ?? 0, 0.4)  // Should have positive peaks
        XCTAssertLessThan(samples.min() ?? 0, -0.4)  // Should have negative peaks
    }

    // MARK: - CoreML Manager Validation (Without Initialization)

    func testManagerBasicValidation() {
        let manager = DiarizerManager()

        // Test audio validation (doesn't require initialization)
        let validAudio = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }

        let result = manager.validateAudio(validAudio)
        XCTAssertTrue(result.isValid)
        XCTAssertEqual(result.durationSeconds, 1.0, accuracy: 0.1)

        // Test cosine distance calculation
        let embedding1: [Float] = [1.0, 0.0, 0.0]
        let embedding2: [Float] = [0.0, 1.0, 0.0]
        let distance = manager.speakerManager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance, 1.0, accuracy: 0.001)

        // Test embedding validation
        XCTAssertTrue(manager.validateEmbedding([0.5, 0.3, -0.2]))
        XCTAssertFalse(manager.validateEmbedding([]))  // Empty
        XCTAssertFalse(manager.validateEmbedding([Float.nan]))  // NaN
    }

    // MARK: - Model Paths Structure

    func testModelPathsStructure() {
        let modelPaths = ModelPaths(
            segmentationPath: URL(fileURLWithPath: "/path/to/segmentation.mlmodelc"),
            embeddingPath: URL(fileURLWithPath: "/path/to/embedding.mlmodelc")
        )

        XCTAssertEqual(modelPaths.segmentationPath.absoluteString, "file:///path/to/segmentation.mlmodelc")
        XCTAssertEqual(modelPaths.embeddingPath.absoluteString, "file:///path/to/embedding.mlmodelc")
    }

    // MARK: - Backend Enum Tests

    func testManagerTypes() {
        // Test that CoreML manager can be created and basic types exist
        let manager = DiarizerManager()
        XCTAssertFalse(manager.isAvailable)  // Not initialized

        // Test that we can create with custom config
        let config = DiarizerConfig(debugMode: true)
        let debugManager = DiarizerManager(config: config)
        XCTAssertFalse(debugManager.isAvailable)
    }

    // MARK: - Performance Baseline Tests

    func testAudioProcessingPerformance() {
        // Test that basic audio operations are reasonably fast
        let largeAudio = Array(repeating: Float(0.5), count: 240000)  // 15 seconds at 16kHz

        measure {
            let manager = DiarizerManager()
            let _ = manager.validateAudio(largeAudio)
        }
    }

    func testCosineDistancePerformance() {
        // Test cosine distance calculation performance
        let embedding1 = Array(repeating: Float(0.5), count: 512)  // Typical embedding size
        let embedding2 = Array(repeating: Float(0.3), count: 512)

        measure {
            let manager = DiarizerManager()
            let _ = manager.speakerManager.cosineDistance(embedding1, embedding2)
        }
    }
}
