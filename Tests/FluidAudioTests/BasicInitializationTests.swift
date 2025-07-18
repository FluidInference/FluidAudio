import XCTest
@testable import FluidAudio

final class BasicInitializationTests: XCTestCase {

    func testDiarizerCreation() {
        // Test CoreML diarizer creation
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)
        XCTAssertFalse(manager.isAvailable) // Not initialized yet
    }

    func testDiarizerWithCustomConfig() {
        // Test CoreML with custom configuration
        let config = DiarizerConfig(
            clusteringThreshold: 0.8,
            minDurationOn: 2.0,
            minDurationOff: 1.0,
            numClusters: 3,
            debugMode: true
        )
        let manager = DiarizerManager(config: config)
        XCTAssertFalse(manager.isAvailable) // Not initialized yet
    }

    func testDiarizerConfigDefaults() {
        // Test default configuration
        let defaultConfig = DiarizerConfig.default
        XCTAssertEqual(defaultConfig.clusteringThreshold, 0.7, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.minDurationOn, 1.0, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.minDurationOff, 0.5, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.numClusters, -1)
        XCTAssertFalse(defaultConfig.debugMode)
        XCTAssertNil(defaultConfig.modelCacheDirectory)
    }
}

// MARK: - CoreML Backend Tests

@available(macOS 13.0, iOS 16.0, *)
final class CoreMLDiarizerTests: XCTestCase {

    func testInitialization() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        XCTAssertFalse(manager.isAvailable, "Manager should not be available before initialization")
    }

    func testNotInitializedErrors() {
        let testSamples = Array(repeating: Float(0.5), count: 16000)
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test diarization fails when not initialized
        do {
            _ = try manager.performCompleteDiarization(testSamples, sampleRate: 16000)
            XCTFail("Should have thrown notInitialized error")
        } catch DiarizerError.notInitialized {
            // Expected error
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testAudioValidation() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test valid audio
        let validSamples = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }

        // Test invalid audio (too short)
        let shortSamples = Array(repeating: Float(0.5), count: 8000) // 0.5 seconds

        // Test silent audio
        let silentSamples = Array(repeating: Float(0.0), count: 16000)

        // Test empty audio
        let emptySamples: [Float] = []

        // Test valid audio
        let validResult = manager.validateAudio(validSamples)
        XCTAssertTrue(validResult.isValid, "Valid audio should pass validation")
        XCTAssertEqual(validResult.durationSeconds, 1.0, accuracy: 0.1, "Duration should be ~1 second")
        XCTAssertTrue(validResult.issues.isEmpty, "Valid audio should have no issues")

        // Test short audio
        let shortResult = manager.validateAudio(shortSamples)
        XCTAssertFalse(shortResult.isValid, "Short audio should fail validation")
        XCTAssertTrue(shortResult.issues.contains("Audio too short (minimum 1 second)"), "Short audio should have correct error")

        // Test silent audio
        let silentResult = manager.validateAudio(silentSamples)
        XCTAssertFalse(silentResult.isValid, "Silent audio should fail validation")
        XCTAssertTrue(silentResult.issues.contains("Audio too quiet or silent"), "Silent audio should have correct error")

        // Test empty audio
        let emptyResult = manager.validateAudio(emptySamples)
        XCTAssertFalse(emptyResult.isValid, "Empty audio should fail validation")
        XCTAssertTrue(emptyResult.issues.contains("No audio data"), "Empty audio should have correct error")
    }

    func testCosineDistance() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test identical embeddings
        let embedding1: [Float] = [1.0, 0.0, 0.0]
        let embedding2: [Float] = [1.0, 0.0, 0.0]
        let distance1 = manager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance1, 0.0, accuracy: 0.001, "Identical embeddings should have 0 distance")

        // Test orthogonal embeddings
        let embedding3: [Float] = [1.0, 0.0, 0.0]
        let embedding4: [Float] = [0.0, 1.0, 0.0]
        let distance2 = manager.cosineDistance(embedding3, embedding4)
        XCTAssertEqual(distance2, 1.0, accuracy: 0.001, "Orthogonal embeddings should have distance 1")

        // Test opposite embeddings
        let embedding5: [Float] = [1.0, 0.0, 0.0]
        let embedding6: [Float] = [-1.0, 0.0, 0.0]
        let distance3 = manager.cosineDistance(embedding5, embedding6)
        XCTAssertEqual(distance3, 2.0, accuracy: 0.001, "Opposite embeddings should have distance 2")
    }

    func testEmbeddingValidation() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test valid embedding
        let validEmbedding: [Float] = [0.5, 0.3, -0.2, 0.8]
        XCTAssertTrue(manager.validateEmbedding(validEmbedding), "Valid embedding should pass validation")

        // Test empty embedding
        let emptyEmbedding: [Float] = []
        XCTAssertFalse(manager.validateEmbedding(emptyEmbedding), "Empty embedding should fail validation")

        // Test embedding with NaN
        let nanEmbedding: [Float] = [0.5, Float.nan, 0.3]
        XCTAssertFalse(manager.validateEmbedding(nanEmbedding), "NaN embedding should fail validation")

        // Test embedding with infinity
        let infEmbedding: [Float] = [0.5, Float.infinity, 0.3]
        XCTAssertFalse(manager.validateEmbedding(infEmbedding), "Infinite embedding should fail validation")

        // Test very small magnitude embedding
        let smallEmbedding: [Float] = [0.01, 0.01, 0.01]
        XCTAssertFalse(manager.validateEmbedding(smallEmbedding), "Small magnitude embedding should fail validation")
    }

    func testCleanup() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test cleanup doesn't crash
        manager.cleanup()
        XCTAssertFalse(manager.isAvailable, "Manager should not be available after cleanup")
    }

    func testSpeakerComparison() async {
        let audio1 = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }
        let audio2 = Array(0..<16000).map { i in
            sin(Float(i) * 0.02) * 0.5
        }

        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        do {
            let similarity = try await manager.compareSpeakers(audio1: audio1, audio2: audio2)
            XCTAssertGreaterThanOrEqual(similarity, 0, "Similarity should be >= 0")
            XCTAssertLessThanOrEqual(similarity, 100, "Similarity should be <= 100")
        } catch DiarizerError.notInitialized {
            // Expected error in test environment
            print("Speaker comparison failed due to not being initialized (expected)")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testModelDownloadPaths() async {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test model download (this might fail in CI/test environment, but should return valid paths)
        do {
            let modelPaths = try await manager.downloadModels()

            XCTAssertFalse(modelPaths.segmentationPath.isEmpty, "Segmentation path should not be empty")
            XCTAssertFalse(modelPaths.embeddingPath.isEmpty, "Embedding path should not be empty")

            // Verify CoreML model directories
            XCTAssertTrue(modelPaths.segmentationPath.contains("coreml"), "CoreML models should be in coreml directory")

        } catch {
            // This may fail in test environment without network access - that's expected
            print("Model download failed (expected in test environment): \(error)")
        }
    }
}

// MARK: - CoreML Backend Specific Test

@available(macOS 13.0, iOS 16.0, *)
final class CoreMLBackendIntegrationTests: XCTestCase {

    func testDiarizerCreationAndBasicFunctionality() async {
        // Test that CoreML diarizer can be created with custom config
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minDurationOn: 1.0,
            minDurationOff: 0.5,
            numClusters: -1,
            debugMode: true
        )

        let diarizer = DiarizerManager(config: config)

        // Verify basic functionality
        XCTAssertFalse(diarizer.isAvailable, "Should not be available before initialization")

        // Test basic validation functionality (doesn't require initialization)
        let validSamples = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }

        let validationResult = diarizer.validateAudio(validSamples)
        XCTAssertTrue(validationResult.isValid, "Valid audio should pass validation")
        XCTAssertEqual(validationResult.durationSeconds, 1.0, accuracy: 0.1, "Duration should be ~1 second")

        // Test cosine distance calculation
        let embedding1: [Float] = [1.0, 0.0, 0.0]
        let embedding2: [Float] = [1.0, 0.0, 0.0]
        let distance = diarizer.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance, 0.0, accuracy: 0.001, "Identical embeddings should have 0 distance")
    }

    func testDiarizerInitializationAttempt() async {
        // Test that initialization attempt works (may fail due to model download but shouldn't crash)
        let config = DiarizerConfig(debugMode: true)
        let diarizer = DiarizerManager(config: config)

        do {
            try await diarizer.initialize()
            XCTAssertTrue(diarizer.isAvailable, "Should be available after successful initialization")
            print("✅ CoreML diarizer initialized successfully!")

            // Test that we can perform basic operations
            let testSamples = Array(repeating: Float(0.5), count: 16000)

            do {
                let result = try diarizer.performCompleteDiarization(testSamples, sampleRate: 16000)
                print("✅ CoreML diarization completed, found \(result.segments.count) segments")
            } catch {
                print("ℹ️  Segmentation test completed (may need more realistic audio): \(error)")
            }

            diarizer.cleanup()

        } catch {
            // This is expected in test environment - models might not download
            print("ℹ️  CoreML initialization test completed (expected in test environment): \(error)")
            XCTAssertFalse(diarizer.isAvailable, "Should not be available if initialization failed")
        }
    }

    func testModelPaths() async throws {
        let manager = DiarizerManager()

        // Initialize to download models
        try await manager.initialize()

        // Get model paths (this is implementation specific)
        // For CoreML, we'll test that the manager initializes properly
        XCTAssertTrue(manager.isAvailable)
    }
}
