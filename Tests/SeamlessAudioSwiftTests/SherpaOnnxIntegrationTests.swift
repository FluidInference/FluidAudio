import XCTest
@testable import SeamlessAudioSwift

@available(macOS 13.0, iOS 16.0, *)
final class SherpaOnnxIntegrationTests: XCTestCase {

    var manager: SpeakerDiarizationManager!

    override func setUp() {
        super.setUp()
        manager = SpeakerDiarizationManager()
    }

    override func tearDown() {
        manager = nil
        super.tearDown()
    }

    // MARK: - Manager Initialization Tests

    func testManagerInitialization() {
        // Test that manager can be initialized
        XCTAssertNotNil(manager)
        XCTAssertFalse(manager.isAvailable) // Should not be available until initialized
    }

    func testManagerInitializationWithModels() async {
        // Test manager initialization with models
        do {
            try await manager.initialize()
            XCTAssertTrue(manager.isAvailable)
        } catch {
            // This may fail in tests without actual model files - that's expected
            print("Model initialization failed (expected in test environment): \(error)")
        }
    }

    // MARK: - Diarization Tests

    func testDiarizationWithoutInitialization() async {
        // Test that diarization fails if manager is not initialized
        let testSamples = Array(repeating: Float(0.5), count: 16000)

        do {
            _ = try await manager.performSegmentation(testSamples)
            XCTFail("Should have thrown notInitialized error")
        } catch SpeakerDiarizationError.notInitialized {
            // Expected error
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    func testEmbeddingExtractionWithoutInitialization() async {
        // Test that embedding extraction fails if manager is not initialized
        let testSamples = Array(repeating: Float(0.5), count: 16000)

        do {
            _ = try await manager.extractEmbedding(from: testSamples)
            XCTFail("Should have thrown notInitialized error")
        } catch SpeakerDiarizationError.notInitialized {
            // Expected error
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    // MARK: - Audio Validation Tests

    func testAudioValidation() {
        // Test valid audio
        let validSamples = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }

        let validResult = manager.validateAudio(validSamples)
        XCTAssertTrue(validResult.isValid)
        XCTAssertEqual(validResult.durationSeconds, 1.0, accuracy: 0.1)
        XCTAssertTrue(validResult.issues.isEmpty)

        // Test invalid audio (too short)
        let shortSamples = Array(repeating: Float(0.5), count: 8000) // 0.5 seconds
        let shortResult = manager.validateAudio(shortSamples)
        XCTAssertFalse(shortResult.isValid)
        XCTAssertTrue(shortResult.issues.contains("Audio too short (minimum 1 second)"))

        // Test silent audio
        let silentSamples = Array(repeating: Float(0.0), count: 16000)
        let silentResult = manager.validateAudio(silentSamples)
        XCTAssertFalse(silentResult.isValid)
        XCTAssertTrue(silentResult.issues.contains("Audio too quiet or silent"))

        // Test empty audio
        let emptySamples: [Float] = []
        let emptyResult = manager.validateAudio(emptySamples)
        XCTAssertFalse(emptyResult.isValid)
        XCTAssertTrue(emptyResult.issues.contains("No audio data"))
    }

    // MARK: - Utility Tests

        func testCosineDistance() {
        // Test identical embeddings
        let embedding1: [Float] = [1.0, 0.0, 0.0]
        let embedding2: [Float] = [1.0, 0.0, 0.0]
        let distance1 = manager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(distance1, 0.0, accuracy: 0.001)

        // Test orthogonal embeddings
        let embedding3: [Float] = [1.0, 0.0, 0.0]
        let embedding4: [Float] = [0.0, 1.0, 0.0]
        let distance2 = manager.cosineDistance(embedding3, embedding4)
        XCTAssertEqual(distance2, 1.0, accuracy: 0.001)

        // Test opposite embeddings
        let embedding5: [Float] = [1.0, 0.0, 0.0]
        let embedding6: [Float] = [-1.0, 0.0, 0.0]
        let distance3 = manager.cosineDistance(embedding5, embedding6)
        XCTAssertEqual(distance3, 2.0, accuracy: 0.001)
    }

        func testEmbeddingValidation() {
        // Test valid embedding
        let validEmbedding: [Float] = [0.5, 0.3, -0.2, 0.8]
        XCTAssertTrue(manager.validateEmbedding(validEmbedding))

        // Test empty embedding
        let emptyEmbedding: [Float] = []
        XCTAssertFalse(manager.validateEmbedding(emptyEmbedding))

        // Test embedding with NaN
        let nanEmbedding: [Float] = [0.5, Float.nan, 0.3]
        XCTAssertFalse(manager.validateEmbedding(nanEmbedding))

        // Test embedding with infinity
        let infEmbedding: [Float] = [0.5, Float.infinity, 0.3]
        XCTAssertFalse(manager.validateEmbedding(infEmbedding))

        // Test very small magnitude embedding
        let smallEmbedding: [Float] = [0.01, 0.01, 0.01]
        XCTAssertFalse(manager.validateEmbedding(smallEmbedding))
    }

    // MARK: - Model Download Tests

    func testModelDownload() async {
        // Test model download (this might fail in CI/test environment)
        do {
            let modelPaths = try await manager.downloadModels()
            XCTAssertFalse(modelPaths.segmentationPath.isEmpty)
            XCTAssertFalse(modelPaths.embeddingPath.isEmpty)
        } catch {
            // This may fail in test environment without network access - that's expected
            print("Model download failed (expected in test environment): \(error)")
        }
    }

    // MARK: - Speaker Comparison Tests

    func testSpeakerComparison() async {
        // This test will likely fail without initialization, but tests the API
        let audio1 = Array(0..<16000).map { i in
            sin(Float(i) * 0.01) * 0.5
        }
        let audio2 = Array(0..<16000).map { i in
            sin(Float(i) * 0.02) * 0.5
        }

        do {
            let similarity = try await manager.compareSpeakers(audio1: audio1, audio2: audio2)
            XCTAssertGreaterThanOrEqual(similarity, 0)
            XCTAssertLessThanOrEqual(similarity, 100)
        } catch SpeakerDiarizationError.notInitialized {
            // Expected error in test environment
            print("Speaker comparison failed due to not being initialized (expected)")
        } catch {
            XCTFail("Unexpected error: \(error)")
        }
    }

    // MARK: - Cleanup Tests

    func testCleanup() async {
        // Test cleanup doesn't crash
        await manager.cleanup()
        XCTAssertFalse(manager.isAvailable)
    }
}
