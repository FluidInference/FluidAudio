@preconcurrency import CoreML
import Foundation
import System
import XCTest

@testable import FluidAudio

final class BasicInitializationTests: XCTestCase {

    func testDiarizerCreation() {
        // Test CoreML diarizer creation
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)
        XCTAssertFalse(manager.isAvailable)  // Not initialized yet
    }

    func testDiarizerWithCustomConfig() {
        // Test CoreML with custom configuration
        let config = DiarizerConfig(
            clusteringThreshold: 0.8,
            minSpeechDuration: 2.0,
            minSilenceGap: 1.0,
            numClusters: 3,
            debugMode: true
        )
        let manager = DiarizerManager(config: config)
        XCTAssertFalse(manager.isAvailable)  // Not initialized yet
    }

    func testDiarizerConfigDefaults() {
        // Test default configuration
        let defaultConfig = DiarizerConfig.default
        XCTAssertEqual(defaultConfig.clusteringThreshold, 0.7, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.minSpeechDuration, 1.0, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.minSilenceGap, 0.5, accuracy: 0.01)
        XCTAssertEqual(defaultConfig.numClusters, -1)
        XCTAssertFalse(defaultConfig.debugMode)
    }
}

// MARK: - CoreML Backend Tests

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
        let shortSamples = Array(repeating: Float(0.5), count: 8000)  // 0.5 seconds

        // Test silent audio
        let silentSamples = Array(repeating: Float(0.0), count: 16000)

        // Test empty audio
        let emptySamples: [Float] = []

        // Test valid audio
        let validResult = manager.validateAudio(validSamples)
        XCTAssertTrue(validResult.isValid, "Valid audio should pass validation")
        XCTAssertEqual(
            validResult.durationSeconds, 1.0, accuracy: 0.1, "Duration should be ~1 second")
        XCTAssertTrue(validResult.issues.isEmpty, "Valid audio should have no issues")

        // Test short audio
        let shortResult = manager.validateAudio(shortSamples)
        XCTAssertFalse(shortResult.isValid, "Short audio should fail validation")
        XCTAssertTrue(
            shortResult.issues.contains("Audio too short (minimum 1 second)"),
            "Short audio should have correct error")

        // Test silent audio
        let silentResult = manager.validateAudio(silentSamples)
        XCTAssertFalse(silentResult.isValid, "Silent audio should fail validation")
        XCTAssertTrue(
            silentResult.issues.contains("Audio too quiet or silent"),
            "Silent audio should have correct error")

        // Test empty audio
        let emptyResult = manager.validateAudio(emptySamples)
        XCTAssertFalse(emptyResult.isValid, "Empty audio should fail validation")
        XCTAssertTrue(
            emptyResult.issues.contains("No audio data"), "Empty audio should have correct error")
    }

    func testCosineDistance() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test identical embeddings
        let embedding1: [Float] = [1.0, 0.0, 0.0]
        let embedding2: [Float] = [1.0, 0.0, 0.0]
        let distance1 = manager.speakerManager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(
            distance1, 0.0, accuracy: 0.001, "Identical embeddings should have 0 distance")

        // Test orthogonal embeddings
        let embedding3: [Float] = [1.0, 0.0, 0.0]
        let embedding4: [Float] = [0.0, 1.0, 0.0]
        let distance2 = manager.speakerManager.cosineDistance(embedding3, embedding4)
        XCTAssertEqual(
            distance2, 1.0, accuracy: 0.001, "Orthogonal embeddings should have distance 1")

        // Test opposite embeddings
        let embedding5: [Float] = [1.0, 0.0, 0.0]
        let embedding6: [Float] = [-1.0, 0.0, 0.0]
        let distance3 = manager.speakerManager.cosineDistance(embedding5, embedding6)
        XCTAssertEqual(
            distance3, 2.0, accuracy: 0.001, "Opposite embeddings should have distance 2")
    }

    func testEmbeddingValidation() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test valid embedding
        let validEmbedding: [Float] = [0.5, 0.3, -0.2, 0.8]
        XCTAssertTrue(
            manager.validateEmbedding(validEmbedding), "Valid embedding should pass validation")

        // Test empty embedding
        let emptyEmbedding: [Float] = []
        XCTAssertFalse(
            manager.validateEmbedding(emptyEmbedding), "Empty embedding should fail validation")

        // Test embedding with NaN
        let nanEmbedding: [Float] = [0.5, Float.nan, 0.3]
        XCTAssertFalse(
            manager.validateEmbedding(nanEmbedding), "NaN embedding should fail validation")

        // Test embedding with infinity
        let infEmbedding: [Float] = [0.5, Float.infinity, 0.3]
        XCTAssertFalse(
            manager.validateEmbedding(infEmbedding), "Infinite embedding should fail validation")

        // Test very small magnitude embedding
        let smallEmbedding: [Float] = [0.01, 0.01, 0.01]
        XCTAssertFalse(
            manager.validateEmbedding(smallEmbedding),
            "Small magnitude embedding should fail validation")
    }

    func testCleanup() {
        let config = DiarizerConfig()
        let manager = DiarizerManager(config: config)

        // Test cleanup doesn't crash
        manager.cleanup()
        XCTAssertFalse(manager.isAvailable, "Manager should not be available after cleanup")
    }
}

// MARK: - Model Loading Tests

extension CoreMLDiarizerTests {

    /// Tests that we can download model files to a user-specified directory and load them from there.
    ///
    func testModelDownloadCustomPath() async throws {

        XCTExpectFailure("Download might fail in CI environment", strict: false)

        // Create a temporary directory.

        let downloadDir = URL.temporaryDirectory.appendingPathComponent(
            "\(Self.self)-testModelDownloadPaths", isDirectory: true)
        try? FileManager.default.removeItem(at: downloadDir)
        try FileManager.default.createDirectory(at: downloadDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: downloadDir) }

        // Download the models to that directory.

        let models = try await DiarizerModels.downloadIfNeeded(to: downloadDir)

        // Check that the models were downloaded successfully
        XCTAssertNotNil(models.segmentationModel)
        XCTAssertNotNil(models.embeddingModel)

        // Check that the model files exist in the download directory
        let repoPath = downloadDir.deletingLastPathComponent()
            .appendingPathComponent(Repo.diarizer.folderName)

        // Clean up any unwanted models that might have been downloaded
        let unwantedModels = ["wespeaker.mlmodelc", "wespeaker_int8.mlmodelc"]
        for modelName in unwantedModels {
            let modelPath = repoPath.appendingPathComponent(modelName)
            try? FileManager.default.removeItem(at: modelPath)
        }

        let segmentationPath = repoPath.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let embeddingPath = repoPath.appendingPathComponent("wespeaker_v2.mlmodelc")

        var isDirectory: ObjCBool = false
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: segmentationPath.path, isDirectory: &isDirectory))
        XCTAssertTrue(isDirectory.boolValue)

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: segmentationPath.appendingPathComponent("coremldata.bin").path,
                isDirectory: &isDirectory))
        XCTAssertFalse(isDirectory.boolValue)

        // Check for embedding model
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: embeddingPath.path, isDirectory: &isDirectory))
        XCTAssertTrue(isDirectory.boolValue)

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: embeddingPath.appendingPathComponent("coremldata.bin").path,
                isDirectory: &isDirectory))
        XCTAssertFalse(isDirectory.boolValue)

        // Consume the models object; we don't need it any more.

        let _ = consume models

        // Load the downloaded models using the predownloaded loader function.
        // We already know that the models are there; it's enough to just check that this works to load them.

        let _ = try await DiarizerModels.load(
            localSegmentationModel: segmentationPath,
            localEmbeddingModel: embeddingPath
        )
    }

    /// Tests that we can download model files to a framework-managed directory and load them from there.
    ///
    func testModelDownloadDefaultPath() async throws {

        XCTExpectFailure("Download might fail in CI environment", strict: false)

        // Download the models to the framework-managed directory.

        let models = try await DiarizerModels.downloadIfNeeded()

        // Check that the models were downloaded successfully
        XCTAssertNotNil(models.segmentationModel)
        XCTAssertNotNil(models.embeddingModel)

        // Get the default models directory
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let modelsDir =
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(Repo.diarizer.folderName, isDirectory: true)

        // Clean up any unwanted models that might have been downloaded
        let unwantedModels = ["wespeaker.mlmodelc", "wespeaker_int8.mlmodelc"]
        for modelName in unwantedModels {
            let modelPath = modelsDir.appendingPathComponent(modelName)
            try? FileManager.default.removeItem(at: modelPath)
        }

        let segmentationPath = modelsDir.appendingPathComponent("pyannote_segmentation.mlmodelc")
        let embeddingPath = modelsDir.appendingPathComponent("wespeaker_v2.mlmodelc")

        // Check that the model files are actually there.
        var isDirectory: ObjCBool = false
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: segmentationPath.path, isDirectory: &isDirectory))
        XCTAssertTrue(isDirectory.boolValue)

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: segmentationPath.appendingPathComponent("coremldata.bin").path,
                isDirectory: &isDirectory))
        XCTAssertFalse(isDirectory.boolValue)

        // Check for embedding model
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: embeddingPath.path, isDirectory: &isDirectory))
        XCTAssertTrue(isDirectory.boolValue)

        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: embeddingPath.appendingPathComponent("coremldata.bin").path,
                isDirectory: &isDirectory))
        XCTAssertFalse(isDirectory.boolValue)

        // Consume the models object; we don't need it any more.

        let _ = consume models

        // Load the downloaded models using the predownloaded loader function.
        // We already know that the models are there; it's enough to just check that this works to load them.

        let _ = try await DiarizerModels.load(
            localSegmentationModel: segmentationPath,
            localEmbeddingModel: embeddingPath
        )
    }
    /// Tests that we can load model files with a user-specified configuration.
    ///
    func testModelLoadingCustomConfig() async throws {

        XCTExpectFailure("Download might fail in CI environment", strict: false)

        // Create a custom configuration.

        let customConfig = MLModelConfiguration()
        customConfig.modelDisplayName = "My custom name"
        customConfig.computeUnits = .cpuOnly

        // Download the models to the framework-managed directory.

        let models = try await DiarizerModels.downloadIfNeeded(configuration: customConfig)

        // Note: The model display name is set during model loading from the .mlmodelc bundle,
        // not from the configuration passed to downloadIfNeeded. The configuration is used
        // for runtime settings like compute units.
        XCTAssertEqual(
            models.segmentationModel.configuration.computeUnits, customConfig.computeUnits)

        XCTAssertEqual(models.embeddingModel.configuration.computeUnits, customConfig.computeUnits)
    }
}

// MARK: - CoreML Backend Specific Test

final class CoreMLBackendIntegrationTests: XCTestCase {

    func testDiarizerCreationAndBasicFunctionality() async {
        // Test that CoreML diarizer can be created with custom config
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minSpeechDuration: 1.0,
            minSilenceGap: 0.5,
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
        XCTAssertEqual(
            validationResult.durationSeconds, 1.0, accuracy: 0.1, "Duration should be ~1 second")

        // Test cosine distance calculation
        let embedding1: [Float] = [1.0, 0.0, 0.0]
        let embedding2: [Float] = [1.0, 0.0, 0.0]
        let distance = diarizer.speakerManager.cosineDistance(embedding1, embedding2)
        XCTAssertEqual(
            distance, 0.0, accuracy: 0.001, "Identical embeddings should have 0 distance")
    }

    func testDiarizerInitializationAttempt() async {
        // Test that initialization attempt works (may fail due to model download but shouldn't crash)
        let config = DiarizerConfig(debugMode: true)
        let diarizer = DiarizerManager(config: config)

        do {
            diarizer.initialize(models: try await .downloadIfNeeded())
            XCTAssertTrue(
                diarizer.isAvailable, "Should be available after successful initialization")

            // Test that we can perform basic operations
            let testSamples = Array(repeating: Float(0.5), count: 16000)

            let _ = try diarizer.performCompleteDiarization(testSamples, sampleRate: 16000)

            diarizer.cleanup()
        } catch {
            // This is expected in test environment - models might not download
            XCTAssertFalse(diarizer.isAvailable, "Should not be available if initialization failed")
        }
    }

    func testModelPaths() async {
        let manager = DiarizerManager()

        do {
            // Initialize to download models
            manager.initialize(models: try await .downloadIfNeeded())

            // Get model paths (this is implementation specific)
            // For CoreML, we'll test that the manager initializes properly
            XCTAssertTrue(manager.isAvailable)
        } catch {
            // This is expected in CI/test environment where models might not be available
            XCTAssertFalse(
                manager.isAvailable, "Manager should not be available if initialization failed")
        }
    }
}
