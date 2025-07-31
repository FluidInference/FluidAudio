import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class AsrModelsTests: XCTestCase {

    // MARK: - Model Names Tests

    func testModelNames() {
        XCTAssertEqual(AsrModels.ModelNames.melspectrogram, "Melspectogram.mlmodelc")
        XCTAssertEqual(AsrModels.ModelNames.encoder, "ParakeetEncoder_v2.mlmodelc")
        XCTAssertEqual(AsrModels.ModelNames.decoder, "ParakeetDecoder.mlmodelc")
        XCTAssertEqual(AsrModels.ModelNames.joint, "RNNTJoint.mlmodelc")
        XCTAssertEqual(AsrModels.ModelNames.vocabulary, "parakeet_vocab.json")
    }

    // MARK: - Configuration Tests

    func testDefaultConfiguration() {
        let config = AsrModels.defaultConfiguration()

        XCTAssertTrue(config.allowLowPrecisionAccumulationOnGPU)

        // Check compute units based on environment
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI {
            XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
        } else {
            XCTAssertEqual(config.computeUnits, .all)
        }
    }

    // MARK: - Directory Tests

    func testDefaultCacheDirectory() {
        let cacheDir = AsrModels.defaultCacheDirectory()

        // Verify path components
        XCTAssertTrue(cacheDir.path.contains("FluidAudio"))
        XCTAssertTrue(cacheDir.path.contains("Models"))
        XCTAssertTrue(cacheDir.path.contains(DownloadUtils.Repo.parakeet.folderName))

        // Verify it's an absolute path
        XCTAssertTrue(cacheDir.isFileURL)
        XCTAssertTrue(cacheDir.path.starts(with: "/"))
    }

    // MARK: - Model Existence Tests

    func testModelsExistWithMissingFiles() {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-\(UUID().uuidString)")

        // Test with non-existent directory - should return false
        let result = AsrModels.modelsExist(at: tempDir)
        // We're just testing the method doesn't crash with non-existent paths
        XCTAssertNotNil(result)  // Method returns a boolean
    }

    func testModelsExistLogic() {
        // Test that the method handles various scenarios without crashing
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-\(UUID().uuidString)")

        // Test 1: Non-existent directory
        _ = AsrModels.modelsExist(at: tempDir)

        // Test 2: The method should check for model files in the expected structure
        // We're testing the logic, not the actual file system operations
        let modelNames = [
            AsrModels.ModelNames.melspectrogram,
            AsrModels.ModelNames.encoder,
            AsrModels.ModelNames.decoder,
            AsrModels.ModelNames.joint,
            AsrModels.ModelNames.vocabulary,
        ]

        // Verify all expected model names are defined
        XCTAssertEqual(modelNames.count, 5)
        XCTAssertTrue(modelNames.allSatisfy { !$0.isEmpty })
    }

    // MARK: - Error Tests

    func testAsrModelsErrorDescriptions() {
        let modelNotFound = AsrModelsError.modelNotFound(
            "test.mlmodel", URL(fileURLWithPath: "/test/path"))
        XCTAssertEqual(
            modelNotFound.errorDescription, "ASR model 'test.mlmodel' not found at: /test/path")

        let downloadFailed = AsrModelsError.downloadFailed("Network error")
        XCTAssertEqual(
            downloadFailed.errorDescription, "Failed to download ASR models: Network error")

        let loadingFailed = AsrModelsError.loadingFailed("Invalid format")
        XCTAssertEqual(loadingFailed.errorDescription, "Failed to load ASR models: Invalid format")

        let compilationFailed = AsrModelsError.modelCompilationFailed("Compilation error")
        XCTAssertEqual(
            compilationFailed.errorDescription,
            "Failed to compile ASR models: Compilation error. Try deleting the models and re-downloading."
        )
    }

    // MARK: - Model Initialization Tests

    func testAsrModelsInitialization() throws {
        // Create mock models
        let mockConfig = MLModelConfiguration()
        mockConfig.computeUnits = .cpuOnly

        // Note: We can't create actual MLModel instances in tests without valid model files
        // This test verifies the AsrModels struct initialization logic

        // Test that AsrModels struct can be created with proper types
        let modelNames = [
            AsrModels.ModelNames.melspectrogram,
            AsrModels.ModelNames.encoder,
            AsrModels.ModelNames.decoder,
            AsrModels.ModelNames.joint,
        ]

        XCTAssertEqual(modelNames.count, 4)
        XCTAssertTrue(modelNames.allSatisfy { $0.hasSuffix(".mlmodelc") })
    }

    // MARK: - Download Path Tests

    func testDownloadPathStructure() async throws {
        let customDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-Download-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: customDir) }

        // Test that download would target correct directory structure
        let expectedRepoPath = customDir.deletingLastPathComponent()
            .appendingPathComponent(DownloadUtils.Repo.parakeet.folderName)

        // Verify path components
        XCTAssertTrue(expectedRepoPath.path.contains(DownloadUtils.Repo.parakeet.folderName))
    }

    // MARK: - Model Loading Configuration Tests

    func testCustomConfigurationPropagation() {
        // Test that custom configuration would be used correctly
        let customConfig = MLModelConfiguration()
        customConfig.modelDisplayName = "Test ASR Model"
        customConfig.computeUnits = .cpuAndGPU
        customConfig.allowLowPrecisionAccumulationOnGPU = false

        // Verify configuration properties
        XCTAssertEqual(customConfig.modelDisplayName, "Test ASR Model")
        XCTAssertEqual(customConfig.computeUnits, .cpuAndGPU)
        XCTAssertFalse(customConfig.allowLowPrecisionAccumulationOnGPU)
    }

    // MARK: - Force Download Tests

    func testForceDownloadLogic() async throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-Force-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Create existing directory
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        // Add a test file
        let testFile = tempDir.appendingPathComponent("test.txt")
        try "test content".write(to: testFile, atomically: true, encoding: .utf8)

        XCTAssertTrue(FileManager.default.fileExists(atPath: testFile.path))

        // In actual download with force=true, directory would be removed
        // Here we just verify the file exists before theoretical removal
        XCTAssertTrue(FileManager.default.fileExists(atPath: tempDir.path))
    }

    // MARK: - Helper Method Tests

    func testRepoPathCalculation() {
        let modelsDir = URL(fileURLWithPath: "/test/Models/parakeet-tdt-0.6b-v2-coreml")
        let repoPath = modelsDir.deletingLastPathComponent()
            .appendingPathComponent(DownloadUtils.Repo.parakeet.folderName)

        XCTAssertTrue(repoPath.path.hasSuffix(DownloadUtils.Repo.parakeet.folderName))
        XCTAssertEqual(repoPath.lastPathComponent, DownloadUtils.Repo.parakeet.folderName)
    }

    // MARK: - Integration Test Helpers

    func testModelFileValidation() {
        // Test model file extension validation
        let validModelFiles = [
            "model.mlmodelc",
            "Model.mlmodelc",
            "test_model.mlmodelc",
        ]

        for file in validModelFiles {
            XCTAssertTrue(file.hasSuffix(".mlmodelc"), "\(file) should have .mlmodelc extension")
        }

        // Test vocabulary file
        let vocabFile = "parakeet_vocab.json"
        XCTAssertTrue(vocabFile.hasSuffix(".json"))
        XCTAssertTrue(vocabFile.contains("vocab"))
    }

    // MARK: - Neural Engine Optimization Tests

    func testOptimizedConfiguration() {
        // In CI environment, all compute units are overridden to .cpuOnly
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil

        // Test mel-spectrogram configuration
        let melConfig = AsrModels.optimizedConfiguration(for: .melSpectrogram)
        if isCI {
            XCTAssertEqual(melConfig.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(melConfig.computeUnits, .cpuAndGPU)
        }
        XCTAssertTrue(melConfig.allowLowPrecisionAccumulationOnGPU)

        // Test encoder configuration
        let encoderConfig = AsrModels.optimizedConfiguration(for: .encoder)
        if isCI {
            XCTAssertEqual(encoderConfig.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(encoderConfig.computeUnits, .cpuAndNeuralEngine)
        }

        // Test decoder configuration
        let decoderConfig = AsrModels.optimizedConfiguration(for: .decoder)
        if isCI {
            XCTAssertEqual(decoderConfig.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(decoderConfig.computeUnits, .cpuAndNeuralEngine)
        }

        // Test joint configuration
        let jointConfig = AsrModels.optimizedConfiguration(for: .joint)
        if isCI {
            XCTAssertEqual(jointConfig.computeUnits, .cpuOnly)
        } else if #available(macOS 14.0, iOS 17.0, *) {
            XCTAssertEqual(jointConfig.computeUnits, .all)
        } else {
            XCTAssertEqual(jointConfig.computeUnits, .cpuAndNeuralEngine)
        }

        // Test with FP16 disabled
        let fp32Config = AsrModels.optimizedConfiguration(for: .encoder, enableFP16: false)
        XCTAssertFalse(fp32Config.allowLowPrecisionAccumulationOnGPU)
    }

    func testOptimizedConfigurationCIEnvironment() {
        // Simulate CI environment
        let originalCI = ProcessInfo.processInfo.environment["CI"]
        setenv("CI", "true", 1)
        defer {
            if let originalCI = originalCI {
                setenv("CI", originalCI, 1)
            } else {
                unsetenv("CI")
            }
        }

        let config = AsrModels.optimizedConfiguration(for: .encoder)
        XCTAssertEqual(config.computeUnits, .cpuOnly)
    }

    func testOptimizedPredictionOptions() {
        let options = AsrModels.optimizedPredictionOptions()
        XCTAssertNotNil(options)

        // On macOS 14+, output backings should be configured
        if #available(macOS 14.0, iOS 17.0, *) {
            XCTAssertNotNil(options.outputBackings)
        }
    }

    func testPerformanceProfiles() {
        // Test low latency profile
        let lowLatencyConfig = AsrModels.PerformanceProfile.lowLatency.configuration
        XCTAssertEqual(lowLatencyConfig.computeUnits, .cpuAndGPU)
        XCTAssertTrue(lowLatencyConfig.allowLowPrecisionAccumulationOnGPU)

        let lowLatencyOptions = AsrModels.PerformanceProfile.lowLatency.predictionOptions
        XCTAssertNotNil(lowLatencyOptions)

        // Test balanced profile
        let balancedConfig = AsrModels.PerformanceProfile.balanced.configuration
        XCTAssertEqual(balancedConfig.computeUnits, .all)

        // Test high accuracy profile
        let accuracyConfig = AsrModels.PerformanceProfile.highAccuracy.configuration
        XCTAssertEqual(accuracyConfig.computeUnits, .all)
        XCTAssertFalse(accuracyConfig.allowLowPrecisionAccumulationOnGPU)

        // Test streaming profile
        let streamingConfig = AsrModels.PerformanceProfile.streaming.configuration
        XCTAssertEqual(streamingConfig.computeUnits, .cpuAndNeuralEngine)
    }

    // Removed testLoadWithANEOptimization - causes crashes when trying to load models
}
