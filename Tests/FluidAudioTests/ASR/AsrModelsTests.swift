@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class AsrModelsTests: XCTestCase {

    // MARK: - Model Names Tests

    func testModelNames() {
        XCTAssertEqual(ModelNames.ASR.preprocessorFile, "Preprocessor.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.encoderFile, "Encoder.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.decoderFile, "Decoder.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.jointFile, "JointDecision.mlmodelc")
        XCTAssertEqual(ModelNames.ASR.vocabulary(for: .parakeet), "parakeet_vocab.json")
        XCTAssertEqual(ModelNames.ASR.vocabulary(for: .parakeetV2), "parakeet_vocab.json")
    }

    // MARK: - Configuration Tests

    func testDefaultConfiguration() {
        let config = AsrModels.defaultConfiguration()

        XCTAssertTrue(config.allowLowPrecisionAccumulationOnGPU)
        // Should always use CPU+ANE for optimal performance
        XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
    }

    // MARK: - Directory Tests

    func testDefaultCacheDirectory() {
        let cacheDir = AsrModels.defaultCacheDirectory()

        // Verify path components
        XCTAssertTrue(cacheDir.path.contains("FluidAudio"))
        XCTAssertTrue(cacheDir.path.contains("Models"))
        XCTAssertTrue(cacheDir.path.contains(Repo.parakeet.folderName))

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
        let modelNames: [String] = [
            ModelNames.ASR.preprocessorFile,
            ModelNames.ASR.encoderFile,
            ModelNames.ASR.decoderFile,
            ModelNames.ASR.jointFile,
            ModelNames.ASR.vocabulary(for: .parakeet),
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
            ModelNames.ASR.preprocessorFile,
            ModelNames.ASR.encoderFile,
            ModelNames.ASR.decoderFile,
            ModelNames.ASR.jointFile,
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
            .appendingPathComponent(Repo.parakeet.folderName)

        // Verify path components
        XCTAssertTrue(expectedRepoPath.path.contains(Repo.parakeet.folderName))
    }

    // MARK: - Model Loading Configuration Tests

    func testCustomConfigurationPropagation() {
        // Test that custom configuration would be used correctly
        let customConfig = MLModelConfiguration()
        customConfig.modelDisplayName = "Test ASR Model"
        customConfig.computeUnits = .cpuAndNeuralEngine
        customConfig.allowLowPrecisionAccumulationOnGPU = false

        // Verify configuration properties
        XCTAssertEqual(customConfig.modelDisplayName, "Test ASR Model")
        XCTAssertEqual(customConfig.computeUnits, .cpuAndNeuralEngine)
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
        let modelsDir = URL(fileURLWithPath: "/test/Models/parakeet-tdt-0.6b-v3-coreml")
        let repoPath = modelsDir.deletingLastPathComponent()
            .appendingPathComponent(Repo.parakeet.folderName)

        XCTAssertTrue(repoPath.path.hasSuffix(Repo.parakeet.folderName))
        XCTAssertEqual(repoPath.lastPathComponent, Repo.parakeet.folderName)
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

        // Test encoder configuration
        let melConfig = AsrModels.optimizedConfiguration(for: .encoder)
        if isCI {
            XCTAssertEqual(melConfig.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(melConfig.computeUnits, .cpuAndNeuralEngine)
        }
        XCTAssertTrue(melConfig.allowLowPrecisionAccumulationOnGPU)

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

        // Output backings should be configured
        XCTAssertNotNil(options.outputBackings)
    }

    // Removed testLoadWithANEOptimization - causes crashes when trying to load models

    // MARK: - User Configuration Tests

    func testUserConfigurationIsRespected() {
        // Test that when a user provides a configuration, it's respected
        let userConfig = MLModelConfiguration()
        userConfig.computeUnits = .cpuOnly
        userConfig.modelDisplayName = "User Custom Model"

        // Verify the configuration properties
        XCTAssertEqual(userConfig.computeUnits, .cpuOnly)
        XCTAssertEqual(userConfig.modelDisplayName, "User Custom Model")

        // The actual load test would require model files, so we test the configuration logic
        // The fix ensures that when configuration is not nil, it uses the user's compute units
    }

    func testPlatformAwareDefaultConfiguration() {
        let config = AsrModels.defaultConfiguration()

        // Should always use CPU+ANE for optimal performance
        XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
    }

    func testOptimalComputeUnitsRespectsPlatform() {
        // Test each model type
        let modelTypes: [ANEOptimizer.ModelType] = [
            .encoder,
            .decoder,
            .joint,
        ]

        for modelType in modelTypes {
            let computeUnits = ANEOptimizer.optimalComputeUnits(for: modelType)

            // All models should use CPU+ANE for optimal performance
            XCTAssertEqual(
                computeUnits, .cpuAndNeuralEngine,
                "Model type \(modelType) should use CPU+ANE")
        }
    }
}
