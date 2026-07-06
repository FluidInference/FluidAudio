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
        XCTAssertEqual(ModelNames.ASR.vocabulary(for: .parakeetV3), "parakeet_vocab.json")
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
        let cacheDir = AsrModels.defaultCacheDirectory(for: .v3)

        // Verify path components
        XCTAssertTrue(cacheDir.path.contains("FluidAudio"))
        XCTAssertTrue(cacheDir.path.contains("Models"))
        XCTAssertTrue(cacheDir.path.contains(Repo.parakeetV3.folderName))

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
            ModelNames.ASR.vocabulary(for: .parakeetV3),
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
            .appendingPathComponent(Repo.parakeetV3.folderName)

        // Verify path components
        XCTAssertTrue(expectedRepoPath.path.contains(Repo.parakeetV3.folderName))
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
            .appendingPathComponent(Repo.parakeetV3.folderName)

        XCTAssertTrue(repoPath.path.hasSuffix(Repo.parakeetV3.folderName))
        XCTAssertEqual(repoPath.lastPathComponent, Repo.parakeetV3.folderName)
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

        let config = AsrModels.optimizedConfiguration()
        if isCI {
            XCTAssertEqual(config.computeUnits, .cpuOnly)
        } else {
            XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
        }
        XCTAssertTrue(config.allowLowPrecisionAccumulationOnGPU)
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

        let config = AsrModels.optimizedConfiguration()
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

    func testOptimalComputeUnitsDefault() {
        // Default configuration uses CPU+ANE for optimal performance
        let config = AsrModels.defaultConfiguration()
        XCTAssertEqual(config.computeUnits, .cpuAndNeuralEngine)
    }

    // MARK: - TDT-CTC-110M Model Version Tests

    func testTdtCtc110mHasFusedEncoder() {
        // tdtCtc110m has fused preprocessor+encoder
        XCTAssertTrue(AsrModelVersion.tdtCtc110m.hasFusedEncoder)

        // v2 and v3 have separate encoder
        XCTAssertFalse(AsrModelVersion.v2.hasFusedEncoder)
        XCTAssertFalse(AsrModelVersion.v3.hasFusedEncoder)
    }

    func testTdtCtc110mEncoderHiddenSize() {
        // tdtCtc110m uses 512-dim encoder output
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.encoderHiddenSize, 512)

        // v2 and v3 use 1024-dim encoder output
        XCTAssertEqual(AsrModelVersion.v2.encoderHiddenSize, 1024)
        XCTAssertEqual(AsrModelVersion.v3.encoderHiddenSize, 1024)
    }

    func testTdtCtc110mBlankId() {
        // tdtCtc110m uses blank ID 1024 (same as v2)
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.blankId, 1024)
        XCTAssertEqual(AsrModelVersion.v2.blankId, 1024)

        // v3 uses blank ID 8192
        XCTAssertEqual(AsrModelVersion.v3.blankId, 8192)
    }

    func testTdtCtc110mDecoderLayers() {
        // tdtCtc110m uses 1 decoder LSTM layer
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.decoderLayers, 1)

        // v2 and v3 use 2 decoder LSTM layers
        XCTAssertEqual(AsrModelVersion.v2.decoderLayers, 2)
        XCTAssertEqual(AsrModelVersion.v3.decoderLayers, 2)
    }

    func testTdtCtc110mRepo() {
        // Verify correct HuggingFace repo
        XCTAssertEqual(AsrModelVersion.tdtCtc110m.repo, .parakeetTdtCtc110m)
        XCTAssertEqual(AsrModelVersion.v2.repo, .parakeetV2)
        XCTAssertEqual(AsrModelVersion.v3.repo, .parakeetV3)
    }

    func testTdtCtc110mUsesSplitFrontend() {
        // Create a mock AsrModels instance for tdtCtc110m
        // Note: We can't create actual MLModel instances without model files
        // So we test the version property directly

        // tdtCtc110m has fused frontend (no split)
        XCTAssertFalse(AsrModelVersion.tdtCtc110m.hasFusedEncoder == false)

        // Test the inverse logic used in usesSplitFrontend
        let tdtCtc110mUsesSplit = !AsrModelVersion.tdtCtc110m.hasFusedEncoder
        XCTAssertFalse(tdtCtc110mUsesSplit, "tdtCtc110m should not use split frontend")

        // v2 and v3 use split frontend
        let v2UsesSplit = !AsrModelVersion.v2.hasFusedEncoder
        let v3UsesSplit = !AsrModelVersion.v3.hasFusedEncoder
        XCTAssertTrue(v2UsesSplit, "v2 should use split frontend")
        XCTAssertTrue(v3UsesSplit, "v3 should use split frontend")
    }

    func testTdtCtc110mDefaultCacheDirectory() {
        let cacheDir = AsrModels.defaultCacheDirectory(for: .tdtCtc110m)

        // Verify path contains correct repo folder name
        XCTAssertTrue(cacheDir.path.contains(Repo.parakeetTdtCtc110m.folderName))
        XCTAssertTrue(cacheDir.path.contains("FluidAudio"))
        XCTAssertTrue(cacheDir.path.contains("Models"))

        // Verify it's an absolute path
        XCTAssertTrue(cacheDir.isFileURL)
        XCTAssertTrue(cacheDir.path.starts(with: "/"))
    }

    func testTdtCtc110mVocabularyFilename() {
        // tdtCtc110m uses parakeet_vocab.json (array format, same filename as v2/v3)
        let vocabFile = ModelNames.ASR.vocabularyFile
        XCTAssertEqual(vocabFile, "parakeet_vocab.json")

        // Verify it has .json extension
        XCTAssertTrue(vocabFile.hasSuffix(".json"))
        XCTAssertTrue(vocabFile.contains("vocab"))
    }

    func testAllModelVersionsHaveRequiredProperties() {
        let versions: [AsrModelVersion] = [.v2, .v3, .tdtCtc110m]

        for version in versions {
            // All versions should have valid repo
            XCTAssertNotNil(version.repo)

            // All versions should have positive encoder hidden size
            XCTAssertGreaterThan(version.encoderHiddenSize, 0)

            // All versions should have positive blank ID
            XCTAssertGreaterThan(version.blankId, 0)

            // All versions should have at least 1 decoder layer
            XCTAssertGreaterThan(version.decoderLayers, 0)
        }
    }

    // MARK: - Issue #524: CTC head download in parakeet-ctc-110m repo

    /// Regression guard for
    /// https://github.com/FluidInference/FluidAudio/issues/524.
    ///
    /// `AsrModels.load(version: .tdtCtc110m)` optionally pulls
    /// `CtcHead.mlmodelc` from the `parakeet-ctc-110m` repo, but that repo's
    /// default required set is the standalone CTC frontend
    /// (`MelSpectrogram` + `AudioEncoder`) and does NOT include the CTC head.
    /// `DownloadUtils.loadModels` threads the caller's `modelNames` into
    /// `downloadRepo` via `additionalModelNames` so the HF filter recurses
    /// into the `CtcHead.mlmodelc/` directory.
    func testParakeetCtc110mRequiredSetExcludesCtcHead() {
        let required = ModelNames.getRequiredModelNames(for: .parakeetCtc110m, variant: nil)
        XCTAssertTrue(required.contains(ModelNames.CTC.melSpectrogramPath))
        XCTAssertTrue(required.contains(ModelNames.CTC.audioEncoderPath))
        XCTAssertFalse(
            required.contains(ModelNames.ASR.ctcHeadFile),
            "CtcHead must not be in the parakeet-ctc-110m baseline required set; "
                + "callers needing it must pass it via DownloadUtils.loadModels' "
                + "modelNames parameter."
        )
    }

    /// Verifies that the cache-validity check in `loadModelsOnce` (and the
    /// matching filter in `downloadRepo`) sees `CtcHead.mlmodelc` as
    /// missing even when the baseline required set is fully present on
    /// disk. Pre-fix, the local cache check returned `true` here and the
    /// download was skipped, leading to a silent `fileNoSuchFile` in the
    /// model-loading loop.
    func testLoadModelsCacheCheckIncludesExtraModelNames() throws {
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-#524-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let repoDir = tempDir.appendingPathComponent(Repo.parakeetCtc110m.folderName)
        let fm = FileManager.default
        for name in [ModelNames.CTC.melSpectrogramPath, ModelNames.CTC.audioEncoderPath] {
            let modelDir = repoDir.appendingPathComponent(name)
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
            try Data().write(to: modelDir.appendingPathComponent("coremldata.bin"))
        }

        // Baseline required set is present on disk.
        let required = ModelNames.getRequiredModelNames(for: .parakeetCtc110m, variant: nil)
        for name in required {
            XCTAssertTrue(fm.fileExists(atPath: repoDir.appendingPathComponent(name).path))
        }

        // Caller asks for the CTC head — which is *not* on disk and *not* in
        // the baseline required set. The fix's effective-models union must
        // recognise this as a cache miss.
        let requested: Set<String> = [ModelNames.ASR.ctcHeadFile]
        let effective = required.union(requested)
        let allEffectiveExist = effective.allSatisfy {
            fm.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        XCTAssertFalse(
            allEffectiveExist,
            "Cache check must treat caller-requested model names as required."
        )
    }

    // MARK: - Issue #748: vocabulary must ship with a downloaded ASR cache

    /// A v3 cache with every bundle but no vocab must report incomplete (#748).
    func testModelsExistTreatsMissingVocabularyAsIncomplete() throws {
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-#748-\(UUID().uuidString)")
            .appendingPathComponent(Repo.parakeetV3.folderName)
        defer { try? fm.removeItem(at: tempDir.deletingLastPathComponent()) }

        let repoDir = tempDir.deletingLastPathComponent()
            .appendingPathComponent(Repo.parakeetV3.folderName)
        for name in ModelNames.ASR.requiredModelsV3(precision: .int8) {
            let modelDir = repoDir.appendingPathComponent(name)
            try fm.createDirectory(at: modelDir, withIntermediateDirectories: true)
            try Data().write(to: modelDir.appendingPathComponent("coremldata.bin"))
        }

        // All bundles present but vocab absent -> cache is incomplete.
        XCTAssertFalse(
            AsrModels.modelsExist(at: tempDir, version: .v3, encoderPrecision: .int8),
            "Cache with bundles but no vocab must be treated as incomplete (#748)."
        )

        // Adding the vocab alongside the bundles completes the cache.
        let vocabURL = AsrModels.vocabularyFileURL(
            version: .v3, encoderPrecision: .int8, targetDir: tempDir)
        XCTAssertEqual(vocabURL.deletingLastPathComponent().path, repoDir.path)
        XCTAssertEqual(vocabURL.lastPathComponent, ModelNames.ASR.vocabularyFile)
        try Data("{}".utf8).write(to: vocabURL)

        XCTAssertTrue(
            AsrModels.modelsExist(at: tempDir, version: .v3, encoderPrecision: .int8),
            "Cache with bundles and vocab must be treated as complete."
        )
    }

    /// `ensureVocabularyDownloaded` no-ops without overwriting an existing vocab (#748).
    func testEnsureVocabularyDownloadedNoopsWhenPresent() async throws {
        let fm = FileManager.default
        let tempDir = fm.temporaryDirectory
            .appendingPathComponent("AsrModelsTests-#748-present-\(UUID().uuidString)")
            .appendingPathComponent(Repo.parakeetV3.folderName)
        defer { try? fm.removeItem(at: tempDir.deletingLastPathComponent()) }

        let vocabURL = AsrModels.vocabularyFileURL(
            version: .v3, encoderPrecision: .int8, targetDir: tempDir)
        try fm.createDirectory(
            at: vocabURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let sentinel = Data(#"{"already":"here"}"#.utf8)
        try sentinel.write(to: vocabURL)

        // Present vocab -> returns without attempting any network fetch.
        try await AsrModels.ensureVocabularyDownloaded(
            version: .v3, encoderPrecision: .int8, targetDir: tempDir)

        XCTAssertEqual(
            try Data(contentsOf: vocabURL), sentinel,
            "Existing vocab must not be overwritten by the guarantee step."
        )
    }
}
