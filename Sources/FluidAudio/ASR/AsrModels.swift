@preconcurrency import CoreML
import Foundation
import OSLog

/// ASR model version enum
public enum AsrModelVersion: Sendable {
    case v2
    case v3

    var repo: Repo {
        switch self {
        case .v2: return .parakeetV2
        case .v3: return .parakeet
        }
    }
}

public struct AsrModels: Sendable {

    /// Required model names for ASR
    public static let requiredModelNames = ModelNames.ASR.requiredModels

    public let encoder: MLModel
    public let preprocessor: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]
    public let version: AsrModelVersion

    private static let logger = AppLogger(category: "AsrModels")

    public init(
        encoder: MLModel,
        preprocessor: MLModel,
        decoder: MLModel,
        joint: MLModel,
        configuration: MLModelConfiguration,
        vocabulary: [Int: String],
        version: AsrModelVersion
    ) {
        self.encoder = encoder
        self.preprocessor = preprocessor
        self.decoder = decoder
        self.joint = joint
        self.configuration = configuration
        self.vocabulary = vocabulary
        self.version = version
    }

    public var usesSplitFrontend: Bool {
        true
    }
}

extension AsrModels {

    private struct ModelSpec {
        let fileName: String
        let computeUnits: MLComputeUnits
    }

    private static func createModelSpecs(using config: MLModelConfiguration) -> [ModelSpec] {
        return [
            // Preprocessor ops map to CPU-only across all platforms. XCode profiling shows
            // that 100% of the the operations map to the CPU anyways.
            ModelSpec(fileName: Names.preprocessorFile, computeUnits: .cpuOnly),

            ModelSpec(fileName: Names.encoderFile, computeUnits: config.computeUnits),
        ]
    }

    /// Helper to get the repo path from a models directory
    private static func repoPath(from modelsDirectory: URL, version: AsrModelVersion = .v3) -> URL {
        return modelsDirectory.deletingLastPathComponent()
            .appendingPathComponent(version.repo.folderName)
    }

    private static func inferredVersion(from directory: URL) -> AsrModelVersion? {
        let directoryPath = directory.path.lowercased()
        let knownVersions: [AsrModelVersion] = [.v2, .v3]

        for version in knownVersions {
            if directoryPath.contains(version.repo.folderName.lowercased()) {
                return version
            }
        }

        return nil
    }

    // Use centralized model names
    private typealias Names = ModelNames.ASR

    /// Load ASR models from a directory
    ///
    /// - Parameters:
    ///   - directory: Directory containing the model files
    ///   - configuration: Optional MLModel configuration. When provided, the configuration's
    ///                   computeUnits will be respected. When nil, platform-optimized defaults
    ///                   are used (per-model optimization based on model type).
    ///   - version: ASR model version to load (defaults to v3)
    ///
    /// - Returns: Loaded ASR models
    ///
    /// - Note: The default configuration pins the preprocessor to CPU and every other
    ///         Parakeet component to `.cpuAndNeuralEngine` to avoid GPU dispatch, which keeps
    ///         background execution permitted on iOS.
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3
    ) async throws -> AsrModels {
        logger.info("Loading ASR models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()

        let parentDirectory = directory.deletingLastPathComponent()
        // Load preprocessor and encoder first; decoder and joint are loaded below as well.
        let specs = createModelSpecs(using: config)

        var loadedModels: [String: MLModel] = [:]

        for spec in specs {
            let models = try await DownloadUtils.loadModels(
                version.repo,
                modelNames: [spec.fileName],
                directory: parentDirectory,
                computeUnits: spec.computeUnits
            )

            if let model = models[spec.fileName] {
                loadedModels[spec.fileName] = model
                let unitsText = Self.describeComputeUnits(spec.computeUnits)
                logger.info("Loaded \(spec.fileName) with compute units: \(unitsText)")
            }
        }

        guard let preprocessorModel = loadedModels[Names.preprocessorFile],
            let encoderModel = loadedModels[Names.encoderFile]
        else {
            throw AsrModelsError.loadingFailed("Failed to load preprocessor or encoder model")
        }

        // Load decoder and joint as well
        let decoderAndJoint = try await DownloadUtils.loadModels(
            version.repo,
            modelNames: [Names.decoderFile, Names.jointFile],
            directory: parentDirectory,
            computeUnits: config.computeUnits
        )

        guard let decoderModel = decoderAndJoint[Names.decoderFile],
            let jointModel = decoderAndJoint[Names.jointFile]
        else {
            throw AsrModelsError.loadingFailed("Failed to load decoder or joint model")
        }

        let asrModels = AsrModels(
            encoder: encoderModel,
            preprocessor: preprocessorModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: config,
            vocabulary: try loadVocabulary(from: directory, version: version),
            version: version
        )
        logger.info("Successfully loaded all ASR models with optimized compute units")
        return asrModels
    }

    private static func loadVocabulary(from directory: URL, version: AsrModelVersion) throws -> [Int: String] {
        let vocabPath = repoPath(from: directory, version: version).appendingPathComponent(
            Names.vocabulary(for: version.repo))

        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning(
                "Vocabulary file not found at \(vocabPath.path). Please ensure the vocab file is downloaded with the models."
            )
            throw AsrModelsError.modelNotFound(Names.vocabulary(for: version.repo), vocabPath)
        }

        do {
            let data = try Data(contentsOf: vocabPath)
            let jsonDict = try JSONSerialization.jsonObject(with: data) as? [String: String] ?? [:]

            var vocabulary: [Int: String] = [:]

            for (key, value) in jsonDict {
                if let tokenId = Int(key) {
                    vocabulary[tokenId] = value
                }
            }

            logger.info("Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch {
            logger.error(
                "Failed to load or parse vocabulary file at \(vocabPath.path): \(error.localizedDescription)"
            )
            throw AsrModelsError.loadingFailed("Vocabulary parsing failed")
        }
    }

    public static func loadFromCache(
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3
    ) async throws -> AsrModels {
        let cacheDir = defaultCacheDirectory(for: version)
        return try await load(from: cacheDir, configuration: configuration, version: version)
    }

    /// Load models with automatic recovery on compilation failures
    public static func loadWithAutoRecovery(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        let targetDir = directory ?? defaultCacheDirectory()
        return try await load(from: targetDir, configuration: configuration)
    }

    /// Load models with ANE-optimized configurations

    private static func describeComputeUnits(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly:
            return "cpuOnly"
        case .cpuAndGPU:
            return "cpuAndGPU"
        case .cpuAndNeuralEngine:
            return "cpuAndNeuralEngine"
        case .all:
            return "all"
        @unknown default:
            return "unknown(\(units.rawValue))"
        }
    }

    public static func loadWithANEOptimization(
        from directory: URL? = nil,
        enableFP16: Bool = true
    ) async throws -> AsrModels {
        let targetDir = directory ?? defaultCacheDirectory()

        logger.info("Loading ASR models with ANE optimization from: \(targetDir.path)")

        // Use the load method that already applies per-model optimizations
        return try await load(from: targetDir, configuration: nil)
    }

    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        // Prefer Neural Engine across platforms for ASR inference to avoid GPU dispatch.
        config.computeUnits = .cpuAndNeuralEngine
        return config
    }

    /// Create optimized configuration for specific model type
    public static func optimizedConfiguration(
        for modelType: ANEOptimizer.ModelType,
        enableFP16: Bool = true
    ) -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = enableFP16
        config.computeUnits = ANEOptimizer.optimalComputeUnits(for: modelType)

        // Enable model-specific optimizations
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if isCI {
            config.computeUnits = .cpuOnly
        }

        return config
    }

    /// Create optimized prediction options for inference
    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        let options = MLPredictionOptions()

        // Enable batching for better GPU utilization
        options.outputBackings = [:]  // Reuse output buffers

        return options
    }
}

extension AsrModels {

    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false,
        version: AsrModelVersion = .v3
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory(for: version)
        logger.info("Downloading ASR models to: \(targetDir.path)")
        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir, version: version) {
            logger.info("ASR models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        struct DownloadSpec {
            let fileName: String
            let computeUnits: MLComputeUnits
        }

        let defaultUnits = defaultConfiguration().computeUnits

        let specs: [DownloadSpec] = [
            // Preprocessor ops map to CPU-only across all platforms.
            DownloadSpec(fileName: Names.preprocessorFile, computeUnits: .cpuOnly),
            DownloadSpec(fileName: Names.encoderFile, computeUnits: defaultUnits),
            DownloadSpec(fileName: Names.decoderFile, computeUnits: defaultUnits),
            DownloadSpec(fileName: Names.jointFile, computeUnits: defaultUnits),
        ]

        for spec in specs {
            _ = try await DownloadUtils.loadModels(
                version.repo,
                modelNames: [spec.fileName],
                directory: parentDir,
                computeUnits: spec.computeUnits
            )
        }

        logger.info("Successfully downloaded ASR models")
        return targetDir
    }

    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        version: AsrModelVersion = .v3
    ) async throws -> AsrModels {
        let targetDir = try await download(to: directory, version: version)
        return try await load(from: targetDir, configuration: configuration, version: version)
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let detectedVersion = inferredVersion(from: directory) ?? .v3
        return modelsExist(at: directory, version: detectedVersion)
    }

    public static func modelsExist(at directory: URL, version: AsrModelVersion) -> Bool {
        let fileManager = FileManager.default
        let requiredFiles = ModelNames.ASR.requiredModels

        // Check in the DownloadUtils repo structure
        let repoPath = repoPath(from: directory, version: version)

        let modelsPresent = requiredFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        // Also check for vocabulary file associated with the version
        let vocabPath = repoPath.appendingPathComponent(Names.vocabulary(for: version.repo))
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    /// Validates that Parakeet models exist and can be loaded.
    /// Throws on Intel Mac since CoreML models require Apple Silicon.
    public static func isModelValid(version: AsrModelVersion = .v3) async throws -> Bool {
        guard SystemInfo.isAppleSilicon else {
            throw ASRError.unsupportedPlatform("Parakeet models require Apple Silicon")
        }

        let cacheDir = defaultCacheDirectory(for: version)
        guard modelsExist(at: cacheDir, version: version) else {
            logger.info("Model validation failed: model files not found")
            return false
        }

        let repoPath = repoPath(from: cacheDir, version: version)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly

        let modelsToValidate = [
            ("Preprocessor", ModelNames.ASR.preprocessorFile),
            ("Encoder", ModelNames.ASR.encoderFile),
            ("Decoder", ModelNames.ASR.decoderFile),
            ("Joint", ModelNames.ASR.jointFile),
        ]

        for (name, fileName) in modelsToValidate {
            let modelPath = repoPath.appendingPathComponent(fileName)
            do {
                _ = try MLModel(contentsOf: modelPath, configuration: config)
            } catch {
                logger.error("Model validation failed: \(name) - \(error.localizedDescription)")
                return false
            }
        }

        return true
    }

    public static func defaultCacheDirectory(for version: AsrModelVersion = .v3) -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(version.repo.folderName, isDirectory: true)
    }

    // Legacy method for backward compatibility
    public static func defaultCacheDirectory() -> URL {
        return defaultCacheDirectory(for: .v3)
    }
}

public enum AsrModelsError: LocalizedError, Sendable {
    case modelNotFound(String, URL)
    case downloadFailed(String)
    case loadingFailed(String)
    case modelCompilationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name, let path):
            return "ASR model '\(name)' not found at: \(path.path)"
        case .downloadFailed(let reason):
            return "Failed to download ASR models: \(reason)"
        case .loadingFailed(let reason):
            return "Failed to load ASR models: \(reason)"
        case .modelCompilationFailed(let reason):
            return
                "Failed to compile ASR models: \(reason). Try deleting the models and re-downloading."
        }
    }
}
