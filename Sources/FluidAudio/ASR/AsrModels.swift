@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct AsrModels: Sendable {

    /// Required model names for ASR
    public static let requiredModelNames = ModelNames.ASR.requiredModels

    public let melEncoder: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "AsrModels")

    public init(
        melEncoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        configuration: MLModelConfiguration,
        vocabulary: [Int: String]
    ) {
        self.melEncoder = melEncoder
        self.decoder = decoder
        self.joint = joint
        self.configuration = configuration
        self.vocabulary = vocabulary
    }
}

@available(macOS 13.0, iOS 16.0, *)
extension AsrModels {

    /// Helper to get the repo path from a models directory
    private static func repoPath(from modelsDirectory: URL) -> URL {
        return modelsDirectory.deletingLastPathComponent()
            .appendingPathComponent(Repo.parakeet.folderName)
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
    ///
    /// - Returns: Loaded ASR models
    ///
    /// - Note: For iOS apps that need background audio processing, consider using
    ///         `iOSBackgroundConfiguration()` or a custom configuration with
    ///         `.cpuAndNeuralEngine` to avoid GPU-related background execution errors.
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        logger.info("Loading ASR models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()

        // Load each model with its optimal compute unit configuration
        let modelConfigs: [(name: String, modelType: ANEOptimizer.ModelType)] = [
            (Names.melEncoderFile, .melEncoder),
            (Names.decoderFile, .decoder),
            (Names.jointFile, .joint),
        ]

        var loadedModels: [String: MLModel] = [:]

        for (modelName, _) in modelConfigs {
            // Use DownloadUtils with optimal compute units
            let models = try await DownloadUtils.loadModels(
                .parakeet,
                modelNames: [modelName],
                directory: directory.deletingLastPathComponent(),
                computeUnits: config.computeUnits
            )

            if let model = models[modelName] {
                loadedModels[modelName] = model
                let computeUnitsDescription = String(describing: config.computeUnits)
                logger.info("Loaded \(modelName) with compute units: \(computeUnitsDescription)")
            }
        }

        guard let melEncoderModel = loadedModels[Names.melEncoderFile],
            let decoderModel = loadedModels[Names.decoderFile],
            let jointModel = loadedModels[Names.jointFile]
        else {
            throw AsrModelsError.loadingFailed("Failed to load one or more ASR models")
        }

        let asrModels = AsrModels(
            melEncoder: melEncoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: config,
            vocabulary: try loadVocabulary(from: directory)
        )

        logger.info("Successfully loaded all ASR models with optimized compute units")
        return asrModels
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = repoPath(from: directory).appendingPathComponent(Names.vocabulary)

        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning(
                "Vocabulary file not found at \(vocabPath.path). Please ensure the vocab file is downloaded with the models."
            )
            throw AsrModelsError.modelNotFound(Names.vocabulary, vocabPath)
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
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        let cacheDir = defaultCacheDirectory()
        return try await load(from: cacheDir, configuration: configuration)
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
        // Always use CPU+ANE for optimal performance
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
        if #available(macOS 14.0, iOS 17.0, *) {
            options.outputBackings = [:]  // Reuse output buffers
        }

        return options
    }
}

@available(macOS 13.0, iOS 16.0, *)
extension AsrModels {

    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        logger.info("Downloading ASR models to: \(targetDir.path)")
        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir) {
            logger.info("ASR models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        // The models will be downloaded to parentDir/parakeet-tdt-0.6b-v3-coreml/
        // by DownloadUtils.loadModels, so we don't need to download separately
        let modelNames = [
            Names.melEncoderFile,
            Names.decoderFile,
            Names.jointFile,
        ]

        // Download models using DownloadUtils (this will download if needed)
        _ = try await DownloadUtils.loadModels(
            .parakeet,
            modelNames: modelNames,
            directory: parentDir,
            computeUnits: defaultConfiguration().computeUnits
        )

        logger.info("Successfully downloaded ASR models")
        return targetDir
    }

    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        let targetDir = try await download(to: directory)
        return try await load(from: targetDir, configuration: configuration)
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let modelFiles = [
            Names.melEncoderFile,
            Names.decoderFile,
            Names.jointFile,
        ]

        // Check in the DownloadUtils repo structure
        let repoPath = repoPath(from: directory)

        let modelsPresent = modelFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        // Also check for vocabulary file
        let vocabPath = repoPath.appendingPathComponent(Names.vocabulary)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    public static func defaultCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(Repo.parakeet.folderName, isDirectory: true)
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
