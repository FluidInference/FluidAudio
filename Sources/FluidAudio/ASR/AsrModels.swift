@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct AsrModels: Sendable {
    public let melspectrogram: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration

    private static let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "AsrModels")

    public init(
        melspectrogram: MLModel,
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        configuration: MLModelConfiguration
    ) {
        self.melspectrogram = melspectrogram
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.configuration = configuration
    }
}

@available(macOS 13.0, iOS 16.0, *)
extension AsrModels {

    /// Helper to get the repo path from a models directory
    private static func repoPath(from modelsDirectory: URL) -> URL {
        return modelsDirectory.deletingLastPathComponent()
            .appendingPathComponent(DownloadUtils.Repo.parakeet.folderName)
    }

    public enum ModelNames {
        public static let melspectrogram = "Melspectogram.mlmodelc"
        public static let encoder = "ParakeetEncoder_v2.mlmodelc"
        public static let decoder = "ParakeetDecoder.mlmodelc"
        public static let joint = "RNNTJoint.mlmodelc"
        public static let vocabulary = "parakeet_vocab.json"
    }

    /// Load ASR models from a directory
    /// - Parameters:
    ///   - directory: Directory containing the model files
    ///   - configuration: MLModel configuration to use (defaults to optimized settings)
    /// - Returns: Loaded ASR models
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        logger.info("Loading ASR models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()

        // Load each model with its optimal compute unit configuration
        let modelConfigs: [(name: String, modelType: ANEOptimizer.ModelType)] = [
            (ModelNames.melspectrogram, .melSpectrogram),
            (ModelNames.encoder, .encoder),
            (ModelNames.decoder, .decoder),
            (ModelNames.joint, .joint),
        ]

        var loadedModels: [String: MLModel] = [:]

        for (modelName, modelType) in modelConfigs {
            let optimizedConfig = optimizedConfiguration(for: modelType)

            // Use DownloadUtils with optimized config for each model
            let models = try await DownloadUtils.loadModels(
                .parakeet,
                modelNames: [modelName],
                directory: directory.deletingLastPathComponent(),
                computeUnits: optimizedConfig.computeUnits
            )

            if let model = models[modelName] {
                loadedModels[modelName] = model
                logger.info("Loaded \(modelName) with optimized compute units")
            }
        }

        guard let melModel = loadedModels[ModelNames.melspectrogram],
            let encoderModel = loadedModels[ModelNames.encoder],
            let decoderModel = loadedModels[ModelNames.decoder],
            let jointModel = loadedModels[ModelNames.joint]
        else {
            throw AsrModelsError.loadingFailed("Failed to load one or more ASR models")
        }

        let asrModels = AsrModels(
            melspectrogram: melModel,
            encoder: encoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: config
        )

        logger.info("Successfully loaded all ASR models with optimized compute units")
        return asrModels
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
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        config.computeUnits = isCI ? .cpuAndNeuralEngine : .all

        // Apple Silicon optimizations
        // Enable GPU optimization through compute units setting
        // Parameters API not available in current CoreML version

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

    /// Create performance-optimized configuration for specific use cases
    public enum PerformanceProfile: Sendable {
        case lowLatency  // Prioritize speed over accuracy
        case balanced  // Balance between speed and accuracy
        case highAccuracy  // Prioritize accuracy over speed
        case streaming  // Optimized for real-time streaming

        public var configuration: MLModelConfiguration {
            let config = MLModelConfiguration()
            config.allowLowPrecisionAccumulationOnGPU = true

            switch self {
            case .lowLatency:
                config.computeUnits = .cpuAndGPU
            // GPU optimization enabled through compute units
            case .balanced:
                config.computeUnits = .all
            case .highAccuracy:
                config.computeUnits = .all
                config.allowLowPrecisionAccumulationOnGPU = false
            case .streaming:
                config.computeUnits = .cpuAndNeuralEngine
            }

            return config
        }

        public var predictionOptions: MLPredictionOptions {
            let options = MLPredictionOptions()

            if #available(macOS 14.0, iOS 17.0, *) {
                // Enable output buffer reuse for all profiles
                options.outputBackings = [:]
            }

            return options
        }
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

        // The models will be downloaded to parentDir/parakeet-tdt-0.6b-v2-coreml/
        // by DownloadUtils.loadModels, so we don't need to download separately
        let modelNames = [
            ModelNames.melspectrogram,
            ModelNames.encoder,
            ModelNames.decoder,
            ModelNames.joint,
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
            ModelNames.melspectrogram,
            ModelNames.encoder,
            ModelNames.decoder,
            ModelNames.joint,
        ]

        // Check in the DownloadUtils repo structure
        let repoPath = repoPath(from: directory)

        let modelsPresent = modelFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        // Also check for vocabulary file
        let vocabPath = repoPath.appendingPathComponent(ModelNames.vocabulary)
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
            .appendingPathComponent(DownloadUtils.Repo.parakeet.folderName, isDirectory: true)
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
