@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct AsrModels {
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
        public static let encoder = "ParakeetEncoder.mlmodelc"
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

        // Use DownloadUtils to load models with auto-recovery
        let modelNames = [
            ModelNames.melspectrogram,
            ModelNames.encoder,
            ModelNames.decoder,
            ModelNames.joint,
        ]

        let models = try await DownloadUtils.loadModels(
            .parakeet,
            modelNames: modelNames,
            directory: directory.deletingLastPathComponent(),
            computeUnits: config.computeUnits
        )

        guard let melModel = models[ModelNames.melspectrogram],
            let encoderModel = models[ModelNames.encoder],
            let decoderModel = models[ModelNames.decoder],
            let jointModel = models[ModelNames.joint]
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

        logger.info("Successfully loaded all ASR models")
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

    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        config.computeUnits = isCI ? .cpuAndNeuralEngine : .all

        return config
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

public enum AsrModelsError: LocalizedError {
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
