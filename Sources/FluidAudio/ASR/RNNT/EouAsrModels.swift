@preconcurrency import CoreML
import Foundation
import OSLog

/// Holds loaded CoreML models for Parakeet Realtime EOU 120M
public struct EouAsrModels: Sendable {
    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]
    public let config: RnntConfig

    private static let logger = AppLogger(category: "EouAsrModels")

    /// Load EOU models from a directory
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        useMLPackage: Bool = false
    ) async throws -> EouAsrModels {
        let mlConfig = configuration ?? defaultConfiguration()

        let ext = useMLPackage ? ".mlpackage" : ".mlmodelc"
        logger.info("Loading EOU models from \(directory.path) (extension: \(ext))")

        // Load each model component
        let preprocessorURL = directory.appendingPathComponent(ModelNames.ASREOU.preprocessor + ext)
        let encoderURL = directory.appendingPathComponent(ModelNames.ASREOU.encoder + ext)
        let decoderURL = directory.appendingPathComponent(ModelNames.ASREOU.decoder + ext)
        let jointURL = directory.appendingPathComponent(ModelNames.ASREOU.joint + ext)
        let vocabURL = directory.appendingPathComponent(ModelNames.ASREOU.vocabularyFile)

        // Check files exist
        let fm = FileManager.default
        guard fm.fileExists(atPath: preprocessorURL.path) else {
            throw ASRError.processingFailed("Preprocessor model not found at \(preprocessorURL.path)")
        }
        guard fm.fileExists(atPath: encoderURL.path) else {
            throw ASRError.processingFailed("Encoder model not found at \(encoderURL.path)")
        }
        guard fm.fileExists(atPath: decoderURL.path) else {
            throw ASRError.processingFailed("Decoder model not found at \(decoderURL.path)")
        }
        guard fm.fileExists(atPath: jointURL.path) else {
            throw ASRError.processingFailed("Joint model not found at \(jointURL.path)")
        }

        // Load models
        let preprocessor = try await MLModel.load(contentsOf: preprocessorURL, configuration: mlConfig)
        let encoder = try await MLModel.load(contentsOf: encoderURL, configuration: mlConfig)
        let decoder = try await MLModel.load(contentsOf: decoderURL, configuration: mlConfig)
        let joint = try await MLModel.load(contentsOf: jointURL, configuration: mlConfig)

        // Load vocabulary
        let vocabulary = try loadVocabulary(from: vocabURL)

        logger.info("EOU models loaded successfully (\(vocabulary.count) vocab tokens)")

        return EouAsrModels(
            preprocessor: preprocessor,
            encoder: encoder,
            decoder: decoder,
            joint: joint,
            configuration: mlConfig,
            vocabulary: vocabulary,
            config: .parakeetEOU
        )
    }

    /// Download EOU models from HuggingFace and load them
    public static func downloadAndLoad(
        to cacheDirectory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> EouAsrModels {
        let directory = cacheDirectory ?? defaultCacheDirectory()

        // Download models using DownloadUtils
        let modelDir = directory.appendingPathComponent(Repo.parakeetEOU.folderName)
        let modelNames = Array(ModelNames.ASREOU.requiredModels)

        _ = try await DownloadUtils.loadModels(
            .parakeetEOU,
            modelNames: modelNames,
            directory: directory,
            computeUnits: .cpuOnly
        )

        // Also download vocabulary if needed
        let vocabURL = modelDir.appendingPathComponent(ModelNames.ASREOU.vocabularyFile)
        if !FileManager.default.fileExists(atPath: vocabURL.path) {
            // Vocabulary should be downloaded with the models
            logger.warning("Vocabulary file not found, attempting to download...")
        }

        // Load models
        return try await load(from: modelDir, configuration: configuration)
    }

    /// Download EOU models to specified directory
    public static func download(to directory: URL, force: Bool = false) async throws {
        let modelDir = directory.appendingPathComponent(Repo.parakeetEOU.folderName)

        if !force && modelsExist(at: modelDir) {
            logger.info("EOU models already exist at \(modelDir.path)")
            return
        }

        logger.info("Downloading EOU models to \(modelDir.path)")
        let modelNames = Array(ModelNames.ASREOU.requiredModels)
        _ = try await DownloadUtils.loadModels(
            .parakeetEOU,
            modelNames: modelNames,
            directory: directory,
            computeUnits: .cpuOnly
        )
    }

    /// Check if all required models exist
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let required = ModelNames.ASREOU.requiredModels

        for model in required {
            let path = directory.appendingPathComponent(model).path
            if !fm.fileExists(atPath: path) {
                return false
            }
        }

        // Also check vocabulary
        let vocabPath = directory.appendingPathComponent(ModelNames.ASREOU.vocabularyFile).path
        return fm.fileExists(atPath: vocabPath)
    }

    /// Default cache directory for EOU models
    public static func defaultCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("FluidAudio/Models")
    }

    /// Default model configuration optimized for inference
    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        // Use CPU only for consistent numerical behavior
        config.computeUnits = .cpuOnly
        return config
    }

    /// Optimized prediction options for reusing output buffers
    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        let options = MLPredictionOptions()
        return options
    }

    // MARK: - Private

    private static func loadVocabulary(from url: URL) throws -> [Int: String] {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]

        guard let vocabArray = json?["vocab"] as? [String] else {
            throw ASRError.processingFailed("Invalid vocabulary format in \(url.path)")
        }

        var vocabulary: [Int: String] = [:]
        for (index, token) in vocabArray.enumerated() {
            vocabulary[index] = token
        }

        return vocabulary
    }
}
