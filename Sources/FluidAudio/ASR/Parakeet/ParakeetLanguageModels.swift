@preconcurrency import CoreML
import Foundation

/// Configuration protocol for language-specific Parakeet models
public protocol ParakeetLanguageModelConfig: Sendable {
    /// Blank token ID for CTC/TDT decoding
    static var blankId: Int { get }

    /// HuggingFace repository for this language model
    static var repository: Repo { get }

    /// Human-readable language label for logging
    static var languageLabel: String { get }

    /// Logger category name
    static var loggerCategory: String { get }

    /// Model file names
    static var preprocessorFile: String { get }
    static var encoderFile: String { get }
    static var decoderFile: String { get }
    static var vocabularyFile: String { get }

    /// Optional joint model file (for TDT models)
    static var jointFile: String? { get }

    /// Whether this model supports int8 encoder variant
    static var supportsInt8Encoder: Bool { get }

    /// FP32 encoder file name (only for models with int8 support)
    static var encoderFp32File: String? { get }
}

/// Generic container for language-specific Parakeet CoreML models
public struct ParakeetLanguageModels<Config: ParakeetLanguageModelConfig>: Sendable {

    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let joint: MLModel?
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]
    public let blankId: Int

    private static var logger: AppLogger {
        AppLogger(category: Config.loggerCategory)
    }

    public init(
        preprocessor: MLModel,
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel? = nil,
        configuration: MLModelConfiguration,
        vocabulary: [Int: String],
        blankId: Int
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.configuration = configuration
        self.vocabulary = vocabulary
        self.blankId = blankId
    }
}

extension ParakeetLanguageModels {

    /// Load models from a directory.
    ///
    /// - Parameters:
    ///   - directory: Directory containing the downloaded CoreML bundles.
    ///   - useInt8Encoder: Whether to use int8 quantized encoder (only used if Config.supportsInt8Encoder is true).
    ///   - configuration: Optional MLModel configuration. When nil, uses default configuration.
    ///   - progressHandler: Optional progress handler for model downloading.
    /// - Returns: Loaded model instance.
    public static func load(
        from directory: URL,
        useInt8Encoder: Bool = true,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> ParakeetLanguageModels<Config> {
        logger.info("Loading \(Config.languageLabel) models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()
        let parentDirectory = directory.deletingLastPathComponent()

        // Determine encoder file name (int8 vs fp32 if supported)
        let encoderFileName: String
        if Config.supportsInt8Encoder {
            encoderFileName = useInt8Encoder ? Config.encoderFile : (Config.encoderFp32File ?? Config.encoderFile)
        } else {
            encoderFileName = Config.encoderFile
        }

        // Build model names list
        var modelNames = [
            Config.preprocessorFile,
            encoderFileName,
            Config.decoderFile,
        ]

        if let jointFile = Config.jointFile {
            modelNames.append(jointFile)
        }

        let models = try await DownloadUtils.loadModels(
            Config.repository,
            modelNames: modelNames,
            directory: parentDirectory,
            computeUnits: config.computeUnits,
            progressHandler: progressHandler
        )

        guard
            let preprocessorModel = models[Config.preprocessorFile],
            let encoderModel = models[encoderFileName],
            let decoderModel = models[Config.decoderFile]
        else {
            throw AsrModelsError.loadingFailed(
                "Failed to load \(Config.languageLabel) models (preprocessor, encoder, or decoder missing)"
            )
        }

        // Load joint model if required
        let jointModel: MLModel?
        if let jointFile = Config.jointFile {
            guard let joint = models[jointFile] else {
                throw AsrModelsError.loadingFailed(
                    "Failed to load \(Config.languageLabel) joint model"
                )
            }
            jointModel = joint
        } else {
            jointModel = nil
        }

        let encoderTypeLabel =
            Config.supportsInt8Encoder
            ? (useInt8Encoder ? "int8" : "fp32")
            : "default"
        logger.info(
            "Loaded preprocessor, encoder (\(encoderTypeLabel)), decoder\(jointModel != nil ? ", and joint" : "")")

        // Load vocabulary
        let vocab = try loadVocabulary(from: directory)

        logger.info("Successfully loaded \(Config.languageLabel) models with \(vocab.count) tokens")

        return ParakeetLanguageModels(
            preprocessor: preprocessorModel,
            encoder: encoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: config,
            vocabulary: vocab,
            blankId: Config.blankId
        )
    }

    /// Download models to the default cache directory.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory).
    ///   - useInt8Encoder: Whether to download int8 quantized encoder (only used if Config.supportsInt8Encoder is true).
    ///   - downloadBothEncoders: If true, downloads both int8 and fp32 encoders (only for int8-capable models).
    ///   - force: Whether to force re-download even if models exist.
    ///   - progressHandler: Optional progress handler for download progress.
    /// - Returns: The directory where models were downloaded.
    @discardableResult
    public static func download(
        to directory: URL? = nil,
        useInt8Encoder: Bool = true,
        downloadBothEncoders: Bool = false,
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        logger.info("Preparing \(Config.languageLabel) models at: \(targetDir.path)")

        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir) {
            logger.info("\(Config.languageLabel) models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        // Determine encoder file name(s)
        var modelNames = [Config.preprocessorFile, Config.decoderFile]

        if Config.supportsInt8Encoder {
            let encoderFileName = useInt8Encoder ? Config.encoderFile : (Config.encoderFp32File ?? Config.encoderFile)
            modelNames.append(encoderFileName)

            // Optionally download both encoder variants
            if downloadBothEncoders, let fp32File = Config.encoderFp32File {
                let otherEncoder = useInt8Encoder ? fp32File : Config.encoderFile
                modelNames.append(otherEncoder)
            }
        } else {
            modelNames.append(Config.encoderFile)
        }

        if let jointFile = Config.jointFile {
            modelNames.append(jointFile)
        }

        _ = try await DownloadUtils.loadModels(
            Config.repository,
            modelNames: modelNames,
            directory: parentDir,
            progressHandler: progressHandler
        )

        logger.info("Successfully downloaded \(Config.languageLabel) models")
        return targetDir
    }

    /// Convenience helper that downloads (if needed) and loads the models.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory).
    ///   - useInt8Encoder: Whether to use int8 quantized encoder (only used if Config.supportsInt8Encoder is true).
    ///   - configuration: Optional MLModel configuration.
    ///   - progressHandler: Optional progress handler.
    /// - Returns: Loaded model instance.
    public static func downloadAndLoad(
        to directory: URL? = nil,
        useInt8Encoder: Bool = true,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> ParakeetLanguageModels<Config> {
        let targetDir = try await download(
            to: directory,
            useInt8Encoder: useInt8Encoder,
            progressHandler: progressHandler
        )
        return try await load(
            from: targetDir,
            useInt8Encoder: useInt8Encoder,
            configuration: configuration,
            progressHandler: progressHandler
        )
    }

    /// Default CoreML configuration for inference.
    public static func defaultConfiguration() -> MLModelConfiguration {
        MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuAndNeuralEngine)
    }

    /// Check whether required model bundles and vocabulary exist at a directory.
    public static func modelsExist(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let repoPath = directory

        var requiredFiles = [Config.preprocessorFile, Config.decoderFile]

        // Check encoder(s)
        if Config.supportsInt8Encoder, let fp32File = Config.encoderFp32File {
            // For int8-capable models, check if at least one encoder variant exists
            let int8EncoderPath = repoPath.appendingPathComponent(Config.encoderFile)
            let fp32EncoderPath = repoPath.appendingPathComponent(fp32File)
            let encoderExists =
                fileManager.fileExists(atPath: int8EncoderPath.path)
                || fileManager.fileExists(atPath: fp32EncoderPath.path)

            if !encoderExists {
                return false
            }
        } else {
            requiredFiles.append(Config.encoderFile)
        }

        if let jointFile = Config.jointFile {
            requiredFiles.append(jointFile)
        }

        let modelsPresent = requiredFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        let vocabPath = repoPath.appendingPathComponent(Config.vocabularyFile)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    /// Default cache directory for models (within Application Support).
    public static func defaultCacheDirectory() -> URL {
        MLModelConfigurationUtils.defaultModelsDirectory(for: Config.repository)
    }

    /// Load vocabulary from vocab.json in the given directory.
    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent(Config.vocabularyFile)
        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            throw AsrModelsError.modelNotFound("vocab.json", vocabPath)
        }

        let data = try Data(contentsOf: vocabPath)

        // Try parsing as array first (standard format: ["<unk>", "▁t", "he", ...])
        if let tokenArray = try? JSONSerialization.jsonObject(with: data) as? [String] {
            var vocabulary: [Int: String] = [:]
            for (index, token) in tokenArray.enumerated() {
                vocabulary[index] = token
            }
            logger.info(
                "Loaded \(Config.languageLabel) vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        }

        // Fallback: try parsing as dictionary ({"0": "<unk>", "1": "▁t", ...})
        if let jsonDict = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            var vocabulary: [Int: String] = [:]
            for (key, value) in jsonDict {
                if let tokenId = Int(key) {
                    vocabulary[tokenId] = value
                }
            }
            logger.info(
                "Loaded \(Config.languageLabel) vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        }

        throw AsrModelsError.loadingFailed("Failed to parse vocab.json - expected array or dictionary format")
    }
}
