@preconcurrency import CoreML
import Foundation

/// Container for Parakeet CTC zh-CN CoreML models (full pipeline)
public struct CtcZhCnModels: Sendable {

    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]
    public let blankId: Int

    private static let logger = AppLogger(category: "CtcZhCnModels")

    public init(
        preprocessor: MLModel,
        encoder: MLModel,
        decoder: MLModel,
        configuration: MLModelConfiguration,
        vocabulary: [Int: String],
        blankId: Int = 7000
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.configuration = configuration
        self.vocabulary = vocabulary
        self.blankId = blankId
    }
}

extension CtcZhCnModels {

    /// Load CTC zh-CN models from a directory.
    ///
    /// - Parameters:
    ///   - directory: Directory containing the downloaded CoreML bundles.
    ///   - useInt8Encoder: Whether to use int8 quantized encoder (default: true).
    ///   - configuration: Optional MLModel configuration. When nil, uses default configuration.
    ///   - progressHandler: Optional progress handler for model downloading.
    /// - Returns: Loaded `CtcZhCnModels` instance.
    public static func load(
        from directory: URL,
        useInt8Encoder: Bool = true,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CtcZhCnModels {
        logger.info("Loading CTC zh-CN models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()
        let parentDirectory = directory.deletingLastPathComponent()

        // Load preprocessor, encoder, and decoder
        let encoderFileName =
            useInt8Encoder
            ? ModelNames.CTCZhCn.encoderFile
            : ModelNames.CTCZhCn.encoderFp32File

        let modelNames = [
            ModelNames.CTCZhCn.preprocessorFile,
            encoderFileName,
            ModelNames.CTCZhCn.decoderFile,
        ]

        let models = try await DownloadUtils.loadModels(
            .parakeetCtcZhCn,
            modelNames: modelNames,
            directory: parentDirectory,
            computeUnits: config.computeUnits,
            progressHandler: progressHandler
        )

        guard
            let preprocessorModel = models[ModelNames.CTCZhCn.preprocessorFile],
            let encoderModel = models[encoderFileName],
            let decoderModel = models[ModelNames.CTCZhCn.decoderFile]
        else {
            throw AsrModelsError.loadingFailed(
                "Failed to load CTC zh-CN models (preprocessor, encoder, or decoder missing)"
            )
        }

        logger.info("Loaded preprocessor, encoder (\(useInt8Encoder ? "int8" : "fp32")), and decoder")

        // Load vocabulary
        let vocab = try loadVocabulary(from: directory)

        logger.info("Successfully loaded CTC zh-CN models with \(vocab.count) tokens")

        return CtcZhCnModels(
            preprocessor: preprocessorModel,
            encoder: encoderModel,
            decoder: decoderModel,
            configuration: config,
            vocabulary: vocab,
            blankId: 7000
        )
    }

    /// Download CTC zh-CN models to the default cache directory.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory).
    ///   - useInt8Encoder: Whether to download int8 quantized encoder (default: true).
    ///   - downloadBothEncoders: If true, downloads both int8 and fp32 encoders (default: false).
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
        logger.info("Preparing CTC zh-CN models at: \(targetDir.path)")

        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir) {
            logger.info("CTC zh-CN models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        // Download encoder variant(s)
        let encoderFileName =
            useInt8Encoder
            ? ModelNames.CTCZhCn.encoderFile
            : ModelNames.CTCZhCn.encoderFp32File

        var modelNames = [
            ModelNames.CTCZhCn.preprocessorFile,
            encoderFileName,
            ModelNames.CTCZhCn.decoderFile,
        ]

        // Optionally download both encoder variants
        if downloadBothEncoders {
            let otherEncoder =
                useInt8Encoder
                ? ModelNames.CTCZhCn.encoderFp32File
                : ModelNames.CTCZhCn.encoderFile
            modelNames.append(otherEncoder)
        }

        _ = try await DownloadUtils.loadModels(
            .parakeetCtcZhCn,
            modelNames: modelNames,
            directory: parentDir,
            progressHandler: progressHandler
        )

        logger.info("Successfully downloaded CTC zh-CN models")
        return targetDir
    }

    /// Convenience helper that downloads (if needed) and loads the CTC zh-CN models.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory).
    ///   - useInt8Encoder: Whether to use int8 quantized encoder (default: true).
    ///   - configuration: Optional MLModel configuration.
    ///   - progressHandler: Optional progress handler.
    /// - Returns: Loaded `CtcZhCnModels` instance.
    public static func downloadAndLoad(
        to directory: URL? = nil,
        useInt8Encoder: Bool = true,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CtcZhCnModels {
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

    /// Default CoreML configuration for CTC zh-CN inference.
    public static func defaultConfiguration() -> MLModelConfiguration {
        MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuAndNeuralEngine)
    }

    /// Check whether required CTC zh-CN model bundles and vocabulary exist at a directory.
    public static func modelsExist(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let repoPath = directory

        // Check if at least one encoder variant exists
        let int8EncoderPath = repoPath.appendingPathComponent(ModelNames.CTCZhCn.encoderFile)
        let fp32EncoderPath = repoPath.appendingPathComponent(ModelNames.CTCZhCn.encoderFp32File)
        let encoderExists =
            fileManager.fileExists(atPath: int8EncoderPath.path)
            || fileManager.fileExists(atPath: fp32EncoderPath.path)

        let requiredFiles = [
            ModelNames.CTCZhCn.preprocessorFile,
            ModelNames.CTCZhCn.decoderFile,
        ]

        let modelsPresent = requiredFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        let vocabPath = repoPath.appendingPathComponent(ModelNames.CTCZhCn.vocabularyFile)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return encoderExists && modelsPresent && vocabPresent
    }

    /// Default cache directory for CTC zh-CN models (within Application Support).
    public static func defaultCacheDirectory() -> URL {
        MLModelConfigurationUtils.defaultModelsDirectory(for: .parakeetCtcZhCn)
    }

    /// Load vocabulary from vocab.json in the given directory.
    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent(ModelNames.CTCZhCn.vocabularyFile)
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
            logger.info("Loaded CTC zh-CN vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
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
            logger.info("Loaded CTC zh-CN vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        }

        throw AsrModelsError.loadingFailed("Failed to parse vocab.json - expected array or dictionary format")
    }
}
