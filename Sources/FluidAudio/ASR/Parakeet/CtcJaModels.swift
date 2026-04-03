@preconcurrency import CoreML
import Foundation

/// Container for Parakeet CTC ja (Japanese) CoreML models (full pipeline)
public struct CtcJaModels: Sendable {

    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]
    public let blankId: Int

    private static let logger = AppLogger(category: "CtcJaModels")

    public init(
        preprocessor: MLModel,
        encoder: MLModel,
        decoder: MLModel,
        configuration: MLModelConfiguration,
        vocabulary: [Int: String],
        blankId: Int = 3072
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.configuration = configuration
        self.vocabulary = vocabulary
        self.blankId = blankId
    }
}

extension CtcJaModels {

    /// Load CTC ja (Japanese) models from a directory.
    ///
    /// - Parameters:
    ///   - directory: Directory containing the downloaded CoreML bundles.
    ///   - configuration: Optional MLModel configuration. When nil, uses default configuration.
    ///   - progressHandler: Optional progress handler for model downloading.
    /// - Returns: Loaded `CtcJaModels` instance.
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CtcJaModels {
        logger.info("Loading CTC ja (Japanese) models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()
        let parentDirectory = directory.deletingLastPathComponent()

        // Load preprocessor, encoder, and decoder
        let modelNames = [
            ModelNames.CTCJa.preprocessorFile,
            ModelNames.CTCJa.encoderFile,
            ModelNames.CTCJa.decoderFile,
        ]

        let models = try await DownloadUtils.loadModels(
            .parakeetCtcJa,
            modelNames: modelNames,
            directory: parentDirectory,
            computeUnits: config.computeUnits,
            progressHandler: progressHandler
        )

        guard
            let preprocessorModel = models[ModelNames.CTCJa.preprocessorFile],
            let encoderModel = models[ModelNames.CTCJa.encoderFile],
            let decoderModel = models[ModelNames.CTCJa.decoderFile]
        else {
            throw AsrModelsError.loadingFailed(
                "Failed to load CTC ja models (preprocessor, encoder, or decoder missing)"
            )
        }

        logger.info("Loaded preprocessor, encoder, and decoder for Japanese")

        // Load vocabulary
        let vocab = try loadVocabulary(from: directory)

        logger.info("Successfully loaded CTC ja models with \(vocab.count) tokens")

        return CtcJaModels(
            preprocessor: preprocessorModel,
            encoder: encoderModel,
            decoder: decoderModel,
            configuration: config,
            vocabulary: vocab,
            blankId: 3072
        )
    }

    /// Download CTC ja (Japanese) models to the default cache directory.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory).
    ///   - force: Whether to force re-download even if models exist.
    ///   - progressHandler: Optional progress handler for download progress.
    /// - Returns: The directory where models were downloaded.
    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        logger.info("Preparing CTC ja (Japanese) models at: \(targetDir.path)")

        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir) {
            logger.info("CTC ja models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        let modelNames = [
            ModelNames.CTCJa.preprocessorFile,
            ModelNames.CTCJa.encoderFile,
            ModelNames.CTCJa.decoderFile,
        ]

        _ = try await DownloadUtils.loadModels(
            .parakeetCtcJa,
            modelNames: modelNames,
            directory: parentDir,
            progressHandler: progressHandler
        )

        logger.info("Successfully downloaded CTC ja models")
        return targetDir
    }

    /// Convenience helper that downloads (if needed) and loads the CTC ja models.
    ///
    /// - Parameters:
    ///   - directory: Custom cache directory (default: uses defaultCacheDirectory).
    ///   - configuration: Optional MLModel configuration.
    ///   - progressHandler: Optional progress handler.
    /// - Returns: Loaded `CtcJaModels` instance.
    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CtcJaModels {
        let targetDir = try await download(
            to: directory,
            progressHandler: progressHandler
        )
        return try await load(
            from: targetDir,
            configuration: configuration,
            progressHandler: progressHandler
        )
    }

    /// Default CoreML configuration for CTC ja inference.
    public static func defaultConfiguration() -> MLModelConfiguration {
        MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuAndNeuralEngine)
    }

    /// Check whether required CTC ja model bundles and vocabulary exist at a directory.
    public static func modelsExist(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let repoPath = directory

        let requiredFiles = [
            ModelNames.CTCJa.preprocessorFile,
            ModelNames.CTCJa.encoderFile,
            ModelNames.CTCJa.decoderFile,
        ]

        let modelsPresent = requiredFiles.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        let vocabPath = repoPath.appendingPathComponent(ModelNames.CTCJa.vocabularyFile)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    /// Default cache directory for CTC ja models (within Application Support).
    public static func defaultCacheDirectory() -> URL {
        MLModelConfigurationUtils.defaultModelsDirectory(for: .parakeetCtcJa)
    }

    /// Load vocabulary from vocab.json in the given directory.
    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent(ModelNames.CTCJa.vocabularyFile)
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
            logger.info("Loaded CTC ja vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
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
            logger.info("Loaded CTC ja vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        }

        throw AsrModelsError.loadingFailed("Failed to parse vocab.json - expected array or dictionary format")
    }
}
