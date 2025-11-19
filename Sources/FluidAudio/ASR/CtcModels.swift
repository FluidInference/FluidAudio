@preconcurrency import CoreML
import Foundation
import OSLog

/// Container for the Parakeet-TDT CTC 110M CoreML models used for
/// keyword spotting (Argmax-style Custom Vocabulary pipeline).
public struct CtcModels: Sendable {

    public let melSpectrogram: MLModel
    public let encoder: MLModel
    public let configuration: MLModelConfiguration
    public let vocabulary: [Int: String]

    private static let logger = AppLogger(category: "CtcModels")

    public init(
        melSpectrogram: MLModel,
        encoder: MLModel,
        configuration: MLModelConfiguration,
        vocabulary: [Int: String]
    ) {
        self.melSpectrogram = melSpectrogram
        self.encoder = encoder
        self.configuration = configuration
        self.vocabulary = vocabulary
    }
}

extension CtcModels {

    /// Load CTC models from a directory.
    ///
    /// - Parameter directory: Directory containing the downloaded CoreML bundles
    ///   for `parakeet-tdt_ctc-110m`.
    /// - Returns: Loaded `CtcModels` instance.
    public static func load(from directory: URL) async throws -> CtcModels {
        logger.info("Loading CTC models from: \(directory.path)")

        let parentDirectory = directory.deletingLastPathComponent()
        let config = defaultConfiguration()

        // DownloadUtils expects the base directory (without the repo folder) and
        // resolves `Repo.parakeetCtc110m.folderName` internally.
        let modelNames = [
            ModelNames.CTC.melSpectrogramPath,
            ModelNames.CTC.audioEncoderPath,
        ]

        let models = try await DownloadUtils.loadModels(
            .parakeetCtc110m,
            modelNames: modelNames,
            directory: parentDirectory,
            computeUnits: config.computeUnits
        )

        guard
            let melModel = models[ModelNames.CTC.melSpectrogramPath],
            let encoderModel = models[ModelNames.CTC.audioEncoderPath]
        else {
            throw AsrModelsError.loadingFailed("Failed to load CTC MelSpectrogram or AudioEncoder models")
        }

        let vocab = try loadVocabulary(from: directory)

        logger.info("Successfully loaded CTC models and vocabulary")

        return CtcModels(
            melSpectrogram: melModel,
            encoder: encoderModel,
            configuration: config,
            vocabulary: vocab
        )
    }

    /// Download CTC models to the default cache directory (or a custom one) if needed.
    @discardableResult
    public static func download(to directory: URL? = nil, force: Bool = false) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        logger.info("Preparing CTC models at: \(targetDir.path)")

        let parentDir = targetDir.deletingLastPathComponent()

        if !force && modelsExist(at: targetDir) {
            logger.info("CTC models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        _ = try await DownloadUtils.loadModels(
            .parakeetCtc110m,
            modelNames: [
                ModelNames.CTC.melSpectrogramPath,
                ModelNames.CTC.audioEncoderPath,
            ],
            directory: parentDir
        )

        logger.info("Successfully downloaded CTC models")
        return targetDir
    }

    /// Convenience helper that downloads (if needed) and loads the CTC models.
    public static func downloadAndLoad(to directory: URL? = nil) async throws -> CtcModels {
        let targetDir = try await download(to: directory)
        return try await load(from: targetDir)
    }

    /// Default CoreML configuration for CTC inference.
    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        config.computeUnits = .cpuAndNeuralEngine
        return config
    }

    /// Check whether required CTC model bundles and vocabulary exist at a directory.
    public static func modelsExist(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let repoPath = directory

        let required = ModelNames.CTC.requiredModels
        let modelsPresent = required.allSatisfy { fileName in
            let path = repoPath.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }

        let vocabPath = repoPath.appendingPathComponent(ModelNames.CTC.vocabularyPath)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    /// Default cache directory for CTC models (within Application Support).
    public static func defaultCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(Repo.parakeetCtc110m.folderName, isDirectory: true)
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent(ModelNames.CTC.vocabularyPath)
        if !FileManager.default.fileExists(atPath: vocabPath.path) {
            logger.warning(
                "CTC vocabulary file not found at \(vocabPath.path). Ensure the vocab file is downloaded with the models."
            )
            throw AsrModelsError.modelNotFound("CTC vocabulary", vocabPath)
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

            logger.info("Loaded CTC vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch {
            logger.error(
                "Failed to load or parse CTC vocabulary at \(vocabPath.path): \(error.localizedDescription)"
            )
            throw AsrModelsError.loadingFailed("CTC vocabulary parsing failed")
        }
    }
}
