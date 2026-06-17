@preconcurrency import CoreML
import Foundation

/// Loaded canary-1b-v2 CoreML models + tokenizer.
///
/// 4 stages from `FluidInference/canary-1b-v2-coreml`:
///   - `preprocessor` (fp32, CPU): waveform [1,240000] → mel [1,128,1501]
///   - `encoder` (int4 ANE / fp16): mel → encoder [1,1024,188]
///   - `decoder` (int4 ANE / fp16): autoregressive transformer → hidden [1,256,1024]
///   - `projection` (fp16, ANE): hidden [1,1024] → logits [1,16384]
///   - `tokenizer`: 16384 SentencePiece pieces (id → piece)
public struct CanaryModels: Sendable {

    public let preprocessor: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let projection: MLModel
    public let tokenizer: Tokenizer

    private static let logger = AppLogger(category: "CanaryModels")

    public init(
        preprocessor: MLModel, encoder: MLModel, decoder: MLModel, projection: MLModel, tokenizer: Tokenizer
    ) {
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.decoder = decoder
        self.projection = projection
        self.tokenizer = tokenizer
    }

    /// Download (if needed) and load all canary models.
    public static func downloadAndLoad(
        precision: CanaryPrecision = .int4,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CanaryModels {
        let directory = try await download(precision: precision, progressHandler: progressHandler)
        return try load(from: directory, precision: precision)
    }

    /// Download the repo into the shared model cache; returns the model directory.
    public static func download(
        precision: CanaryPrecision = .int4,
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = modelsRootDirectory()
        let targetDir = modelsRoot.appendingPathComponent(Repo.canary1bV2.folderName, isDirectory: true)

        if !force && modelsExist(at: targetDir, precision: precision) {
            logger.info("Canary models already present at: \(targetDir.path)")
            return targetDir
        }
        if force { try? FileManager.default.removeItem(at: targetDir) }

        logger.info("Downloading Canary models (\(precision.rawValue)) from HuggingFace...")
        try await DownloadUtils.downloadRepo(
            .canary1bV2, to: modelsRoot, variant: precision.rawValue, progressHandler: progressHandler)
        logger.info("Successfully downloaded Canary models")
        return targetDir
    }

    public static func modelsExist(at directory: URL, precision: CanaryPrecision = .int4) -> Bool {
        let fm = FileManager.default
        let required = ModelNames.Canary.requiredModels(precision: precision)
        return required.allSatisfy { fm.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    /// Load models from a directory that already contains the artifacts.
    public static func load(from directory: URL, precision: CanaryPrecision = .int4) throws -> CanaryModels {
        // Preprocessor runs fp32 on CPU (power-spectrum / log exceed fp16 range).
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits = precision.computeUnits

        let neConfig = MLModelConfiguration()
        neConfig.computeUnits = .cpuAndNeuralEngine

        let preprocessor = try loadModel(
            named: ModelNames.Canary.preprocessor, from: directory, configuration: cpuConfig)
        let encoder = try loadModel(named: precision.encoderName, from: directory, configuration: aneConfig)
        let decoder = try loadModel(named: precision.decoderName, from: directory, configuration: aneConfig)
        let projection = try loadModel(
            named: ModelNames.Canary.projection, from: directory, configuration: neConfig)
        let tokenizer = try Tokenizer(
            vocabPath: directory.appendingPathComponent(ModelNames.Canary.vocabularyFile))

        logger.info("Loaded Canary (encoder/decoder: \(precision.rawValue))")
        return CanaryModels(
            preprocessor: preprocessor, encoder: encoder, decoder: decoder, projection: projection,
            tokenizer: tokenizer)
    }

    // MARK: - Private

    private static func loadModel(
        named name: String, from directory: URL, configuration: MLModelConfiguration
    ) throws -> MLModel {
        let compiledPath = directory.appendingPathComponent("\(name).mlmodelc")
        let packagePath = directory.appendingPathComponent("\(name).mlpackage")
        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledPath.path) {
            modelURL = compiledPath
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            modelURL = try MLModel.compileModel(at: packagePath)
        } else {
            throw ASRError.processingFailed("Canary model not found: \(name)")
        }
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

    private static func modelsRootDirectory() -> URL {
        let fm = FileManager.default
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            return
                appSupport
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return fm.temporaryDirectory
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}
