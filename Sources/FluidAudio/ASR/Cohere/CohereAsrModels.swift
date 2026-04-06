@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "CohereAsrModels")

/// Cohere Transcribe model variant (precision).
public enum CohereAsrVariant: String, CaseIterable, Sendable {
    /// Full precision (FP16 weights). ~4.2 GB.
    case fp16
    /// INT8 quantized weights. 2.6x smaller (~1.6 GB), <1% WER increase.
    case int8

    /// Corresponding HuggingFace model repository.
    public var repo: Repo {
        switch self {
        case .fp16: return .cohereTranscribeCoreml
        case .int8: return .cohereTranscribeCoremlInt8
        }
    }
}

// MARK: - Cohere Transcribe CoreML Model Container

/// Holds CoreML model components for Cohere Transcribe ASR.
///
/// Components:
/// - `encoder`: Mel spectrogram -> encoder hidden states (1, 376, 1024)
/// - `decoder`: Cached decoder with self-attention and cross-attention
@available(macOS 14, iOS 17, *)
public struct CohereAsrModels: Sendable {
    public let encoder: MLModel
    public let decoder: MLModel
    public let vocabulary: [Int: String]

    /// Load Cohere Transcribe models from a directory.
    ///
    /// Expected directory structure:
    /// ```
    /// cohere-transcribe/
    ///   cohere_encoder.mlmodelc
    ///   cohere_decoder_cached.mlmodelc
    /// ```
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> CohereAsrModels {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits

        logger.info("Loading Cohere Transcribe models from \(directory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        // Load encoder
        let encoder = try await loadModel(
            named: ModelNames.CohereTranscribe.encoder,
            from: directory,
            configuration: modelConfig
        )

        // Load decoder (stateful - uses CoreML state API)
        let decoder = try await loadModel(
            named: ModelNames.CohereTranscribe.decoderStateful,
            from: directory,
            configuration: modelConfig
        )

        // Load vocabulary
        let vocabulary = try loadVocabulary(from: directory)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Loaded Cohere Transcribe models in \(String(format: "%.2f", elapsed))s")

        return CohereAsrModels(
            encoder: encoder,
            decoder: decoder,
            vocabulary: vocabulary
        )
    }

    /// Load vocabulary from JSON file.
    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent("vocab.json")

        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            logger.error("Vocabulary file not found at \(vocabPath.path)")
            throw CohereAsrError.modelNotFound("vocab.json not found at \(vocabPath.path)")
        }

        do {
            let data = try Data(contentsOf: vocabPath)
            let json = try JSONSerialization.jsonObject(with: data)

            var vocabulary: [Int: String] = [:]

            if let jsonDict = json as? [String: String] {
                // Dictionary format: {"0": "<unk>", "1": "<|nospeech|>", ...}
                for (key, value) in jsonDict {
                    if let tokenId = Int(key) {
                        vocabulary[tokenId] = value
                    }
                }
            } else {
                throw CohereAsrError.modelNotFound("Invalid vocab.json format")
            }

            logger.info("Loaded vocabulary with \(vocabulary.count) tokens from \(vocabPath.path)")
            return vocabulary
        } catch {
            logger.error("Failed to load vocabulary: \(error.localizedDescription)")
            throw CohereAsrError.modelNotFound("Failed to load vocab.json: \(error.localizedDescription)")
        }
    }

    /// Download models from HuggingFace and load them.
    ///
    /// Downloads to the default cache directory if not already present,
    /// then loads all model components.
    public static func downloadAndLoad(
        variant: CohereAsrVariant = .int8,
        to directory: URL? = nil,
        computeUnits: MLComputeUnits = .all,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CohereAsrModels {
        let targetDir = try await download(variant: variant, to: directory, progressHandler: progressHandler)
        return try await load(from: targetDir, computeUnits: computeUnits)
    }

    /// Download Cohere Transcribe models from HuggingFace.
    ///
    /// - Parameters:
    ///   - variant: Model variant to download (`.fp16` or `.int8`).
    ///   - directory: Target directory. Uses default cache directory if nil.
    ///   - force: Force re-download even if models exist.
    ///   - progressHandler: Optional callback for download progress updates.
    /// - Returns: Path to the directory containing the downloaded models.
    @discardableResult
    public static func download(
        variant: CohereAsrVariant = .int8,
        to directory: URL? = nil,
        force: Bool = false,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory(variant: variant)
        let modelsRoot = modelsRootDirectory()

        if !force && modelsExist(at: targetDir) {
            logger.info("Cohere Transcribe \(variant.rawValue) models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            try? FileManager.default.removeItem(at: targetDir)
        }

        logger.info("Downloading Cohere Transcribe \(variant.rawValue) models from HuggingFace...")
        try await DownloadUtils.downloadRepo(variant.repo, to: modelsRoot, progressHandler: progressHandler)
        logger.info("Successfully downloaded Cohere Transcribe \(variant.rawValue) models")
        return targetDir
    }

    /// Check if all required model files exist locally.
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let requiredFiles = [
            ModelNames.CohereTranscribe.encoderFile,
            ModelNames.CohereTranscribe.decoderFile,
            "vocab.json",
        ]
        return requiredFiles.allSatisfy { file in
            fm.fileExists(atPath: directory.appendingPathComponent(file).path)
        }
    }

    /// Root directory for all FluidAudio model caches.
    private static func modelsRootDirectory() -> URL {
        guard
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            // Fallback to temporary directory if application support unavailable
            return FileManager.default.temporaryDirectory
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    /// Default cache directory for Cohere Transcribe models.
    public static func defaultCacheDirectory(variant: CohereAsrVariant = .int8) -> URL {
        modelsRootDirectory()
            .appendingPathComponent(variant.repo.folderName, isDirectory: true)
    }
}

// MARK: - Helpers

@available(macOS 14, iOS 17, *)
extension CohereAsrModels {
    private static func loadModel(
        named name: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiledURL = directory.appendingPathComponent("\(name).mlmodelc")
        let packageURL = directory.appendingPathComponent("\(name).mlpackage")

        // Try .mlmodelc first (faster), fall back to .mlpackage
        if FileManager.default.fileExists(atPath: compiledURL.path) {
            logger.debug("Loading \(name) from compiled model")
            return try await MLModel.load(contentsOf: compiledURL, configuration: configuration)
        } else if FileManager.default.fileExists(atPath: packageURL.path) {
            logger.debug("Loading \(name) from package")
            return try await MLModel.load(contentsOf: packageURL, configuration: configuration)
        } else {
            logger.error("Model not found: \(name)")
            throw CohereAsrError.modelNotFound("Model not found: \(name)")
        }
    }
}

// MARK: - Error

public enum CohereAsrError: Error, LocalizedError {
    case modelNotFound(String)
    case encodingFailed(String)
    case decodingFailed(String)
    case invalidInput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let msg): return "Model not found: \(msg)"
        case .encodingFailed(let msg): return "Encoding failed: \(msg)"
        case .decodingFailed(let msg): return "Decoding failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        }
    }
}
