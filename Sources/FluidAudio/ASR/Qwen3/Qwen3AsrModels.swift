import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "Qwen3AsrModels")

// MARK: - Qwen3-ASR CoreML Model Container

/// Holds all CoreML model components for Qwen3-ASR inference.
///
/// Components:
/// - `audioEncoder`: mel spectrogram -> 1024-dim audio features (single window)
/// - `embedding`: token IDs -> 1024-dim embeddings
/// - `decoderStateful`: stateful decoder with fused lmHead, outputs logits directly (macOS 15+)
public struct Qwen3AsrModels {
    public let audioEncoder: MLModel
    public let embedding: MLModel
    public let decoderStateful: MLModel
    public let vocabulary: [Int: String]
    public let config: Qwen3AsrConfig

    /// Load all Qwen3-ASR CoreML models from a directory.
    ///
    /// Expected directory structure:
    /// ```
    /// qwen3-asr/
    ///   qwen3_asr_audio_encoder.mlmodelc
    ///   qwen3_asr_embedding.mlmodelc
    ///   qwen3_asr_decoder_stateful.mlmodelc  (fused with lmHead)
    ///   vocab.json
    /// ```
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> Qwen3AsrModels {
        let config = Qwen3AsrConfig.default

        // CPU+GPU — ANE degrades both speed and accuracy for this model
        let decodeConfig = MLModelConfiguration()
        decodeConfig.computeUnits = .cpuAndGPU

        let encoderConfig = MLModelConfiguration()
        encoderConfig.computeUnits = .cpuAndGPU

        logger.info("Loading Qwen3-ASR models from \(directory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        // Load audio encoder
        let audioEncoder = try await loadModel(
            named: "qwen3_asr_audio_encoder",
            from: directory,
            configuration: encoderConfig
        )

        // Load embedding
        let embedding = try await loadModel(
            named: "qwen3_asr_embedding",
            from: directory,
            configuration: decodeConfig
        )

        // Load stateful decoder (fused with lmHead, outputs logits directly)
        let decoderStateful = try await loadModel(
            named: "qwen3_asr_decoder_stateful",
            from: directory,
            configuration: decodeConfig
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Loaded all Qwen3-ASR models in \(String(format: "%.2f", elapsed))s")

        // Load vocabulary from tokenizer
        let vocabulary = try loadVocabulary(from: directory)

        return Qwen3AsrModels(
            audioEncoder: audioEncoder,
            embedding: embedding,
            decoderStateful: decoderStateful,
            vocabulary: vocabulary,
            config: config
        )
    }

    /// Download models from HuggingFace and load them.
    ///
    /// Downloads to the default cache directory if not already present,
    /// then loads all model components.
    public static func downloadAndLoad(
        to directory: URL? = nil,
        computeUnits: MLComputeUnits = .all
    ) async throws -> Qwen3AsrModels {
        let targetDir = try await download(to: directory)
        return try await load(from: targetDir, computeUnits: computeUnits)
    }

    /// Download Qwen3-ASR models from HuggingFace.
    ///
    /// - Parameter directory: Target directory. Uses default cache directory if nil.
    /// - Returns: Path to the directory containing the downloaded models.
    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()

        // downloadRepo creates directory/repo.folderName/, so pass the Models root
        let modelsRoot = modelsRootDirectory()

        if !force && modelsExist(at: targetDir) {
            logger.info("Qwen3-ASR models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            try? FileManager.default.removeItem(at: targetDir)
        }

        logger.info("Downloading Qwen3-ASR models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.qwen3Asr, to: modelsRoot)
        logger.info("Successfully downloaded Qwen3-ASR models")
        return targetDir
    }

    /// Check if all required model files exist locally.
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let requiredModels = [
            ModelNames.Qwen3ASR.audioEncoderFile,
            ModelNames.Qwen3ASR.embeddingFile,
            ModelNames.Qwen3ASR.decoderStatefulFile,
        ]
        return requiredModels.allSatisfy { model in
            let path = directory.appendingPathComponent(model)
            return fm.fileExists(atPath: path.path)
        }
    }

    /// Root directory for all FluidAudio model caches.
    private static func modelsRootDirectory() -> URL {
        let appSupport = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    /// Default cache directory for Qwen3-ASR models.
    public static func defaultCacheDirectory() -> URL {
        modelsRootDirectory()
            .appendingPathComponent(Repo.qwen3Asr.folderName, isDirectory: true)
    }

    // MARK: Private

    private static func loadModel(
        named name: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        // Try .mlmodelc first (pre-compiled), then compile .mlpackage on the fly
        let compiledPath = directory.appendingPathComponent("\(name).mlmodelc")
        let packagePath = directory.appendingPathComponent("\(name).mlpackage")

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledPath.path) {
            modelURL = compiledPath
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            // .mlpackage must be compiled to .mlmodelc before loading
            logger.info("Compiling \(name).mlpackage -> .mlmodelc ...")
            let compileStart = CFAbsoluteTimeGetCurrent()
            let compiledURL = try await MLModel.compileModel(at: packagePath)
            let compileElapsed = CFAbsoluteTimeGetCurrent() - compileStart
            logger.info("  \(name): compiled in \(String(format: "%.2f", compileElapsed))s")

            // Move compiled model next to the package for caching
            let cachedCompiledPath = compiledPath
            try? FileManager.default.removeItem(at: cachedCompiledPath)
            try FileManager.default.copyItem(at: compiledURL, to: cachedCompiledPath)
            // Clean up the temp compiled model
            try? FileManager.default.removeItem(at: compiledURL)

            modelURL = cachedCompiledPath
        } else {
            throw Qwen3AsrError.modelNotFound(name)
        }

        let start = CFAbsoluteTimeGetCurrent()
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.debug("  \(name): loaded in \(String(format: "%.2f", elapsed))s")
        return model
    }

    private static func loadVocabulary(from directory: URL) throws -> [Int: String] {
        let vocabPath = directory.appendingPathComponent("vocab.json")
        guard FileManager.default.fileExists(atPath: vocabPath.path) else {
            logger.warning("vocab.json not found at \(vocabPath.path), using empty vocabulary")
            return [:]
        }

        let data = try Data(contentsOf: vocabPath)
        guard let stringToId = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw Qwen3AsrError.invalidVocabulary
        }

        // Invert: token string -> token ID becomes token ID -> token string
        var idToString: [Int: String] = [:]
        for (token, id) in stringToId {
            idToString[id] = token
        }
        logger.info("Loaded vocabulary: \(idToString.count) tokens")
        return idToString
    }
}

// MARK: - Errors

public enum Qwen3AsrError: Error, LocalizedError {
    case modelNotFound(String)
    case invalidVocabulary
    case encoderFailed(String)
    case decoderFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name):
            return "Qwen3-ASR model not found: \(name)"
        case .invalidVocabulary:
            return "Invalid vocabulary file"
        case .encoderFailed(let detail):
            return "Audio encoder failed: \(detail)"
        case .decoderFailed(let detail):
            return "Decoder failed: \(detail)"
        case .generationFailed(let detail):
            return "Generation failed: \(detail)"
        }
    }
}
