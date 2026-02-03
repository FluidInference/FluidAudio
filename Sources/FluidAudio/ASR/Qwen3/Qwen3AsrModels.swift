import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "Qwen3AsrModels")

// MARK: - Qwen3-ASR CoreML Model Container (2-model pipeline)

/// Holds CoreML model components for the optimized 2-model Qwen3-ASR pipeline.
///
/// This uses Swift-side embedding lookup from a preloaded weight matrix,
/// eliminating the embedding CoreML model. Reduces CoreML calls from 3 to 2 per token.
///
/// Components:
/// - `audioEncoder`: mel spectrogram -> 1024-dim audio features (single window)
/// - `decoderStateful`: stateful decoder with fused lmHead (outputs logits directly)
/// - `embeddingWeights`: [151936, 1024] float16 matrix for Swift-side embedding lookup
@available(macOS 15, iOS 18, *)
public struct Qwen3AsrModels {
    public let audioEncoder: MLModel
    public let decoderStateful: MLModel
    public let embeddingWeights: EmbeddingWeights
    public let vocabulary: [Int: String]
    public let config: Qwen3AsrConfig

    /// Load Qwen3-ASR models (2-model pipeline with Swift-side embedding) from a directory.
    ///
    /// Expected directory structure:
    /// ```
    /// qwen3-asr/
    ///   qwen3_asr_audio_encoder.mlmodelc
    ///   qwen3_asr_decoder_stateful.mlmodelc
    ///   qwen3_asr_embeddings.bin  (float16 embedding weights)
    ///   vocab.json
    /// ```
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> Qwen3AsrModels {
        let config = Qwen3AsrConfig.default

        // CPU+GPU — ANE degrades both speed and accuracy for this model
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .cpuAndGPU

        logger.info("Loading Qwen3-ASR models (2-model pipeline) from \(directory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        // Load audio encoder
        let audioEncoder = try await loadModel(
            named: "qwen3_asr_audio_encoder",
            from: directory,
            configuration: modelConfig
        )

        // Load stateful decoder (with fused lmHead)
        let decoderStateful = try await loadModel(
            named: "qwen3_asr_decoder_stateful",
            from: directory,
            configuration: modelConfig
        )

        // Load embedding weights for Swift-side lookup
        let embeddingWeights = try loadEmbeddingWeights(from: directory)

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Loaded Qwen3-ASR models (2-model) in \(String(format: "%.2f", elapsed))s")

        // Load vocabulary from tokenizer
        let vocabulary = try loadVocabulary(from: directory)

        return Qwen3AsrModels(
            audioEncoder: audioEncoder,
            decoderStateful: decoderStateful,
            embeddingWeights: embeddingWeights,
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
        let requiredFiles = [
            ModelNames.Qwen3ASR.audioEncoderFile,
            ModelNames.Qwen3ASR.decoderStatefulFile,
            ModelNames.Qwen3ASR.embeddingsFile,
        ]
        return requiredFiles.allSatisfy { file in
            let path = directory.appendingPathComponent(file)
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

    private static func loadEmbeddingWeights(from directory: URL) throws -> EmbeddingWeights {
        let path = directory.appendingPathComponent(ModelNames.Qwen3ASR.embeddingsFile)
        guard FileManager.default.fileExists(atPath: path.path) else {
            throw Qwen3AsrError.modelNotFound("qwen3_asr_embeddings.bin")
        }

        let start = CFAbsoluteTimeGetCurrent()
        let weights = try EmbeddingWeights(contentsOf: path)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info(
            "Loaded embedding weights in \(String(format: "%.2f", elapsed))s (\(weights.vocabSize) x \(weights.hiddenSize))"
        )
        return weights
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

// MARK: - Embedding Weights

/// Preloaded embedding weights for Swift-side token embedding lookup.
/// Eliminates the need for a separate embedding CoreML model.
public final class EmbeddingWeights: Sendable {
    public let vocabSize: Int
    public let hiddenSize: Int
    private let data: Data

    /// Load embedding weights from a binary file.
    /// Format: uint32 vocabSize, uint32 hiddenSize, then float16[vocabSize * hiddenSize]
    public init(contentsOf url: URL) throws {
        let fileData = try Data(contentsOf: url)
        guard fileData.count >= 8 else {
            throw Qwen3AsrError.invalidVocabulary
        }

        // Read header
        let vocab = fileData.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
        let hidden = fileData.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) }
        self.vocabSize = Int(vocab)
        self.hiddenSize = Int(hidden)

        // Verify size
        let expectedSize = 8 + vocabSize * hiddenSize * 2  // header + float16 data
        guard fileData.count == expectedSize else {
            throw Qwen3AsrError.generationFailed(
                "Embedding file size mismatch: expected \(expectedSize), got \(fileData.count)"
            )
        }

        self.data = fileData
    }

    /// Get embedding vector for a token ID.
    /// Returns float32 array of length hiddenSize.
    public func embedding(for tokenId: Int) -> [Float] {
        guard tokenId >= 0, tokenId < vocabSize else {
            return [Float](repeating: 0, count: hiddenSize)
        }

        let offset = 8 + tokenId * hiddenSize * 2  // header + token offset (float16)
        var result = [Float](repeating: 0, count: hiddenSize)

        data.withUnsafeBytes { ptr in
            let f16Ptr = ptr.baseAddress!.advanced(by: offset)
                .assumingMemoryBound(to: UInt16.self)

            for i in 0..<hiddenSize {
                // Convert float16 to float32
                result[i] = float16ToFloat32(f16Ptr[i])
            }
        }

        return result
    }

    /// Get embeddings for multiple token IDs.
    /// Returns [seqLen][hiddenSize] array.
    public func embeddings(for tokenIds: [Int32]) -> [[Float]] {
        tokenIds.map { embedding(for: Int($0)) }
    }

    /// Convert float16 (stored as UInt16) to float32.
    @inline(__always)
    private func float16ToFloat32(_ h: UInt16) -> Float {
        let sign = (h >> 15) & 0x1
        let exp = (h >> 10) & 0x1F
        let mant = h & 0x3FF

        if exp == 0 {
            // Subnormal or zero
            if mant == 0 {
                return sign == 0 ? 0.0 : -0.0
            }
            // Subnormal: value = (-1)^sign * 2^-14 * (mant/1024)
            let f = Float(mant) / 1024.0 * powf(2.0, -14.0)
            return sign == 0 ? f : -f
        } else if exp == 31 {
            // Inf or NaN
            if mant == 0 {
                return sign == 0 ? Float.infinity : -Float.infinity
            }
            return Float.nan
        }

        // Normal: value = (-1)^sign * 2^(exp-15) * (1 + mant/1024)
        let f = (1.0 + Float(mant) / 1024.0) * powf(2.0, Float(Int(exp) - 15))
        return sign == 0 ? f : -f
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
