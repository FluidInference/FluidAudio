import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "Qwen3AsrModelsFull")

// MARK: - Qwen3-ASR Full Model Container (2-model pipeline)

/// Holds CoreML model components for the optimized 2-model Qwen3-ASR pipeline.
///
/// This is the fully-fused variant where the decoder includes both embedding
/// lookup and lmHead projection. Reduces overhead from 3 CoreML calls to 1 per token.
///
/// Components:
/// - `audioEncoder`: mel spectrogram -> 1024-dim audio features (single window)
/// - `decoderFull`: token_ids (int32) -> logits (fused embedding + decoder + lmHead)
@available(macOS 15, iOS 18, *)
public struct Qwen3AsrModelsFull {
    public let audioEncoder: MLModel
    public let decoderFull: MLModel
    public let vocabulary: [Int: String]
    public let config: Qwen3AsrConfig

    /// Load Qwen3-ASR models (2-model full pipeline) from a directory.
    ///
    /// Expected directory structure:
    /// ```
    /// qwen3-asr/
    ///   qwen3_asr_audio_encoder.mlmodelc
    ///   qwen3_asr_decoder_full.mlmodelc  (fused embedding + decoder + lmHead)
    ///   vocab.json
    /// ```
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> Qwen3AsrModelsFull {
        let config = Qwen3AsrConfig.default

        // CPU+GPU — ANE degrades both speed and accuracy for this model
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = .cpuAndGPU

        logger.info("Loading Qwen3-ASR models (full pipeline) from \(directory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        // Load audio encoder
        let audioEncoder = try await loadModel(
            named: "qwen3_asr_audio_encoder",
            from: directory,
            configuration: modelConfig
        )

        // Load full decoder (embedding + decoder layers + lmHead fused)
        let decoderFull = try await loadModel(
            named: "qwen3_asr_decoder_full",
            from: directory,
            configuration: modelConfig
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Loaded Qwen3-ASR models (full) in \(String(format: "%.2f", elapsed))s")

        // Load vocabulary from tokenizer
        let vocabulary = try loadVocabulary(from: directory)

        return Qwen3AsrModelsFull(
            audioEncoder: audioEncoder,
            decoderFull: decoderFull,
            vocabulary: vocabulary,
            config: config
        )
    }

    /// Check if all required model files exist locally.
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let requiredModels = [
            ModelNames.Qwen3ASR.audioEncoderFile,
            ModelNames.Qwen3ASR.decoderFullFile,
        ]
        return requiredModels.allSatisfy { model in
            let path = directory.appendingPathComponent(model)
            return fm.fileExists(atPath: path.path)
        }
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
