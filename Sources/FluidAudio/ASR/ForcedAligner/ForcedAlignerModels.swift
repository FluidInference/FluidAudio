@preconcurrency import CoreML
import Foundation
import OSLog

private let logger = Logger(subsystem: "FluidAudio", category: "ForcedAlignerModels")

/// Holds the 3 CoreML model components for the Qwen3-ForcedAligner pipeline.
///
/// Components:
/// - `audioEncoder`: mel spectrogram -> 1024-dim audio features (f32)
/// - `embedding`: input_ids -> 1024-dim text embeddings (int8)
/// - `decoderWithLmHead`: merged embeddings + MRoPE -> logits (int8, fused decoder+norm+lm_head)
@available(macOS 14, iOS 17, *)
public struct ForcedAlignerModels: Sendable {
    public let audioEncoder: MLModel
    public let embedding: MLModel
    public let decoderWithLmHead: MLModel

    /// Load ForcedAligner models from a local directory.
    ///
    /// Expected directory structure:
    /// ```
    /// forced-aligner-int8/
    ///   forced_aligner_audio_encoder.mlmodelc
    ///   forced_aligner_embedding.mlmodelc
    ///   forced_aligner_decoder_with_lm_head.mlmodelc
    /// ```
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .all
    ) async throws -> ForcedAlignerModels {
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits

        logger.info("Loading ForcedAligner models from \(directory.path)")
        let start = CFAbsoluteTimeGetCurrent()

        let audioEncoder = try await loadModel(
            named: "forced_aligner_audio_encoder",
            from: directory,
            configuration: modelConfig
        )

        let embedding = try await loadModel(
            named: "forced_aligner_embedding",
            from: directory,
            configuration: modelConfig
        )

        let decoderWithLmHead = try await loadModel(
            named: "forced_aligner_decoder_with_lm_head",
            from: directory,
            configuration: modelConfig
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.info("Loaded ForcedAligner models in \(String(format: "%.2f", elapsed))s")

        return ForcedAlignerModels(
            audioEncoder: audioEncoder,
            embedding: embedding,
            decoderWithLmHead: decoderWithLmHead
        )
    }

    /// Download models from HuggingFace and load them.
    public static func downloadAndLoad(
        to directory: URL? = nil,
        computeUnits: MLComputeUnits = .all
    ) async throws -> ForcedAlignerModels {
        let targetDir = try await download(to: directory)
        return try await load(from: targetDir, computeUnits: computeUnits)
    }

    /// Download ForcedAligner models from HuggingFace.
    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        let modelsRoot = modelsRootDirectory()

        if !force && modelsExist(at: targetDir) {
            logger.info("ForcedAligner models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            try? FileManager.default.removeItem(at: targetDir)
        }

        logger.info("Downloading ForcedAligner int8 models from HuggingFace...")
        try await DownloadUtils.downloadRepo(.forcedAlignerInt8, to: modelsRoot)
        logger.info("Successfully downloaded ForcedAligner models")
        return targetDir
    }

    /// Check if all required model files exist locally.
    public static func modelsExist(at directory: URL) -> Bool {
        let fm = FileManager.default
        let requiredFiles = [
            ModelNames.ForcedAligner.audioEncoderFile,
            ModelNames.ForcedAligner.embeddingFile,
            ModelNames.ForcedAligner.decoderWithLmHeadFile,
        ]
        return requiredFiles.allSatisfy { file in
            fm.fileExists(atPath: directory.appendingPathComponent(file).path)
        }
    }

    /// Default cache directory for ForcedAligner models.
    public static func defaultCacheDirectory() -> URL {
        modelsRootDirectory()
            .appendingPathComponent(Repo.forcedAlignerInt8.folderName, isDirectory: true)
    }

    // MARK: Private

    private static func modelsRootDirectory() -> URL {
        guard
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory, in: .userDomainMask
            ).first
        else {
            return FileManager.default.temporaryDirectory
                .appendingPathComponent("FluidAudio", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        return
            appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    private static func loadModel(
        named name: String,
        from directory: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        let compiledPath = directory.appendingPathComponent("\(name).mlmodelc")
        let packagePath = directory.appendingPathComponent("\(name).mlpackage")

        let modelURL: URL
        if FileManager.default.fileExists(atPath: compiledPath.path) {
            modelURL = compiledPath
        } else if FileManager.default.fileExists(atPath: packagePath.path) {
            logger.info("Compiling \(name).mlpackage -> .mlmodelc ...")
            let compileStart = CFAbsoluteTimeGetCurrent()
            let compiledURL = try await MLModel.compileModel(at: packagePath)
            let compileElapsed = CFAbsoluteTimeGetCurrent() - compileStart
            logger.info("  \(name): compiled in \(String(format: "%.2f", compileElapsed))s")

            try? FileManager.default.removeItem(at: compiledPath)
            try FileManager.default.copyItem(at: compiledURL, to: compiledPath)
            try? FileManager.default.removeItem(at: compiledURL)

            modelURL = compiledPath
        } else {
            throw ForcedAlignerError.modelNotFound(name)
        }

        let start = CFAbsoluteTimeGetCurrent()
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        logger.debug("  \(name): loaded in \(String(format: "%.2f", elapsed))s")
        return model
    }
}
