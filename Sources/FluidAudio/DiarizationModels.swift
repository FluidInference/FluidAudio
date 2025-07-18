import CoreML
import OSLog

/// Models required for diarization.
///
internal struct DiarizationModels {

    private static var SegmentationModelFileName: String { "pyannote_segmentation.mlmodelc" }
    private static var EmbeddingModelFileName: String { "wespeaker.mlmodelc" }

    // ML models
    internal var segmentationModel: MLModel
    internal var embeddingModel: MLModel
    internal var paths: ModelPaths

    // Timing tracking
    internal var downloadTime: Duration
    internal var compilationTime: Duration

    private init(
        segmentationModel: MLModel,
        embeddingModel: MLModel,
        paths: ModelPaths,
        downloadTime: Duration,
        compilationTime: Duration
    ) {
        self.segmentationModel = segmentationModel
        self.embeddingModel = embeddingModel
        self.paths = paths
        self.downloadTime = downloadTime
        self.compilationTime = compilationTime
    }
}

// ---------------------------
// MARK: - Downloading models.
// ---------------------------

extension DiarizationModels {

    private static var HuggingFaceRepoPath: String { "bweng/speaker-diarization-coreml" }

    /// Download the models, if needed, to a directory managed by the framework.
    ///
    public static func download(
        logger: Logger
    ) async throws -> DiarizationModels {

        let directory: URL
        #if os(iOS)
            // Use Documents directory on iOS for better compatibility with sandboxing
            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            directory = documents.appendingPathComponent("FluidAudio/models/diarization", isDirectory: true)
        #else
            // Use Application Support on macOS
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            directory = appSupport.appendingPathComponent("SpeakerKitModels/coreml", isDirectory: true)
        #endif

        return try await download(to: directory, logger: logger)
    }

    /// Download the models, if needed, to the given directory.
    ///
    public static func download(
        to modelsDirectory: URL,
        logger: Logger
    ) async throws -> DiarizationModels {

        try deleteNoncompiledModels(modelsDirectory: modelsDirectory, logger)

        try FileManager.default.createDirectory(
            at: modelsDirectory,
            withIntermediateDirectories: true
        )

        var modelPaths: ModelPaths!
        let downloadTime = try await ContinuousClock().measure {
            modelPaths = try await downloadModelsIfNeeded(to: modelsDirectory, logger)
        }

        var models: (segmentation: MLModel, embedding: MLModel)!
        let compilationTime = try await ContinuousClock().measure {
            models = try await loadModelsWithAutoRecovery(
                segmentationURL: modelPaths.segmentationPath,
                embeddingURL: modelPaths.embeddingPath
            )
        }

        return DiarizationModels(
            segmentationModel: models.segmentation,
            embeddingModel: models.embedding,
            paths: modelPaths,
            downloadTime: downloadTime,
            compilationTime: compilationTime
        )
    }

    /// Deletes the existing segmentation and embedding models in the given directory
    /// if they are not compiled.
    ///
    private static func deleteNoncompiledModels(
        modelsDirectory: URL,
        _ logger: Logger
    ) throws {

        let segmentationModelPath = modelsDirectory
            .appendingPathComponent(SegmentationModelFileName, isDirectory: true)
        let embeddingModelPath = modelsDirectory
            .appendingPathComponent(EmbeddingModelFileName, isDirectory: true)

        if FileManager.default.fileExists(atPath: segmentationModelPath.path)
            && !DownloadUtils.isModelCompiled(at: segmentationModelPath)
        {
            logger.info("Removing broken segmentation model")
            try FileManager.default.removeItem(at: segmentationModelPath)
        }

        if FileManager.default.fileExists(atPath: embeddingModelPath.path)
            && !DownloadUtils.isModelCompiled(at: embeddingModelPath)
        {
            logger.info("Removing broken embedding model")
            try FileManager.default.removeItem(at: embeddingModelPath)
        }
    }

    /// Download required models for diarization to the given directory.
    ///
    /// If compiled models already exist, they will not be redownloaded.
    ///
    private static func downloadModelsIfNeeded(
        to modelsDirectory: URL,
        _ logger: Logger
    ) async throws -> ModelPaths {

        logger.info("Checking for existing diarization models")

        let segmentationModelPath = modelsDirectory
            .appendingPathComponent(SegmentationModelFileName, isDirectory: true)
        let embeddingModelPath = modelsDirectory
            .appendingPathComponent(EmbeddingModelFileName, isDirectory: true)

        // Check if models already exist and are compiled.

        let segmentationExists =
            FileManager.default.fileExists(atPath: segmentationModelPath.path)
            && DownloadUtils.isModelCompiled(at: segmentationModelPath)
        let embeddingExists =
            FileManager.default.fileExists(atPath: embeddingModelPath.path)
            && DownloadUtils.isModelCompiled(at: embeddingModelPath)

        if segmentationExists && embeddingExists {
            logger.info("Valid models already exist, skipping download")
            return ModelPaths(
                segmentationPath: segmentationModelPath,
                embeddingPath: embeddingModelPath
            )
        }

        logger.info("Downloading missing or invalid diarization models from Hugging Face")

        // Download models if needed.

        try await withThrowingTaskGroup(of: Void.self) { group in

            if !segmentationExists {
                group.addTask {
                    logger.info("Downloading segmentation model bundle from Hugging Face")
                    try await DownloadUtils.downloadMLModelBundle(
                        repoPath: HuggingFaceRepoPath,
                        modelName: SegmentationModelFileName,
                        outputPath: segmentationModelPath
                    )
                    logger.info("Downloaded segmentation model bundle from Hugging Face")
                }
            }

            if !embeddingExists {
                group.addTask {
                    logger.info("Downloading embedding model bundle from Hugging Face")
                    try await DownloadUtils.downloadMLModelBundle(
                        repoPath: HuggingFaceRepoPath,
                        modelName: EmbeddingModelFileName,
                        outputPath: embeddingModelPath
                    )
                    logger.info("Downloaded embedding model bundle from Hugging Face")
                }
            }

            try await group.waitForAll()
        }

        logger.info("Successfully ensured diarization models are available")

        return ModelPaths(
            segmentationPath: segmentationModelPath,
            embeddingPath: embeddingModelPath
        )
    }

    /// Loads the models at the given local locations.
    ///
    /// If the models fail to load, they will be deleted and redownloaded, up to the given
    /// number of retry attempts.
    ///
    private static func loadModelsWithAutoRecovery(
        segmentationURL: URL,
        embeddingURL: URL,
        maxRetries: Int = 2
    ) async throws -> (segmentation: MLModel, embedding: MLModel) {

        assert(segmentationURL.isFileURL, "This should be a file URL to a downloaded model")
        assert(embeddingURL.isFileURL, "This should be a file URL to a downloaded model")

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        async let segmentationModel = DownloadUtils.withAutoRecovery {
            try MLModel(contentsOf: segmentationURL, configuration: config)
        } recovery: {
            try await DownloadUtils.performModelRecovery(
                modelPaths: [segmentationURL],
                downloadAction: {
                    try await DownloadUtils.downloadMLModelBundle(
                        repoPath: HuggingFaceRepoPath,
                        modelName: SegmentationModelFileName,
                        outputPath: segmentationURL
                    )
                }
            )
        }

        async let embeddingModel = DownloadUtils.withAutoRecovery {
            try MLModel(contentsOf: embeddingURL, configuration: config)
        } recovery: {
            try await DownloadUtils.performModelRecovery(
                modelPaths: [embeddingURL],
                downloadAction: {
                    try await DownloadUtils.downloadMLModelBundle(
                        repoPath: HuggingFaceRepoPath,
                        modelName: EmbeddingModelFileName,
                        outputPath: embeddingURL
                    )
                }
            )
        }

        return (segmentation: try await segmentationModel, embedding: try await embeddingModel)
    }
}

// -----------------------------
// MARK: - Predownloaded models.
// -----------------------------

extension DiarizationModels {

    /// Loads the models from the given local files.
    ///
    /// If the models fail to load, no recovery will be attempted. No models are downloaded.
    ///
    public static func load(
        localSegmentationModel: URL,
        localEmbeddingModel: URL
    ) throws -> DiarizationModels {

        guard localEmbeddingModel.isFileURL, localEmbeddingModel.isFileURL else {
            throw LoadingErrors.notAFileURL
        }

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine

        var segmentationModel: MLModel!
        var embeddingModel: MLModel!
        let compilationTime = try ContinuousClock().measure {
            segmentationModel = try MLModel(contentsOf: localSegmentationModel, configuration: config)
            embeddingModel    = try MLModel(contentsOf: localEmbeddingModel, configuration: config)
        }

        return DiarizationModels(
            segmentationModel: segmentationModel,
            embeddingModel: embeddingModel,
            paths: ModelPaths(segmentationPath: localSegmentationModel, embeddingPath: localEmbeddingModel),
            downloadTime: .zero,
            compilationTime: compilationTime
        )
    }

    private enum LoadingErrors: Error, CustomStringConvertible {
        case notAFileURL

        var description: String {
            switch self {
            case .notAFileURL: "The given URL is not a file URL"
            }
        }
    }
}
