@preconcurrency import CoreML
import Foundation
import OSLog

public enum CoreMLDiarizer {
    public typealias SegmentationModel = MLModel
    public typealias EmbeddingModel = MLModel
}

public struct DiarizerModels: Sendable {

    /// Required model names for Diarizer
    public static let requiredModelNames = ModelNames.Diarizer.requiredModels

    public let segmentationModel: CoreMLDiarizer.SegmentationModel
    public let embeddingModel: CoreMLDiarizer.EmbeddingModel
    public let compilationDuration: TimeInterval

    init(
        segmentation: MLModel, embedding: MLModel,
        compilationDuration: TimeInterval = 0
    ) {
        self.segmentationModel = segmentation
        self.embeddingModel = embedding
        self.compilationDuration = compilationDuration
    }
}

// -----------------------------
// MARK: - Download from Hugging Face.
// -----------------------------

extension DiarizerModels {

    private static let SegmentationModelFileName = ModelNames.Diarizer.segmentation
    private static let EmbeddingModelFileName = ModelNames.Diarizer.embedding

    // MARK: - Private Model Loading Helpers

    public static func download(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {
        let logger = AppLogger(category: "DiarizerModels")
        logger.info("Checking for diarizer models...")

        let startTime = Date()
        let directory = directory ?? defaultModelsDirectory()
        let config = configuration ?? defaultConfiguration()

        // Download required models
        let segmentationModelName = ModelNames.Diarizer.segmentationFile
        let embeddingModelName = ModelNames.Diarizer.embeddingFile

        let models = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: Array(requiredModelNames),
            directory: directory.deletingLastPathComponent(),
            computeUnits: config.computeUnits
        )

        // Load segmentation model
        guard let segmentationModel = models[segmentationModelName] else {
            throw DiarizerError.modelDownloadFailed
        }

        // Load embedding model
        guard let embeddingModel = models[embeddingModelName] else {
            throw DiarizerError.modelDownloadFailed
        }

        let endTime = Date()
        let totalDuration = endTime.timeIntervalSince(startTime)

        return DiarizerModels(
            segmentation: segmentationModel,
            embedding: embeddingModel,
            compilationDuration: totalDuration)
    }

    public static func load(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {
        let directory = directory ?? defaultModelsDirectory()
        return try await download(to: directory, configuration: configuration)
    }

    public static func downloadIfNeeded(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {
        return try await download(to: directory, configuration: configuration)
    }

    public static func defaultModelsDirectory() -> URL {
        let applicationSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return
            applicationSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(Repo.diarizer.folderName, isDirectory: true)
    }

    static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        // Enable Float16 optimization for ~2x speedup
        config.allowLowPrecisionAccumulationOnGPU = true
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        config.computeUnits = isCI ? .cpuAndNeuralEngine : .all
        return config
    }
}

// -----------------------------
// MARK: - Predownloaded models.
// -----------------------------

extension DiarizerModels {

    /// Load the models from the given local files.
    ///
    /// If the models fail to load, no recovery will be attempted. No models are downloaded.
    ///
    public static func load(
        localSegmentationModel: URL,
        localEmbeddingModel: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {

        let logger = AppLogger(category: "DiarizerModels")
        logger.info("Loading predownloaded models")

        let configuration = configuration ?? defaultConfiguration()

        let startTime = Date()
        let segmentationModel = try MLModel(contentsOf: localSegmentationModel, configuration: configuration)
        let embeddingModel = try MLModel(contentsOf: localEmbeddingModel, configuration: configuration)

        let endTime = Date()
        let loadDuration = endTime.timeIntervalSince(startTime)
        return DiarizerModels(
            segmentation: segmentationModel, embedding: embeddingModel,
            compilationDuration: loadDuration)
    }
}
