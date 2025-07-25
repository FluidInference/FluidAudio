import CoreML
import Foundation
import OSLog

public enum CoreMLDiarizer {
    public typealias SegmentationModel = MLModel
    public typealias EmbeddingModel = MLModel
}

@available(macOS 13.0, iOS 16.0, *)
public struct DiarizerModels {

    public let segmentationModel: CoreMLDiarizer.SegmentationModel
    public let embeddingModel: CoreMLDiarizer.EmbeddingModel
    public let downloadDuration: TimeInterval
    public let compilationDuration: TimeInterval

    init(segmentation: MLModel, embedding: MLModel, downloadDuration: TimeInterval = 0, compilationDuration: TimeInterval = 0) {
        self.segmentationModel = segmentation
        self.embeddingModel = embedding
        self.downloadDuration = downloadDuration
        self.compilationDuration = compilationDuration
    }
}

// -----------------------------
// MARK: - Download from Hugging Face.
// -----------------------------

extension DiarizerModels {

    private static let SegmentationModelFileName = "pyannote_segmentation"
    private static let EmbeddingModelFileName = "wespeaker"

    public static func download(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizerModels {
        let logger = Logger(subsystem: "FluidAudio", category: "DiarizerModels")
        logger.info("Checking for diarizer models...")

        let startTime = Date()
        let directory = directory ?? defaultModelsDirectory()
        let config = configuration ?? defaultConfiguration()

        let modelNames = [
            SegmentationModelFileName + ".mlmodelc",
            EmbeddingModelFileName + ".mlmodelc"
        ]

        let models = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: modelNames,
            directory: directory.deletingLastPathComponent(),
            computeUnits: config.computeUnits
        )

        guard let segmentationModel = models[SegmentationModelFileName + ".mlmodelc"],
              let embeddingModel = models[EmbeddingModelFileName + ".mlmodelc"] else {
            throw DiarizerError.modelDownloadFailed
        }

        let endTime = Date()
        let totalDuration = endTime.timeIntervalSince(startTime)
        // For now, we don't have separate download vs compilation times, so we'll estimate
        // In reality, if models are cached, download time is 0
        let downloadDuration: TimeInterval = 0 // Models are typically cached
        let compilationDuration = totalDuration // Most time is spent on compilation
        
        return DiarizerModels(segmentation: segmentationModel, embedding: embeddingModel, downloadDuration: downloadDuration, compilationDuration: compilationDuration)
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

    static func defaultModelsDirectory() -> URL {
        let applicationSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return applicationSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent(DownloadUtils.Repo.diarizer.folderName, isDirectory: true)
    }

    static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
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

        let logger = Logger(subsystem: "FluidAudio", category: "DiarizerModels")
        logger.info("Loading predownloaded models")

        let configuration = configuration ?? defaultConfiguration()

        let startTime = Date()
        let segmentationModel = try MLModel(contentsOf: localSegmentationModel, configuration: configuration)
        let embeddingModel = try MLModel(contentsOf: localEmbeddingModel, configuration: configuration)

        let endTime = Date()
        let loadDuration = endTime.timeIntervalSince(startTime)
        return DiarizerModels(segmentation: segmentationModel, embedding: embeddingModel, downloadDuration: 0, compilationDuration: loadDuration)
    }
}
