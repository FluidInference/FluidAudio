@preconcurrency import CoreML
import Foundation
import OSLog

public enum CoreMLDiarizer {
    public typealias SegmentationModel = MLModel
    public typealias EmbeddingModel = MLModel
}

@available(macOS 13.0, iOS 16.0, *)
public struct DiarizerModels: @unchecked Sendable {

    public let segmentationModel: CoreMLDiarizer.SegmentationModel
    public let embeddingModel: CoreMLDiarizer.EmbeddingModel
    public let downloadTime: Date
    public let compilationTime: Date

    init(segmentation: MLModel, embedding: MLModel, downloadTime: Date = Date(), compilationTime: Date = Date()) {
        self.segmentationModel = segmentation
        self.embeddingModel = embedding
        self.downloadTime = downloadTime
        self.compilationTime = compilationTime
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
        logger.info("Starting model download")

        let directory = directory ?? defaultModelsDirectory()
        let config = configuration ?? defaultConfiguration()

        // Use new DownloadUtils system
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

        let downloadEndTime = Date()
        return DiarizerModels(segmentation: segmentationModel, embedding: embeddingModel, downloadTime: Date(), compilationTime: downloadEndTime)
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

extension Date {
    var timeInterval: TimeInterval {
        self.timeIntervalSinceReferenceDate
    }
}

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

        let segmentationModel = try MLModel(contentsOf: localSegmentationModel, configuration: configuration)
        let embeddingModel = try MLModel(contentsOf: localEmbeddingModel, configuration: configuration)

        let downloadEndTime = Date()
        return DiarizerModels(segmentation: segmentationModel, embedding: embeddingModel, downloadTime: Date(), compilationTime: downloadEndTime)
    }
}
