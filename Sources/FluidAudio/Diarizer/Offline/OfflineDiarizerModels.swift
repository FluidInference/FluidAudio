@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct OfflineDiarizerModels: Sendable {
    public let segmentationModel: MLModel
    public let embeddingModel: MLModel
    public let pldaRhoModel: MLModel
    public let pldaPsi: [Double]

    public let downloadDuration: TimeInterval
    public let compilationDuration: TimeInterval

    private static let logger = AppLogger(category: "OfflineDiarizerModels")

    private static func loadPLDAPsi(from directory: URL) throws -> [Double] {
        let candidatePaths = [
            directory.appendingPathComponent("plda-parameters.json", isDirectory: false),
            directory.appendingPathComponent("speaker-diarization-coreml/plda-parameters.json", isDirectory: false),
            directory.appendingPathComponent("speaker-diarization-offline/plda-parameters.json", isDirectory: false),
        ]
        guard let parametersURL = candidatePaths.first(where: { FileManager.default.fileExists(atPath: $0.path) })
        else {
            throw OfflineDiarizationError.processingFailed("PLDA parameters file not found in \(directory.path)")
        }

        let data = try Data(contentsOf: parametersURL)
        let jsonObject = try JSONSerialization.jsonObject(with: data, options: [])
        guard
            let root = jsonObject as? [String: Any],
            let tensors = root["tensors"] as? [String: Any],
            let psiInfo = tensors["psi"] as? [String: Any],
            let base64 = psiInfo["data_base64"] as? String,
            let decoded = Data(base64Encoded: base64, options: [.ignoreUnknownCharacters])
        else {
            throw OfflineDiarizationError.processingFailed("Failed to decode PLDA psi parameters")
        }

        let floatCount = decoded.count / MemoryLayout<Float>.size
        guard floatCount > 0 else {
            throw OfflineDiarizationError.processingFailed("PLDA psi tensor is empty")
        }

        var floats = [Float](repeating: 0, count: floatCount)
        _ = floats.withUnsafeMutableBytes { destination in
            decoded.copyBytes(to: destination)
        }

        return floats.map { Double($0) }
    }

    public init(
        segmentationModel: MLModel,
        embeddingModel: MLModel,
        pldaRhoModel: MLModel,
        pldaPsi: [Double],
        downloadDuration: TimeInterval,
        compilationDuration: TimeInterval
    ) {
        self.segmentationModel = segmentationModel
        self.embeddingModel = embeddingModel
        self.pldaRhoModel = pldaRhoModel
        self.pldaPsi = pldaPsi
        self.downloadDuration = downloadDuration
        self.compilationDuration = compilationDuration
    }

    public static func defaultModelsDirectory() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return
            base
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    private static func defaultConfiguration() -> MLModelConfiguration {
        let configuration = MLModelConfiguration()
        configuration.allowLowPrecisionAccumulationOnGPU = true
        configuration.computeUnits = .all
        return configuration
    }

    public static func load(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> OfflineDiarizerModels {
        let modelsDirectory = directory ?? defaultModelsDirectory()
        let logger = Self.logger
        logger.info("Loading offline diarization models from \(modelsDirectory.path)")

        let loadStart = Date()
        let config = configuration ?? defaultConfiguration()

        let requiredNames = Array(ModelNames.OfflineDiarizer.requiredModels)
        let loadedModels = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: requiredNames,
            directory: modelsDirectory,
            computeUnits: config.computeUnits,
            variant: "offline"
        )

        guard let segmentation = loadedModels[ModelNames.OfflineDiarizer.segmentationPath] else {
            throw OfflineDiarizationError.modelNotLoaded(ModelNames.OfflineDiarizer.segmentation)
        }
        guard let embedding = loadedModels[ModelNames.OfflineDiarizer.embeddingPath] else {
            throw OfflineDiarizationError.modelNotLoaded(ModelNames.OfflineDiarizer.embedding)
        }
        guard let plda = loadedModels[ModelNames.OfflineDiarizer.pldaRhoPath] else {
            throw OfflineDiarizationError.modelNotLoaded(ModelNames.OfflineDiarizer.pldaRho)
        }

        let pldaPsi = try loadPLDAPsi(from: modelsDirectory)
        let compilationDuration = Date().timeIntervalSince(loadStart)
        logger.info(
            "Offline diarization models ready (compile: \(String(format: "%.3f", compilationDuration))s)"
        )

        return OfflineDiarizerModels(
            segmentationModel: segmentation,
            embeddingModel: embedding,
            pldaRhoModel: plda,
            pldaPsi: pldaPsi,
            downloadDuration: 0,
            compilationDuration: compilationDuration
        )
    }
}
