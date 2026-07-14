@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 14.0, iOS 17.0, *)
public struct OfflineDiarizerModels: Sendable {
    public let segmentationModel: MLModel
    /// WeSpeaker embedder stack — `nil` when the models are loaded for the
    /// TitaNet-10s backend. NME-SC needs only segmentation; the 192-d TitaNet
    /// embedder and its clustering never touch FBANK/WeSpeaker/PLDA, so those
    /// files are neither downloaded nor compiled on the TitaNet path.
    public let fbankModel: MLModel?
    public let embeddingModel: MLModel?
    public let pldaRhoModel: MLModel?
    public let pldaPsi: [Double]

    public let compilationDuration: TimeInterval

    private static let logger = AppLogger(category: "OfflineDiarizerModels")

    private static func loadPLDAPsi(from directory: URL) throws -> [Double] {
        let candidatePaths = [
            directory.appendingPathComponent("plda-parameters.json", isDirectory: false),
            directory.appendingPathComponent("speaker-diarization/plda-parameters.json", isDirectory: false),
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
        fbankModel: MLModel? = nil,
        embeddingModel: MLModel? = nil,
        pldaRhoModel: MLModel? = nil,
        pldaPsi: [Double] = [],
        compilationDuration: TimeInterval
    ) {
        self.segmentationModel = segmentationModel
        self.fbankModel = fbankModel
        self.embeddingModel = embeddingModel
        self.pldaRhoModel = pldaRhoModel
        self.pldaPsi = pldaPsi
        self.compilationDuration = compilationDuration
    }

    public static func defaultModelsDirectory() -> URL {
        MLModelConfigurationUtils.defaultModelsDirectory()
    }

    private static func defaultConfiguration() -> MLModelConfiguration {
        MLModelConfigurationUtils.defaultConfiguration(computeUnits: .all)
    }

    public static func load(
        from directory: URL? = nil,
        configuration _: MLModelConfiguration? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil,
        backend: OfflineDiarizerConfig.Embedding.Backend = .wespeaker
    ) async throws -> OfflineDiarizerModels {
        let modelsDirectory = directory ?? defaultModelsDirectory()
        let logger = Self.logger
        logger.info(
            "Loading offline diarization models from \(modelsDirectory.path) (backend: \(backend.rawValue))"
        )

        let loadStart = Date()
        let inferenceComputeUnits: MLComputeUnits = .cpuAndNeuralEngine

        // Segmentation is always required — it drives VAD + the per-speaker
        // masks consumed by BOTH the WeSpeaker and TitaNet embedders. The variant
        // controls the DOWNLOAD gate: `.titanet10s` uses "offline-titanet" so
        // `downloadRepo` fetches ONLY Segmentation (not the WeSpeaker/FBANK/PLDA
        // files), while `.wespeaker` keeps "offline" (full 5-file set).
        let segmentationVariant = backend == .titanet10s ? "offline-titanet" : "offline"
        let segmentationModels = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: [ModelNames.OfflineDiarizer.segmentationPath],
            directory: modelsDirectory,
            computeUnits: inferenceComputeUnits,
            variant: segmentationVariant,
            progressHandler: progressHandler
        )
        guard let segmentation = segmentationModels[ModelNames.OfflineDiarizer.segmentationPath] else {
            throw OfflineDiarizationError.modelNotLoaded(ModelNames.OfflineDiarizer.segmentation)
        }

        // TitaNet-10s (NME-SC) uses only segmentation; the WeSpeaker embedder,
        // FBANK frontend, and PLDA/VBx artifacts are never touched at inference,
        // so skip downloading + compiling them entirely.
        if backend == .titanet10s {
            let compilationDuration = Date().timeIntervalSince(loadStart)
            logger.info(
                "Offline diarization models ready (TitaNet: segmentation only, compile: \(String(format: "%.3f", compilationDuration))s)"
            )
            return OfflineDiarizerModels(
                segmentationModel: segmentation,
                compilationDuration: compilationDuration
            )
        }

        // WeSpeaker path — full stack: embedding + PLDA (rho) + FBANK + psi.
        let embeddingAndPldaNames: [String] = [
            ModelNames.OfflineDiarizer.embeddingPath,
            ModelNames.OfflineDiarizer.pldaRhoPath,
        ]
        let embeddingAndPldaModels = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: embeddingAndPldaNames,
            directory: modelsDirectory,
            computeUnits: inferenceComputeUnits,
            variant: "offline",
            progressHandler: progressHandler
        )
        guard let embedding = embeddingAndPldaModels[ModelNames.OfflineDiarizer.embeddingPath] else {
            throw OfflineDiarizationError.modelNotLoaded(ModelNames.OfflineDiarizer.embedding)
        }
        guard let plda = embeddingAndPldaModels[ModelNames.OfflineDiarizer.pldaRhoPath] else {
            throw OfflineDiarizationError.modelNotLoaded(ModelNames.OfflineDiarizer.pldaRho)
        }

        let fbankComputeUnits: MLComputeUnits = .cpuOnly
        let fbankModels = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: [ModelNames.OfflineDiarizer.fbankPath],
            directory: modelsDirectory,
            computeUnits: fbankComputeUnits,
            variant: "offline",
            progressHandler: progressHandler
        )
        guard let fbank = fbankModels[ModelNames.OfflineDiarizer.fbankPath] else {
            throw OfflineDiarizationError.modelNotLoaded(ModelNames.OfflineDiarizer.fbank)
        }

        let pldaPsi = try loadPLDAPsi(from: modelsDirectory)
        let compilationDuration = Date().timeIntervalSince(loadStart)
        let compileString = String(format: "%.3f", compilationDuration)
        logger.info(
            "Offline diarization models ready (compile: \(compileString)s, computeUnits: segmentation/embedding/plda=\(inferenceComputeUnits.label), fbank=\(fbankComputeUnits.label))"
        )

        return OfflineDiarizerModels(
            segmentationModel: segmentation,
            fbankModel: fbank,
            embeddingModel: embedding,
            pldaRhoModel: plda,
            pldaPsi: pldaPsi,
            compilationDuration: compilationDuration
        )
    }
}

private extension MLComputeUnits {
    var label: String {
        switch self {
        case .cpuOnly:
            return ".cpuOnly"
        case .cpuAndGPU:
            return ".cpuAndGPU"
        case .cpuAndNeuralEngine:
            return ".cpuAndNeuralEngine"
        case .all:
            return ".all"
        @unknown default:
            return ".unknown"
        }
    }
}
