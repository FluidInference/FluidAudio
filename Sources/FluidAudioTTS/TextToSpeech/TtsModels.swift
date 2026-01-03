@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

public struct TtsModels: Sendable {
    private let kokoroModels: [ModelNames.TTS.Variant: MLModel]

    private static let logger = AppLogger(category: "TtsModels")

    public init(models: [ModelNames.TTS.Variant: MLModel]) {
        self.kokoroModels = models
    }

    internal var modelsByVariant: [ModelNames.TTS.Variant: MLModel] {
        kokoroModels
    }

    public var availableVariants: Set<ModelNames.TTS.Variant> {
        Set(kokoroModels.keys)
    }

    public func model(for variant: ModelNames.TTS.Variant = ModelNames.TTS.defaultVariant) -> MLModel? {
        kokoroModels[variant]
    }

    public static func download(
        variants requestedVariants: Set<ModelNames.TTS.Variant>? = nil,
        from repo: String = TtsConstants.defaultRepository,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> TtsModels {
        let cacheDirectory = try getCacheDirectory()
        // Pass Models subdirectory so models end up in ~/.cache/fluidaudio/Models/kokoro/
        let modelsDirectory = cacheDirectory.appendingPathComponent(TtsConstants.defaultModelsSubdirectory)
        let targetVariants: [ModelNames.TTS.Variant] = {
            if let requested = requestedVariants, !requested.isEmpty {
                return requested.sorted { $0.fileName < $1.fileName }
            }
            return ModelNames.TTS.Variant.allCases
        }()
        let modelNames = targetVariants.map { $0.fileName }
        let dict = try await DownloadUtils.loadModels(
            .kokoro,
            modelNames: modelNames,
            directory: modelsDirectory,
            // Only a small fraction of the model can run on ANE, and compile time takes a long time because of the complicated arch
            computeUnits: .cpuAndGPU
        )
        var loaded: [ModelNames.TTS.Variant: MLModel] = [:]
        var warmUpDurations: [ModelNames.TTS.Variant: TimeInterval] = [:]

        for variant in targetVariants {
            let name = variant.fileName
            guard let model = dict[name] else {
                throw TTSError.modelNotFound(name)
            }
            loaded[variant] = model
        }

        try await withThrowingTaskGroup(of: (ModelNames.TTS.Variant, TimeInterval).self) { group in
            for (variant, model) in loaded {
                group.addTask(priority: .userInitiated) {
                    let warmUpStart = Date()
                    await warmUpModel(model, variant: variant)
                    let warmUpDuration = Date().timeIntervalSince(warmUpStart)
                    return (variant, warmUpDuration)
                }
            }

            for try await (variant, duration) in group {
                warmUpDurations[variant] = duration
            }
        }

        for variant in targetVariants {
            if let duration = warmUpDurations[variant] {
                let formatted = String(format: "%.2f", duration)
                logger.info("Warm-up completed for \(variantDescription(variant)) in \(formatted)s")
            }
        }

        return TtsModels(models: loaded)
    }

    private static func getCacheDirectory() throws -> URL {
        let baseDirectory: URL
        #if os(macOS)
        baseDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard let first = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first else {
            throw TTSError.processingFailed("Failed to locate caches directory")
        }
        baseDirectory = first
        #endif

        let cacheDirectory = baseDirectory.appendingPathComponent("fluidaudio")

        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory,
                withIntermediateDirectories: true
            )
        }

        return cacheDirectory
    }

    public static func cacheDirectoryURL() throws -> URL {
        return try getCacheDirectory()
    }

    public static func optimizedPredictionOptions() -> MLPredictionOptions {
        let options = MLPredictionOptions()
        // Enable batching for better GPU utilization
        options.outputBackings = [:]  // Reuse output buffers
        return options
    }

    // Run a lightweight pseudo generation to prime Core ML caches for subsequent real syntheses.
    private static func warmUpModel(_ model: MLModel, variant: ModelNames.TTS.Variant) async {
        do {
            let tokenLength = max(1, KokoroSynthesizer.inferTokenLength(from: model))

            let inputIds = try MLMultiArray(
                shape: [1, NSNumber(value: tokenLength)] as [NSNumber],
                dataType: .int32
            )
            let attentionMask = try MLMultiArray(
                shape: [1, NSNumber(value: tokenLength)] as [NSNumber],
                dataType: .int32
            )

            // Fill the complete token window for this variant (5s vs 15s models expose different lengths).
            for index in 0..<tokenLength {
                inputIds[index] = NSNumber(value: 0)
                attentionMask[index] = NSNumber(value: 1)
            }

            let refDim = max(1, KokoroSynthesizer.refDim(from: model))
            let refStyle = try MLMultiArray(
                shape: [1, NSNumber(value: refDim)] as [NSNumber],
                dataType: .float32
            )
            for index in 0..<refDim {
                refStyle[index] = NSNumber(value: Float(0))
            }

            let phasesShape =
                model.modelDescription.inputDescriptionsByName["random_phases"]?.multiArrayConstraint?.shape
                ?? [NSNumber(value: 1), NSNumber(value: 9)]
            let randomPhases = try MLMultiArray(
                shape: phasesShape,
                dataType: .float32
            )
            for index in 0..<randomPhases.count {
                randomPhases[index] = NSNumber(value: Float(0))
            }

            let features = try MLDictionaryFeatureProvider(dictionary: [
                "input_ids": inputIds,
                "attention_mask": attentionMask,
                "ref_s": refStyle,
                "random_phases": randomPhases,
            ])

            let options: MLPredictionOptions = optimizedPredictionOptions()
            _ = try await model.compatPrediction(from: features, options: options)
        } catch {
            logger.warning(
                "Warm-up prediction failed for variant \(variantDescription(variant)): \(error.localizedDescription)"
            )
        }
    }

    private static func variantDescription(_ variant: ModelNames.TTS.Variant) -> String {
        switch variant {
        case .fiveSecond:
            return "5s"
        case .fifteenSecond:
            return "15s"
        }
    }
}

public enum TTSError: LocalizedError {
    case downloadFailed(String)
    case corruptedModel(String)
    case modelNotFound(String)
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .corruptedModel(let name):
            return "Model \(name) is corrupted"
        case .modelNotFound(let name):
            return "Model \(name) not found"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        }
    }
}
