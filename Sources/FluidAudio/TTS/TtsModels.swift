import Accelerate
@preconcurrency import CoreML
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
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> TtsModels {
        let targetDir = try directory ?? getCacheDirectory()
        // Pass Models subdirectory so models end up in ~/.cache/fluidaudio/Models/kokoro/
        let modelsDirectory = targetDir.appendingPathComponent(TtsConstants.defaultModelsSubdirectory)
        let targetVariants: [ModelNames.TTS.Variant] = {
            if let requested = requestedVariants, !requested.isEmpty {
                return requested.sorted { $0.fileName < $1.fileName }
            }
            return ModelNames.TTS.Variant.allCases
        }()
        let modelNames = targetVariants.map { $0.fileName }
        // Pass single variant name so only the requested model is downloaded
        let variantFilter: String? = targetVariants.count == 1 ? targetVariants[0].fileName : nil
        let dict = try await DownloadUtils.loadModels(
            .kokoro,
            modelNames: modelNames,
            directory: modelsDirectory,
            // v2 models converted with fp16 precision schedule BERT + generator ops to ANE (1.67x speedup)
            computeUnits: .all,
            variant: variantFilter,
            progressHandler: progressHandler
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

        for (variant, model) in loaded {
            let warmUpStart = Date()
            await warmUpModel(model, variant: variant)
            warmUpDurations[variant] = Date().timeIntervalSince(warmUpStart)
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

            var inputDict: [String: Any] = [
                "input_ids": inputIds,
                "attention_mask": attentionMask,
                "ref_s": refStyle,
                "random_phases": randomPhases,
            ]

            // Source noise only required for v2 models (macOS)
            // v1 models (iOS) don't have this input
            if model.modelDescription.inputDescriptionsByName["source_noise"] != nil {
                let maxSeconds = variant.maxDurationSeconds
                let noiseLength = TtsConstants.audioSampleRate * maxSeconds
                let sourceNoise = try MLMultiArray(
                    shape: [1, NSNumber(value: noiseLength), 9],
                    dataType: .float16
                )
                // Generate random Float32 values and convert to Float16 using vImage
                // This avoids direct Float16 usage which isn't available in all build configurations
                let totalElements = noiseLength * 9
                let floatBuffer = [Float](unsafeUninitializedCapacity: totalElements) { buffer, initializedCount in
                    for i in 0..<totalElements {
                        buffer[i] = Float.random(in: -1...1)
                    }
                    initializedCount = totalElements
                }

                let noisePointer = sourceNoise.dataPointer.bindMemory(to: UInt16.self, capacity: totalElements)

                // Convert Float32 to Float16 (UInt16) using vImage
                floatBuffer.withUnsafeBytes { floatBytes in
                    var sourceBuffer = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: floatBytes.baseAddress!),
                        height: 1,
                        width: vImagePixelCount(totalElements),
                        rowBytes: totalElements * MemoryLayout<Float>.stride
                    )

                    var destBuffer = vImage_Buffer(
                        data: noisePointer,
                        height: 1,
                        width: vImagePixelCount(totalElements),
                        rowBytes: totalElements * MemoryLayout<UInt16>.stride
                    )

                    vImageConvert_PlanarFtoPlanar16F(&sourceBuffer, &destBuffer, 0)
                }
                inputDict["source_noise"] = sourceNoise
            }

            let features = try MLDictionaryFeatureProvider(dictionary: inputDict)

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
