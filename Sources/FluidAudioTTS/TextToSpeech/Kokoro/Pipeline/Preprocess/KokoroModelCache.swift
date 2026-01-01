@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

public actor KokoroModelCache {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroModelCache")
    private var kokoroModels: [ModelNames.TTS.Variant: MLModel] = [:]
    private var tokenLengthCache: [ModelNames.TTS.Variant: Int] = [:]
    private var downloadedModels: [ModelNames.TTS.Variant: MLModel] = [:]
    private var referenceDimension: Int?

    public init() {}

    public func loadModelsIfNeeded(variants: Set<ModelNames.TTS.Variant>? = nil) async throws {
        let targetVariants: Set<ModelNames.TTS.Variant> = {
            if let variants = variants, !variants.isEmpty {
                return variants
            }
            return Set(ModelNames.TTS.Variant.allCases)
        }()

        let missingVariants = targetVariants.filter { kokoroModels[$0] == nil }
        if missingVariants.isEmpty { return }

        let variantsNeedingDownload = missingVariants.filter { downloadedModels[$0] == nil }

        if !variantsNeedingDownload.isEmpty {
            let newlyDownloaded = try await TtsModels.download(variants: Set(variantsNeedingDownload))
            for (variant, model) in newlyDownloaded.modelsByVariant {
                downloadedModels[variant] = model
            }
        }

        for variant in missingVariants {
            guard let model = downloadedModels[variant] else {
                throw TTSError.modelNotFound(ModelNames.TTS.bundle(for: variant))
            }
            kokoroModels[variant] = model
            tokenLengthCache[variant] = KokoroSynthesizer.inferTokenLength(from: model)
            logger.info("Loaded Kokoro \(variantDescription(variant)) model from cache")
        }

        if referenceDimension == nil,
            let referenceModel = kokoroModels[ModelNames.TTS.defaultVariant] ?? kokoroModels.values.first
        {
            referenceDimension = KokoroSynthesizer.refDim(from: referenceModel)
        }

        let loadedVariants = kokoroModels.keys.map { variantDescription($0) }.sorted().joined(separator: ", ")
        logger.info("Kokoro models ready: [\(loadedVariants)]")
    }

    public func model(for variant: ModelNames.TTS.Variant) async throws -> MLModel {
        if let existing = kokoroModels[variant] {
            return existing
        }
        try await loadModelsIfNeeded(variants: Set([variant]))
        guard let model = kokoroModels[variant] else {
            throw TTSError.modelNotFound(ModelNames.TTS.bundle(for: variant))
        }
        return model
    }

    public func tokenLength(for variant: ModelNames.TTS.Variant) async throws -> Int {
        if let cached = tokenLengthCache[variant] {
            return cached
        }
        let model = try await model(for: variant)
        let length = KokoroSynthesizer.inferTokenLength(from: model)
        tokenLengthCache[variant] = length
        return length
    }

    public func referenceEmbeddingDimension() async throws -> Int {
        if let cached = referenceDimension { return cached }

        if let defaultModel = kokoroModels[ModelNames.TTS.defaultVariant] {
            let dim = KokoroSynthesizer.refDim(from: defaultModel)
            referenceDimension = dim
            return dim
        }

        if let existing = kokoroModels.values.first {
            let dim = KokoroSynthesizer.refDim(from: existing)
            referenceDimension = dim
            return dim
        }

        try await loadModelsIfNeeded(variants: Set([ModelNames.TTS.defaultVariant]))
        if let model = kokoroModels[ModelNames.TTS.defaultVariant] ?? kokoroModels.values.first {
            let dim = KokoroSynthesizer.refDim(from: model)
            referenceDimension = dim
            return dim
        }

        throw TTSError.modelNotFound("Kokoro reference embedding model not available")
    }

    public func registerPreloadedModels(_ models: TtsModels) {
        for (variant, model) in models.modelsByVariant {
            downloadedModels[variant] = model
            kokoroModels[variant] = model
            tokenLengthCache[variant] = KokoroSynthesizer.inferTokenLength(from: model)
        }

        if referenceDimension == nil {
            let defaultModel =
                models.model(for: ModelNames.TTS.defaultVariant)
                ?? kokoroModels[ModelNames.TTS.defaultVariant]
                ?? kokoroModels.values.first
            if let defaultModel {
                referenceDimension = KokoroSynthesizer.refDim(from: defaultModel)
            }
        }
    }

    private func variantDescription(_ variant: ModelNames.TTS.Variant) -> String {
        switch variant {
        case .fiveSecond:
            return "5s"
        case .fifteenSecond:
            return "15s"
        }
    }
}
