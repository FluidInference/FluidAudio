import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
actor KokoroModelCache {
    static let shared = KokoroModelCache()

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "KokoroModelCache")
    private var kokoroModels: [ModelNames.TTS.Variant: MLModel] = [:]
    private var tokenLengthCache: [ModelNames.TTS.Variant: Int] = [:]
    private var downloadedModelBundle: TtsModels?
    private var referenceDimension: Int?

    func loadModelsIfNeeded() async throws {
        if kokoroModels.count == ModelNames.TTS.Variant.allCases.count {
            return
        }

        if downloadedModelBundle == nil {
            downloadedModelBundle = try await TtsModels.download()
        }

        guard let bundle = downloadedModelBundle else {
            throw TTSError.modelNotFound("Kokoro models unavailable after download attempt")
        }

        for variant in ModelNames.TTS.Variant.allCases where kokoroModels[variant] == nil {
            guard let model = bundle.model(for: variant) else {
                throw TTSError.modelNotFound(ModelNames.TTS.bundle(for: variant))
            }
            kokoroModels[variant] = model
            tokenLengthCache[variant] = KokoroSynthesizer.inferTokenLength(from: model)
            logger.info("Loaded Kokoro \(variantDescription(variant)) model from cache")
        }

        let loadedVariants = kokoroModels.keys.map { variantDescription($0) }.sorted().joined(separator: ", ")
        logger.info("Kokoro models ready: [\(loadedVariants)]")
    }

    func model(for variant: ModelNames.TTS.Variant) async throws -> MLModel {
        if let existing = kokoroModels[variant] {
            return existing
        }
        try await loadModelsIfNeeded()
        guard let model = kokoroModels[variant] else {
            throw TTSError.modelNotFound(ModelNames.TTS.bundle(for: variant))
        }
        return model
    }

    func tokenLength(for variant: ModelNames.TTS.Variant) async throws -> Int {
        if let cached = tokenLengthCache[variant] {
            return cached
        }
        let model = try await model(for: variant)
        let length = KokoroSynthesizer.inferTokenLength(from: model)
        tokenLengthCache[variant] = length
        return length
    }

    func referenceEmbeddingDimension() async throws -> Int {
        if let cached = referenceDimension { return cached }
        let model = try await model(for: ModelNames.TTS.defaultVariant)
        let dim = KokoroSynthesizer.refDim(from: model)
        referenceDimension = dim
        return dim
    }

    func registerPreloadedModels(_ models: TtsModels) {
        downloadedModelBundle = models

        for variant in ModelNames.TTS.Variant.allCases {
            if let model = models.model(for: variant) {
                kokoroModels[variant] = model
                tokenLengthCache[variant] = KokoroSynthesizer.inferTokenLength(from: model)
            }
        }
        if referenceDimension == nil,
            let defaultModel = models.model(for: ModelNames.TTS.defaultVariant)
        {
            referenceDimension = KokoroSynthesizer.refDim(from: defaultModel)
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
