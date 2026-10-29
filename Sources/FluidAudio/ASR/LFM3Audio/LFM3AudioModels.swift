#if arch(arm64)
@preconcurrency import CoreML
import Foundation

/// Compute units per LFM3-Audio component. The defaults are the fastest measured on Apple silicon (M5 Pro):
/// the LM, offline encoder and depth decoder on the GPU, the LiveASR encoder chunk and the codec on the ANE.
public struct LFM3AudioComputeUnits: Sendable {
    public var languageModel: MLComputeUnits
    public var offlineEncoder: MLComputeUnits
    public var liveEncoder: MLComputeUnits
    public var depthDecoder: MLComputeUnits
    public var codec: MLComputeUnits

    public init(
        languageModel: MLComputeUnits = .cpuAndGPU, offlineEncoder: MLComputeUnits = .cpuAndGPU,
        liveEncoder: MLComputeUnits = .cpuAndNeuralEngine, depthDecoder: MLComputeUnits = .cpuAndGPU,
        codec: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.languageModel = languageModel
        self.offlineEncoder = offlineEncoder
        self.liveEncoder = liveEncoder
        self.depthDecoder = depthDecoder
        self.codec = codec
    }

    public static let `default` = LFM3AudioComputeUnits()
}

/// Loaded LFM3-Audio-2B models and host tables.
///
/// From `FluidInference/lfm3-audio-2b-coreml` (Core ML) and, for `.coreAI`, the LM bundle from
/// `FluidInference/lfm3-audio-2b-coreai`:
///   - backbone: LFM3 2048 x 24 (int8, block 32) as two stateful stages + a chunked top-k head
///   - `AudioEncoder-fp16`: causal LFM2 audio encoder, mel (100 Hz) -> LM rows (12.5 Hz)
///   - TTS (`includeSpeech`): MaskGIT depth decoder (12 codebooks per 80 ms frame) + LiquidCodec decoder (24 kHz)
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
public struct LFM3AudioModels: Sendable {

    enum LanguageModel: Sendable {
        /// `"<stage index>:<function>"` -> model, plus the head.
        case coreML(stages: [String: MLModel], head: MLModel)
        /// Core AI bundle, loaded by the manager (macOS 27 / iOS 27).
        case coreAI(bundle: URL)
    }

    public let backend: LFM3AudioBackend
    public let includesSpeech: Bool
    let config: LFM3AudioPackConfig
    let embedding: LFM3AudioEmbeddingTable
    let tokenizerURL: URL
    let languageModel: LanguageModel
    let encoder: [Int: MLModel]
    #if TTS
    let speech: LFM3AudioSpeechModels?
    #endif

    private static let logger = AppLogger(category: "LFM3AudioModels")

    /// Download (if needed) and load the models.
    ///
    /// - Parameters:
    ///   - backend: `.coreML` (macOS 15 / iOS 18+) or `.coreAI` (macOS 27 / iOS 27+; the audio encoder and speech
    ///     components still run on Core ML).
    ///   - includeSpeech: also load the TTS components (depth decoder, codec; ~0.8 GB more).
    public static func downloadAndLoad(
        backend: LFM3AudioBackend = .coreML,
        includeSpeech: Bool = true,
        computeUnits: LFM3AudioComputeUnits = .default,
        progressHandler: ProgressHandler? = nil
    ) async throws -> LFM3AudioModels {
        // Check before downloading: ModelHub.loadWithRecovery treats a load() error as a corrupt cache, so on an
        // unsupported OS it would delete the multi-GB download and fetch it again before failing.
        try checkPlatform(backend)
        let root = modelsRootDirectory()
        let coreMLDirectory = root.appendingPathComponent(Repo.lfm3Audio.folderName, isDirectory: true)
        let wantsSpeech = speechAvailable && includeSpeech
        let names = ModelNames.LFM3Audio.self
        let variant = downloadVariant(backend: backend, speech: wantsSpeech)
        let files = names.requiredModels(variant: variant)
        if backend == .coreAI {
            _ = try await ModelHub.loadWithRecovery(
                .lfm3AudioCoreAI, directory: root, requiredFiles: ModelNames.LFM3Audio.coreAIRequiredModels,
                progressHandler: progressHandler
            ) { () }
        }
        let coreAIDirectory = root.appendingPathComponent(Repo.lfm3AudioCoreAI.folderName, isDirectory: true)
        return try await ModelHub.loadWithRecovery(
            .lfm3Audio, directory: root, requiredFiles: files, variant: variant, progressHandler: progressHandler
        ) {
            try load(
                from: coreMLDirectory, backend: backend, coreAIDirectory: coreAIDirectory,
                includeSpeech: wantsSpeech, computeUnits: computeUnits)
        }
    }

    /// Load from local directories laid out like the HF repos (`coreml/`, `assets/`; Core AI: `coreai/`).
    public static func load(
        from directory: URL,
        backend: LFM3AudioBackend = .coreML,
        coreAIDirectory: URL? = nil,
        includeSpeech: Bool = true,
        computeUnits: LFM3AudioComputeUnits = .default
    ) throws -> LFM3AudioModels {
        try checkPlatform(backend)
        let names = ModelNames.LFM3Audio.self
        let config = try LFM3AudioPackConfig.load(directory.appendingPathComponent(names.packConfigFile))
        guard config.embedding.shape.count == 2 else { throw LFM3AudioError.invalidModel("embedding shape") }
        let embedding = try LFM3AudioEmbeddingTable(
            url: directory.appendingPathComponent(names.embeddingFile), vocab: config.embedding.shape[0],
            hidden: config.embedding.shape[1])
        let tokenizerURL = directory.appendingPathComponent(names.tokenizerFile)
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw LFM3AudioError.modelNotFound(names.tokenizerFile)
        }

        let languageModel: LanguageModel
        switch backend {
        case .coreML:
            var stages: [String: MLModel] = [:]
            for plan in config.callPlans {
                for (index, function) in plan.calls.enumerated() where stages["\(index):\(function)"] == nil {
                    guard index < config.stages.count else { throw LFM3AudioError.invalidModel("stage \(index)") }
                    stages["\(index):\(function)"] = try loadModel(
                        stem(config.stages[index].package), from: directory, function: function,
                        units: computeUnits.languageModel)
                }
            }
            let head = try loadModel(stem(config.head.package), from: directory, units: computeUnits.languageModel)
            languageModel = .coreML(stages: stages, head: head)
        case .coreAI:
            let base =
                coreAIDirectory
                ?? directory.deletingLastPathComponent()
                .appendingPathComponent(Repo.lfm3AudioCoreAI.folderName, isDirectory: true)
            let bundle = base.appendingPathComponent(names.coreAIBundleFile(names.coreAILanguageModel))
            guard FileManager.default.fileExists(atPath: bundle.path) else {
                throw LFM3AudioError.modelNotFound(names.coreAIBundleFile(names.coreAILanguageModel))
            }
            languageModel = .coreAI(bundle: bundle)
        }

        var encoder: [Int: MLModel] = [:]
        encoder[LFM3AudioConfig.liveEncoderTokens] = try loadModel(
            names.encoder, from: directory, function: "n\(LFM3AudioConfig.liveEncoderTokens)",
            units: computeUnits.liveEncoder)
        for tokens in LFM3AudioConfig.offlineEncoderTokens {
            encoder[tokens] = try loadModel(
                names.encoder, from: directory, function: "n\(tokens)", units: computeUnits.offlineEncoder)
        }

        #if TTS
        let speech =
            includeSpeech ? try LFM3AudioSpeechModels.load(from: directory, computeUnits: computeUnits) : nil
        logger.info("Loaded LFM3-Audio-2B (\(backend.rawValue), speech: \(speech != nil))")
        return LFM3AudioModels(
            backend: backend, includesSpeech: speech != nil, config: config, embedding: embedding,
            tokenizerURL: tokenizerURL, languageModel: languageModel, encoder: encoder, speech: speech)
        #else
        logger.info("Loaded LFM3-Audio-2B (\(backend.rawValue), recognition only)")
        return LFM3AudioModels(
            backend: backend, includesSpeech: false, config: config, embedding: embedding,
            tokenizerURL: tokenizerURL, languageModel: languageModel, encoder: encoder)
        #endif
    }

    // MARK: - Internal

    /// `ModelNames.LFM3Audio.requiredModels(variant:)` key: the Core AI backend never downloads the Core ML LM.
    static func downloadVariant(backend: LFM3AudioBackend, speech: Bool) -> String? {
        switch (backend, speech) {
        case (.coreML, true): return nil
        case (.coreML, false): return "recognition"
        case (.coreAI, true): return "coreai"
        case (.coreAI, false): return "coreai-recognition"
        }
    }

    static var speechAvailable: Bool {
        #if TTS
        return true
        #else
        return false
        #endif
    }

    static func checkPlatform(_ backend: LFM3AudioBackend) throws {
        switch backend {
        case .coreML:
            return
        case .coreAI:
            #if canImport(CoreAI)
            if #available(macOS 27, iOS 27, tvOS 27, visionOS 27, *) { return }
            #endif
            throw LFM3AudioError.unsupportedPlatform(
                "The LFM3-Audio Core AI backend needs macOS 27 / iOS 27 (Core AI); use .coreML")
        }
    }

    /// `coreml/<name>.mlmodelc` (or a `.mlpackage` next to it, compiled on load), one function of it if given.
    static func loadModel(
        _ name: String, from directory: URL, function: String? = nil, units: MLComputeUnits
    ) throws -> MLModel {
        let compiled = directory.appendingPathComponent(ModelNames.LFM3Audio.compiledModelFile(name))
        let package = compiled.deletingPathExtension().appendingPathExtension("mlpackage")
        let url: URL
        if FileManager.default.fileExists(atPath: compiled.path) {
            url = compiled
        } else if FileManager.default.fileExists(atPath: package.path) {
            url = try MLModel.compileModel(at: package)
        } else {
            throw LFM3AudioError.modelNotFound(ModelNames.LFM3Audio.compiledModelFile(name))
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units
        if let function { configuration.functionName = function }
        return try MLModel(contentsOf: url, configuration: configuration)
    }

    /// File stem of a pack-config package name (`LFM3-A-C4160-int8b32.mlpackage` -> `LFM3-A-C4160-int8b32`).
    private static func stem(_ package: String) -> String {
        (package as NSString).deletingPathExtension
    }

    private static func modelsRootDirectory() -> URL {
        let base =
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.temporaryDirectory
        return base.appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }
}
#endif
