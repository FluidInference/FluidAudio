#if arch(arm64)
@preconcurrency import CoreML
import Foundation

#if canImport(CoreAI)
import CoreAI
#endif

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
        /// `"<stage index>:<function>"` -> stage model (loaded, or deferred to first use), plus the head.
        case coreML(stages: [String: LFM3AudioStageModel], head: MLModel)
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
            let coreAIVariant: String? = wantsSpeech ? nil : "recognition"
            _ = try await ModelHub.loadWithRecovery(
                .lfm3AudioCoreAI, directory: root,
                requiredFiles: names.coreAIRequiredModels(variant: coreAIVariant), variant: coreAIVariant,
                progressHandler: progressHandler
            ) { () }
        }
        let coreAIDirectory = root.appendingPathComponent(Repo.lfm3AudioCoreAI.folderName, isDirectory: true)
        return try await ModelHub.loadWithRecovery(
            .lfm3Audio, directory: root, requiredFiles: files, variant: variant, progressHandler: progressHandler
        ) {
            do {
                return try load(
                    from: coreMLDirectory, backend: backend, coreAIDirectory: coreAIDirectory,
                    includeSpeech: wantsSpeech, computeUnits: computeUnits)
            } catch {
                // On a low disk the system can purge Core ML's compile cache mid-load, which surfaces as bogus load
                // errors ("functionName must be nil unless ML Program"). Keep the download instead of letting
                // loadWithRecovery purge and re-fetch the multi-GB pack.
                guard let free = LFM3AudioConfig.freeDiskBytes(), free < LFM3AudioConfig.lowDiskBytes else {
                    throw error
                }
                throw LFM3AudioError.lowDiskSpace(freeBytes: free, underlying: error.localizedDescription)
            }
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
        warnIfLowDisk(computeUnits)
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
        var coreAIDepth: URL?
        switch backend {
        case .coreML:
            var stages: [String: LFM3AudioStageModel] = [:]
            let units = computeUnits.languageModel
            for plan in config.callPlans {
                let deferred = LFM3AudioConfig.deferredPrefillRows.contains(plan.q)
                for (index, function) in plan.calls.enumerated() where stages["\(index):\(function)"] == nil {
                    guard index < config.stages.count else { throw LFM3AudioError.invalidModel("stage \(index)") }
                    let name = stem(config.stages[index].package)
                    if deferred {
                        stages["\(index):\(function)"] = .deferred {
                            logger.info("Loading backbone \(name) \(function) on first use")
                            return try loadModel(name, from: directory, function: function, units: units)
                        }
                    } else {
                        stages["\(index):\(function)"] = .loaded(
                            try loadModel(name, from: directory, function: function, units: units))
                    }
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
            let depth = base.appendingPathComponent(names.coreAIBundleFile(names.coreAIDepthDecoder))
            if includeSpeech && speechAvailable {
                guard FileManager.default.fileExists(atPath: depth.path) else {
                    throw LFM3AudioError.modelNotFound(names.coreAIBundleFile(names.coreAIDepthDecoder))
                }
                coreAIDepth = depth
            }
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
            includeSpeech
            ? try LFM3AudioSpeechModels.load(from: directory, coreAIDepth: coreAIDepth, computeUnits: computeUnits)
            : nil
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

    /// Delete the Core AI specializations of the LFM3-Audio bundles. The first Core AI load specializes the LM for
    /// the GPU and caches it (~3.3 GB); the system also purges these entries under storage pressure or when the
    /// bundles change. The next Core AI load specializes again.
    public static func clearCoreAICache(coreAIDirectory: URL? = nil) throws {
        #if canImport(CoreAI)
        if #available(macOS 27, iOS 27, tvOS 27, visionOS 27, *) {
            let names = ModelNames.LFM3Audio.self
            let base =
                coreAIDirectory
                ?? modelsRootDirectory().appendingPathComponent(Repo.lfm3AudioCoreAI.folderName, isDirectory: true)
            for name in [names.coreAILanguageModel, names.coreAIDepthDecoder] {
                let bundle = base.appendingPathComponent(names.coreAIBundleFile(name))
                guard FileManager.default.fileExists(atPath: bundle.path) else { continue }
                try AIModelCache.default.deleteEntries(for: bundle)
            }
        }
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

    /// Cold loads on a low disk can fail or abort on the first prediction (`LFM3AudioConfig.lowDiskBytes`).
    static func warnIfLowDisk(_ computeUnits: LFM3AudioComputeUnits) {
        let gpu = [computeUnits.languageModel, computeUnits.offlineEncoder, computeUnits.depthDecoder].contains {
            $0 == .cpuAndGPU || $0 == .all
        }
        guard gpu, let free = LFM3AudioConfig.freeDiskBytes(), free < LFM3AudioConfig.lowDiskBytes else { return }
        logger.warning(
            "Only \(free / 1_000_000_000) GB free disk. A first LFM3-Audio load compiles several GB of GPU kernels "
                + "into the Core ML cache, and below \(LFM3AudioConfig.lowDiskBytes / 1_000_000_000) GB free the "
                + "system can purge them mid-load (load errors, or an MPSGraphTensorData abort on the first "
                + "prediction). Free up disk space if that happens.")
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
