@preconcurrency import CoreML
import Foundation

/// Actor-based store for PocketTTS CoreML models and constants.
///
/// Manages loading and storing of the four CoreML models
/// (cond_step, flowlm_step, flow_decoder, mimi_decoder),
/// the binary constants bundle, and voice conditioning data.
///
/// A store is bound to a single `PocketTtsLanguage` for its lifetime; switch
/// languages by creating a new store/manager.
public actor PocketTtsModelStore {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "PocketTtsModelStore")

    private var condStepModel: MLModel?
    private var condPrefillModel: MLModel?
    private var flowlmStepModel: MLModel?
    private var flowDecoderModel: MLModel?
    private var mimiDecoderModel: MLModel?
    private var mimiEncoderModel: MLModel?
    private var constantsBundle: PocketTtsConstantsBundle?
    private var voiceCache: [String: PocketTtsVoiceData] = [:]
    private var languageRootDirectory: URL?
    private var condLayerKeys: PocketTtsLayerKeys?
    private var condPrefillLayerKeys: PocketTtsLayerKeys?
    private var flowlmLayerKeys: PocketTtsLayerKeys?
    private var mimiDecoderKeysCache: PocketTtsMimiKeys?
    private let directory: URL?
    public let language: PocketTtsLanguage
    public let precision: PocketTtsPrecision

    /// - Parameters:
    ///   - language: Which upstream language pack to load. Defaults to
    ///     `.english`.
    ///   - directory: Optional override for the base cache directory. When
    ///     `nil`, uses the default platform cache location.
    ///   - precision: Which FlowLM precision to load (default: `.fp16`,
    ///     matching upstream's on-disk weight format). `.int8` swaps
    ///     `flowlm_step.mlmodelc` for `flowlm_stepv2.mlmodelc` from the
    ///     same upstream `v2/<lang>/` directory; the other three submodels
    ///     stay at fp16.
    public init(
        language: PocketTtsLanguage = .english,
        directory: URL? = nil,
        precision: PocketTtsPrecision = .fp16
    ) {
        self.language = language
        self.directory = directory
        self.precision = precision
    }

    /// Load all four CoreML models and the constants bundle.
    public func loadIfNeeded() async throws {
        guard condStepModel == nil else { return }

        let languageRoot = try await PocketTtsResourceDownloader.ensureModels(
            language: language,
            directory: directory,
            precision: precision
        )
        self.languageRootDirectory = languageRoot

        logger.info(
            "Loading PocketTTS CoreML models (language=\(self.language.rawValue), precision=\(self.precision))..."
        )

        // Per-model compute units. The global `.cpuAndGPU` hammer was set to
        // stop the Mimi decoder beeping (its streaming-state fp16 feedback loop
        // compounds ANE float16 error into audible artifacts — see
        // mobius IOS_COREML_ISSUES.md #7). But that ban only needs to apply to
        // Mimi; pinning every model off the ANE also throws away the documented
        // wins. Configs below are the MEASURED fastest per model (M-series /
        // macOS 26, coreml-cli medians across all 4 compute-unit configs):
        //   cond / cond_prefill : .all     prefill 4.7ms @ all vs 7.5 @ cpuAndGPU
        //                                   (ANE compile fails on rank-5 → GPU,
        //                                   but `.all` GPU placement is faster)
        //   flowlm_step         : .all     fp16 3.4ms @ all vs 5.0 @ cpuAndGPU
        //                                   (GPU — NOT ANE; rank-5 scatter blocks ANE)
        //   flow_decoder_fused  : .all     1.09ms, the ONE model that is 100% ANE
        //   mimi_decoder        : .cpuOnly 6.0ms, faster than GPU + avoids ANE beep
        // NOTE: the earlier "flowlm 1.97× on ANE" claim was disproven on-device —
        // only the fused decoder reaches the ANE; the rest are GPU/CPU.
        func config(_ units: MLComputeUnits) -> MLModelConfiguration {
            let c = MLModelConfiguration()
            c.computeUnits = units
            return c
        }
        let condConfig = config(.all)
        let flowlmConfig = config(.all)
        let flowDecoderConfig = config(.all)
        let mimiConfig = config(.cpuOnly)

        let loadStart = Date()

        // v2.1 required set: cond_prefill (one-shot conditioner) + fused flow
        // decoder replace v2's cond_step + per-step flow_decoder.
        let modelSpecs: [(file: String, config: MLModelConfiguration)] = [
            (ModelNames.PocketTTS.condPrefillFile, condConfig),
            (ModelNames.PocketTTS.flowlmStepFile(precision: precision), flowlmConfig),
            (ModelNames.PocketTTS.flowDecoderFusedFile, flowDecoderConfig),
            (ModelNames.PocketTTS.mimiDecoderFile, mimiConfig),
        ]

        var loadedModels: [MLModel] = []
        for spec in modelSpecs {
            let modelURL = languageRoot.appendingPathComponent(spec.file)
            let model = try MLModel(contentsOf: modelURL, configuration: spec.config)
            loadedModels.append(model)
            logger.info("Loaded \(spec.file) (computeUnits=\(spec.config.computeUnits.rawValue))")
        }

        // In v2.1 the conditioner IS cond_prefill (no per-token cond_step).
        // Assign it to both condStepModel (legacy accessor) and condPrefillModel
        // so the prefill fast-path runs; the per-token fallback never fires
        // (useCondPrefill=true, text chunks <= T_max).
        condStepModel = loadedModels[0]
        condPrefillModel = loadedModels[0]
        flowlmStepModel = loadedModels[1]
        flowDecoderModel = loadedModels[2]  // flow_decoder_fused
        mimiDecoderModel = loadedModels[3]

        // Discover per-model output names. Names differ between 6L and 24L
        // packs because CoreML auto-generates them during tracing.
        let expectedLayers = language.transformerLayers
        condLayerKeys = try PocketTtsLayerKeys.discover(
            from: loadedModels[0],
            kind: .condStep,  // cond_prefill shares cond_step's output schema
            expectedLayers: expectedLayers,
            modelName: "cond_prefill"
        )
        flowlmLayerKeys = try PocketTtsLayerKeys.discover(
            from: loadedModels[1],
            kind: .flowlmStep,
            expectedLayers: expectedLayers,
            modelName: "flowlm_step"
        )

        // cond_prefill is the required v2.1 conditioner (loaded above as
        // loadedModels[0]); its layer keys match cond_step's schema.
        condPrefillLayerKeys = condLayerKeys

        // Discover Mimi decoder schema (per-state input→output mapping +
        // audio output name). CoreML auto-generates `var_NNN` output names
        // during conversion so the exact names vary across packs.
        mimiDecoderKeysCache = try PocketTtsMimiKeys.discover(from: loadedModels[3])

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All PocketTTS models loaded in \(String(format: "%.2f", elapsed))s")

        // Load constants
        constantsBundle = try PocketTtsConstantsLoader.load(from: languageRoot)
        logger.info("PocketTTS constants loaded")
    }

    /// The conditioning step model (KV cache prefill).
    public func condStep() throws -> MLModel {
        guard let model = condStepModel else {
            throw PocketTTSError.modelNotFound("PocketTTS cond_step model not loaded")
        }
        return model
    }

    /// The one-shot conditioning prefill model. Throws when the pack doesn't
    /// ship `cond_prefill`; gate with `hasCondPrefill()` first (callers fall
    /// back to per-token cond_step). Returned non-optional because the
    /// (preconcurrency-Sendable) `MLModel` crosses the actor boundary while
    /// `Optional<MLModel>` does not.
    public func condPrefill() throws -> MLModel {
        guard let model = condPrefillModel else {
            throw PocketTTSError.modelNotFound("PocketTTS cond_prefill model not loaded")
        }
        return model
    }

    /// Whether the optional one-shot prefill model is available.
    public func hasCondPrefill() -> Bool {
        condPrefillModel != nil
    }

    /// Discovered output names for the cond_prefill model (same schema as
    /// cond_step). `nil` when cond_prefill isn't loaded. `PocketTtsLayerKeys`
    /// is Sendable, so `Optional<PocketTtsLayerKeys>` crosses the boundary fine.
    func condPrefillStepLayerKeys() -> PocketTtsLayerKeys? {
        condPrefillLayerKeys
    }

    /// The autoregressive generation step model.
    public func flowlmStep() throws -> MLModel {
        guard let model = flowlmStepModel else {
            throw PocketTTSError.modelNotFound("PocketTTS flowlm_step model not loaded")
        }
        return model
    }

    /// The LSD flow decoder model.
    public func flowDecoder() throws -> MLModel {
        guard let model = flowDecoderModel else {
            throw PocketTTSError.modelNotFound("PocketTTS flow_decoder model not loaded")
        }
        return model
    }

    /// The Mimi streaming audio decoder model.
    public func mimiDecoder() throws -> MLModel {
        guard let model = mimiDecoderModel else {
            throw PocketTTSError.modelNotFound("PocketTTS mimi_decoder model not loaded")
        }
        return model
    }

    /// The pre-loaded binary constants.
    public func constants() throws -> PocketTtsConstantsBundle {
        guard let bundle = constantsBundle else {
            throw PocketTTSError.modelNotFound("PocketTTS constants not loaded")
        }
        return bundle
    }

    /// Discovered output names for the cond_step transformer model.
    func condStepLayerKeys() throws -> PocketTtsLayerKeys {
        guard let keys = condLayerKeys else {
            throw PocketTTSError.modelNotFound("PocketTTS cond_step layer keys not discovered")
        }
        return keys
    }

    /// Discovered output names for the flowlm_step transformer model.
    func flowLMStepLayerKeys() throws -> PocketTtsLayerKeys {
        guard let keys = flowlmLayerKeys else {
            throw PocketTTSError.modelNotFound("PocketTTS flowlm_step layer keys not discovered")
        }
        return keys
    }

    /// Discovered I/O schema for the Mimi audio decoder model (state mapping,
    /// audio output name, declared state shapes).
    func mimiDecoderKeys() throws -> PocketTtsMimiKeys {
        guard let keys = mimiDecoderKeysCache else {
            throw PocketTTSError.modelNotFound("PocketTTS mimi_decoder keys not discovered")
        }
        return keys
    }

    /// The language root directory (`<repoDir>/v2/<lang>`) — contains the
    /// four model files, `constants_bin/`, and is the right base for
    /// `loadMimiInitialState`.
    public func repoDir() throws -> URL {
        guard let dir = languageRootDirectory else {
            throw PocketTTSError.modelNotFound("PocketTTS repository not loaded")
        }
        return dir
    }

    /// Load and cache voice conditioning data, downloading from HuggingFace if missing.
    public func voiceData(for voice: String) async throws -> PocketTtsVoiceData {
        if let cached = voiceCache[voice] {
            return cached
        }
        guard let languageRoot = languageRootDirectory else {
            throw PocketTTSError.modelNotFound("PocketTTS repository not loaded")
        }
        let data = try await PocketTtsResourceDownloader.ensureVoice(
            voice,
            language: language,
            languageRoot: languageRoot
        )
        voiceCache[voice] = data
        return data
    }

    // MARK: - Voice Cloning

    /// Load the Mimi encoder model for voice cloning (lazy, on-demand).
    ///
    /// Downloads the model from HuggingFace if not already cached. The Mimi
    /// encoder is language-agnostic and lives at the repo root, shared
    /// across all language packs.
    public func loadMimiEncoderIfNeeded() async throws {
        guard mimiEncoderModel == nil else { return }

        // Ensure the mimi_encoder is downloaded (downloads if needed)
        let modelURL = try await PocketTtsResourceDownloader.ensureMimiEncoder(directory: directory)

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        logger.info("Loading Mimi encoder for voice cloning...")
        let loadStart = Date()
        mimiEncoderModel = try MLModel(contentsOf: modelURL, configuration: config)
        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("Mimi encoder loaded in \(String(format: "%.2f", elapsed))s")
    }

    /// The Mimi encoder model for voice cloning.
    public func mimiEncoder() throws -> MLModel {
        guard let model = mimiEncoderModel else {
            throw PocketTTSError.modelNotFound(
                "Mimi encoder not loaded. Call loadMimiEncoderIfNeeded() first."
            )
        }
        return model
    }

    /// Check if the Mimi encoder model is available.
    public func isMimiEncoderAvailable() -> Bool {
        // The Mimi encoder lives at the repo root, two levels above any
        // `v2/<lang>/` language root.
        guard let langRoot = languageRootDirectory else { return false }
        let repoRoot = langRoot.deletingLastPathComponent().deletingLastPathComponent()
        let modelURL = repoRoot.appendingPathComponent(ModelNames.PocketTTS.mimiEncoderFile)
        return FileManager.default.fileExists(atPath: modelURL.path)
    }

    /// Clone a voice from an audio URL within the actor's isolation context.
    public func cloneVoice(from audioURL: URL) throws -> PocketTtsVoiceData {
        let encoder = try mimiEncoder()
        return try PocketTtsVoiceCloner.cloneVoice(from: audioURL, using: encoder)
    }

    /// Clone a voice from audio samples within the actor's isolation context.
    public func cloneVoice(from samples: [Float]) throws -> PocketTtsVoiceData {
        let encoder = try mimiEncoder()
        return try PocketTtsVoiceCloner.cloneVoice(from: samples, using: encoder)
    }
}
