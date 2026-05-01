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
    private var condStepChunkModelStorage: MLModel?
    private var flowlmStepModel: MLModel?
    private var flowDecoderModel: MLModel?
    private var mimiDecoderModel: MLModel?
    private var mimiEncoderModel: MLModel?
    private var constantsBundle: PocketTtsConstantsBundle?
    private var voiceCache: [String: PocketTtsVoiceData] = [:]
    private var languageRootDirectory: URL?
    private var condLayerKeys: PocketTtsLayerKeys?
    private var condStepChunkLayerKeysCache: PocketTtsLayerKeys?
    private var flowlmLayerKeys: PocketTtsLayerKeys?
    private var mimiDecoderKeysCache: PocketTtsMimiKeys?
    private let directory: URL?
    public let language: PocketTtsLanguage
    public let precision: PocketTtsPrecision
    public let condStepMode: PocketTtsCondStepMode

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
    ///   - condStepMode: Which `cond_step` dispatch strategy the synthesizer
    ///     should use for KV cache prefill. Defaults to `.legacy` (per-token
    ///     dispatch — preserves the upstream behaviour). `.chunked(chunk:
    ///     16)` additionally loads `cond_step_chunk16.mlmodelc` from the
    ///     same `v2/<lang>/` directory and lets the synthesizer dispatch
    ///     prompt prefill in 16-token chunks plus a per-token tail. The
    ///     chunk-16 file is **not yet published on HuggingFace** — see
    ///     `PocketTtsCondStepMode.chunked` for placement details.
    public init(
        language: PocketTtsLanguage = .english,
        directory: URL? = nil,
        precision: PocketTtsPrecision = .fp16,
        condStepMode: PocketTtsCondStepMode = .legacy
    ) {
        self.language = language
        self.directory = directory
        self.precision = precision
        self.condStepMode = condStepMode
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

        // Use CPU+GPU for all models to avoid ANE float16 precision loss.
        // The ANE processes in native float16, which causes audible artifacts
        // in the Mimi decoder's streaming state feedback loop and may degrade
        // quality in the other models. CPU/GPU compute in float32 matches the
        // Python reference implementation.
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let loadStart = Date()

        let modelFiles: [String] = [
            ModelNames.PocketTTS.condStepFile,
            ModelNames.PocketTTS.flowlmStepFile(precision: precision),
            ModelNames.PocketTTS.flowDecoderFile,
            ModelNames.PocketTTS.mimiDecoderFile,
        ]

        var loadedModels: [MLModel] = []
        for file in modelFiles {
            let modelURL = languageRoot.appendingPathComponent(file)
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            loadedModels.append(model)
            logger.info("Loaded \(file)")
        }

        condStepModel = loadedModels[0]
        flowlmStepModel = loadedModels[1]
        flowDecoderModel = loadedModels[2]
        mimiDecoderModel = loadedModels[3]

        // Discover per-model output names. Names differ between 6L and 24L
        // packs because CoreML auto-generates them during tracing.
        let expectedLayers = language.transformerLayers
        condLayerKeys = try PocketTtsLayerKeys.discover(
            from: loadedModels[0],
            kind: .condStep,
            expectedLayers: expectedLayers,
            modelName: "cond_step"
        )
        flowlmLayerKeys = try PocketTtsLayerKeys.discover(
            from: loadedModels[1],
            kind: .flowlmStep,
            expectedLayers: expectedLayers,
            modelName: "flowlm_step"
        )

        // Discover Mimi decoder schema (per-state input→output mapping +
        // audio output name). CoreML auto-generates `var_NNN` output names
        // during conversion so the exact names vary across packs.
        mimiDecoderKeysCache = try PocketTtsMimiKeys.discover(from: loadedModels[3])

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All PocketTTS models loaded in \(String(format: "%.2f", elapsed))s")

        // Optionally load the chunked cond_step variant. The chunk-N model
        // shares the K/V cache + position output schema of chunk-1, so the
        // existing `.condStep` discovery kind applies — only the input
        // sequence dim differs (1 → N).
        if case .chunked(let chunk) = condStepMode {
            try await loadCondStepChunkModel(
                chunk: chunk,
                languageRoot: languageRoot,
                config: config,
                expectedLayers: expectedLayers
            )
        }

        // Load constants
        constantsBundle = try PocketTtsConstantsLoader.load(from: languageRoot)
        logger.info("PocketTTS constants loaded")
    }

    private func loadCondStepChunkModel(
        chunk: Int,
        languageRoot: URL,
        config: MLModelConfiguration,
        expectedLayers: Int
    ) async throws {
        // Only chunk-16 is supported initially. Reject other sizes loudly so
        // callers don't silently fall back to a missing artifact.
        guard chunk == 16 else {
            throw PocketTTSError.modelNotFound(
                "PocketTTS chunked cond_step only supports chunk=16 today (requested \(chunk))"
            )
        }

        let file = ModelNames.PocketTTS.condStepChunk16File
        let modelURL = languageRoot.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw PocketTTSError.modelNotFound(
                "PocketTTS \(file) not found at \(modelURL.path). "
                    + "The chunked cond_step variant is not yet published on HuggingFace; "
                    + "place the compiled mlmodelc at this path manually to enable .chunked(chunk: 16)."
            )
        }

        let chunkLoadStart = Date()
        let model = try MLModel(contentsOf: modelURL, configuration: config)
        condStepChunkModelStorage = model
        condStepChunkLayerKeysCache = try PocketTtsLayerKeys.discover(
            from: model,
            kind: .condStep,
            expectedLayers: expectedLayers,
            modelName: ModelNames.PocketTTS.condStepChunk16
        )
        let chunkElapsed = Date().timeIntervalSince(chunkLoadStart)
        logger.info(
            "Loaded \(file) (chunk=\(chunk)) in \(String(format: "%.2f", chunkElapsed))s"
        )
    }

    /// The conditioning step model (KV cache prefill).
    public func condStep() throws -> MLModel {
        guard let model = condStepModel else {
            throw PocketTTSError.modelNotFound("PocketTTS cond_step model not loaded")
        }
        return model
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

    /// The chunked cond_step model. Throws when the store was initialized
    /// in `.legacy` mode (or when the chunk model file failed to load) —
    /// callers should gate on `condStepChunkSize() != nil` first.
    ///
    /// Returns a non-optional `MLModel` to match the Sendable behaviour of
    /// `condStep()`; `Optional<MLModel>` would require `MLModel` itself to
    /// satisfy strict-mode Sendable (the `@preconcurrency import CoreML`
    /// trick only covers the bare `MLModel`).
    public func condStepChunkModel() throws -> MLModel {
        guard let model = condStepChunkModelStorage else {
            throw PocketTTSError.modelNotFound(
                "PocketTTS chunked cond_step model not loaded (mode = \(condStepMode))"
            )
        }
        return model
    }

    /// Discovered output names for the chunked cond_step model. Same
    /// throwing semantics as `condStepChunkModel()`.
    func condStepChunkLayerKeys() throws -> PocketTtsLayerKeys {
        guard let keys = condStepChunkLayerKeysCache else {
            throw PocketTTSError.modelNotFound(
                "PocketTTS chunked cond_step layer keys not discovered (mode = \(condStepMode))"
            )
        }
        return keys
    }

    /// The chunk size if the store was initialized in `.chunked` mode,
    /// otherwise `nil`. Use this as the cheap gate before calling the
    /// throwing chunk-model accessors.
    public func condStepChunkSize() -> Int? {
        if case .chunked(let chunk) = condStepMode {
            return chunk
        }
        return nil
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
