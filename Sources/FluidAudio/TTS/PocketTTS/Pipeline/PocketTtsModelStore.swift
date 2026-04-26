@preconcurrency import CoreML
import Foundation
import OSLog

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
    private var flowlmStepModel: MLModel?
    private var flowDecoderModel: MLModel?
    private var mimiDecoderModel: MLModel?
    private var mimiEncoderModel: MLModel?
    private var constantsBundle: PocketTtsConstantsBundle?
    private var voiceCache: [String: PocketTtsVoiceData] = [:]
    private var languageRootDirectory: URL?
    private var condLayerKeys: PocketTtsLayerKeys?
    private var flowlmLayerKeys: PocketTtsLayerKeys?
    private var mimiSchema: PocketTtsMimiSchema?
    private let directory: URL?
    public let language: PocketTtsLanguage
    public let quantization: PocketTtsQuantization

    /// - Parameters:
    ///   - language: Which upstream language pack to load. Defaults to
    ///     `.english` for backward compatibility.
    ///   - quantization: Per-submodel precision (fp16/int8). Defaults to
    ///     all-fp16. Each submodel can be independently swapped — see
    ///     ``PocketTtsQuantization`` for presets and quality tradeoffs.
    ///   - directory: Optional override for the base cache directory. When
    ///     `nil`, uses the default platform cache location.
    public init(
        language: PocketTtsLanguage = .english,
        quantization: PocketTtsQuantization = .allFp16,
        directory: URL? = nil
    ) {
        self.language = language
        self.quantization = quantization
        self.directory = directory
    }

    /// Load all four CoreML models and the constants bundle.
    public func loadIfNeeded() async throws {
        guard condStepModel == nil else { return }

        let resolved = try await PocketTtsResourceDownloader.ensureResolvedModels(
            language: language,
            quantization: quantization,
            directory: directory
        )
        self.languageRootDirectory = resolved.languageRoot

        logger.info(
            "Loading PocketTTS CoreML models (language=\(self.language.rawValue), quant=cond:\(self.quantization.condStep.rawValue),flowlm:\(self.quantization.flowlmStep.rawValue),flow:\(self.quantization.flowDecoder.rawValue),mimi:\(self.quantization.mimiDecoder.rawValue))..."
        )

        // Per-model compute units — profiled on Apple Silicon, FP16 mlpackages:
        //
        //   model         units            why
        //   ────────────  ───────────────  ─────────────────────────────────────────
        //   cond_step     .cpuAndGPU       ANE ≈ GPU (no benefit), and ANE prefill
        //                                  occasionally hits MPSGraph rank-5/zero-
        //                                  shape assert. GPU path is robust.
        //   flowlm_step   .all             1.97× faster on ANE than GPU; this is
        //                                  the autoregressive bottleneck (called
        //                                  once per output frame).
        //   flow_decoder  .all             Tiny model called 8× per frame;
        //                                  CPU+NE/ALL are both fast.
        //   mimi_decoder  .cpuOnly         GPU dispatch overhead exceeds GPU gain
        //                                  on this small streaming-conv model
        //                                  (~1.74× faster on CPU). Cannot use ANE:
        //                                  segfaults on 64-byte stride misalign in
        //                                  some state tensors.
        //
        // FP32 mlpackages: CoreML still chooses ANE for ANE-eligible ops at
        // FP16 internally; precision loss is bounded by the ops dispatched
        // (autoregressive softmax/lm-head still run at FP32 on CPU/GPU).
        let condConfig = MLModelConfiguration()
        condConfig.computeUnits = .cpuAndGPU
        let flowlmConfig = MLModelConfiguration()
        flowlmConfig.computeUnits = .all
        let flowConfig = MLModelConfiguration()
        flowConfig.computeUnits = .all
        let mimiConfig = MLModelConfiguration()
        mimiConfig.computeUnits = .cpuOnly

        let loadStart = Date()

        // Each submodel may live at the language root (fp16) or under
        // `languages/<lang>/int8/` (int8). The resolver above gives us
        // pre-validated URLs.
        let modelLoads: [(URL, MLModelConfiguration, PocketTtsModelPrecision)] = [
            (resolved.condStepURL, condConfig, quantization.condStep),
            (resolved.flowlmStepURL, flowlmConfig, quantization.flowlmStep),
            (resolved.flowDecoderURL, flowConfig, quantization.flowDecoder),
            (resolved.mimiDecoderURL, mimiConfig, quantization.mimiDecoder),
        ]

        var loadedModels: [MLModel] = []
        for (modelURL, config, precision) in modelLoads {
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            loadedModels.append(model)
            logger.info(
                "Loaded \(modelURL.lastPathComponent) [\(precision.rawValue)] (units=\(config.computeUnits.rawValue))"
            )
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

        // Discover Mimi I/O schema (semantic v2 names preferred, falls back to
        // hardcoded v1 names for the legacy English pack).
        mimiSchema = try PocketTtsMimiSchema.discover(from: loadedModels[3])
        logger.info(
            "Mimi schema: audio=\(self.mimiSchema?.audioOutputName ?? "?"), states=\(self.mimiSchema?.stateMapping.count ?? 0)"
        )

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All PocketTTS models loaded in \(String(format: "%.2f", elapsed))s")

        // Load constants from the fp16 language root (constants_bin/ is not
        // duplicated in the int8 tree).
        constantsBundle = try PocketTtsResourceDownloader.ensureConstants(
            languageRoot: resolved.languageRoot)
        logger.info("PocketTTS constants loaded")
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

    /// Discovered output names for the flowlm_step transformer model.
    func flowLMStepLayerKeys() throws -> PocketTtsLayerKeys {
        guard let keys = flowlmLayerKeys else {
            throw PocketTTSError.modelNotFound("PocketTTS flowlm_step layer keys not discovered")
        }
        return keys
    }

    /// Discovered I/O schema for the Mimi decoder model.
    func mimiSchemaKeys() throws -> PocketTtsMimiSchema {
        guard let schema = mimiSchema else {
            throw PocketTTSError.modelNotFound("PocketTTS mimi_decoder schema not discovered")
        }
        return schema
    }

    /// The language root directory (legacy repo root for English, or
    /// `<repoDir>/v2/<lang>` otherwise) — contains the four model files,
    /// `constants_bin/`, and is the right base for `loadMimiInitialState`.
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
    /// encoder is shared across all language packs and lives at the legacy
    /// repo root.
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
        // The Mimi encoder always lives at the repo root regardless of the
        // currently selected language pack.
        let repoRoot: URL
        if let langRoot = languageRootDirectory {
            repoRoot =
                (language.repoSubdirectory == nil)
                ? langRoot
                : langRoot.deletingLastPathComponent().deletingLastPathComponent()
        } else {
            return false
        }
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
