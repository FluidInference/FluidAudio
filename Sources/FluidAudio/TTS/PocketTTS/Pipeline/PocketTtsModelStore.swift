@preconcurrency import CoreML
import Foundation
import OSLog

/// Actor-based store for PocketTTS CoreML models and constants.
///
/// Manages loading and storing of the four CoreML models
/// (cond_step, flowlm_step, flow_decoder, mimi_decoder),
/// the binary constants bundle, and voice conditioning data.
public actor PocketTtsModelStore {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "PocketTtsModelStore")

    private var condStepModel: MLModel?
    private var flowlmStepModel: MLModel?
    private var flowDecoderModel: MLModel?
    private var mimiDecoderModel: MLModel?
    private var mimiEncoderModel: MLModel?
    private var constantsBundle: PocketTtsConstantsBundle?
    private var voiceCache: [String: PocketTtsVoiceData] = [:]
    private var repoDirectory: URL?
    private let directory: URL?

    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public init(directory: URL? = nil) {
        self.directory = directory
    }

    /// Load all four CoreML models and the constants bundle.
    public func loadIfNeeded() async throws {
        guard condStepModel == nil else { return }

        let repoDir = try await PocketTtsResourceDownloader.ensureModels(directory: directory)
        self.repoDirectory = repoDir

        logger.info("Loading PocketTTS CoreML models...")

        // Use CPU+GPU for all models to avoid ANE float16 precision loss.
        // The ANE processes in native float16, which causes audible artifacts
        // in the Mimi decoder's streaming state feedback loop and may degrade
        // quality in the other models. CPU/GPU compute in float32 matches the
        // Python reference implementation.
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let loadStart = Date()

        let modelFiles = [
            ModelNames.PocketTTS.condStepFile,
            ModelNames.PocketTTS.flowlmStepFile,
            ModelNames.PocketTTS.flowDecoderFile,
            ModelNames.PocketTTS.mimiDecoderFile,
        ]

        var loadedModels: [MLModel] = []
        for file in modelFiles {
            let modelURL = repoDir.appendingPathComponent(file)
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            loadedModels.append(model)
            logger.info("Loaded \(file)")
        }

        condStepModel = loadedModels[0]
        flowlmStepModel = loadedModels[1]
        flowDecoderModel = loadedModels[2]
        mimiDecoderModel = loadedModels[3]

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All PocketTTS models loaded in \(String(format: "%.2f", elapsed))s")

        // Load constants
        constantsBundle = try PocketTtsResourceDownloader.ensureConstants(
            repoDirectory: repoDir)
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

    /// The repository directory containing models and constants.
    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else {
            throw PocketTTSError.modelNotFound("PocketTTS repository not loaded")
        }
        return dir
    }

    /// Load and cache voice conditioning data.
    public func voiceData(for voice: String) throws -> PocketTtsVoiceData {
        if let cached = voiceCache[voice] {
            return cached
        }
        guard let repoDir = repoDirectory else {
            throw PocketTTSError.modelNotFound("PocketTTS repository not loaded")
        }
        let data = try PocketTtsResourceDownloader.ensureVoice(voice, repoDirectory: repoDir)
        voiceCache[voice] = data
        return data
    }

    // MARK: - Voice Cloning

    /// Load the Mimi encoder model for voice cloning (lazy, on-demand).
    ///
    /// Downloads the model from HuggingFace if not already cached.
    public func loadMimiEncoderIfNeeded() async throws {
        guard mimiEncoderModel == nil else { return }

        // Ensure the mimi_encoder is downloaded (downloads if needed)
        let modelURL = try await PocketTtsResourceDownloader.ensureMimiEncoder(directory: directory)

        // Update repoDirectory if not set
        if repoDirectory == nil {
            repoDirectory = modelURL.deletingLastPathComponent()
        }

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
        guard let repoDir = repoDirectory else { return false }
        let modelURL = repoDir.appendingPathComponent(ModelNames.PocketTTS.mimiEncoderFile)
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
