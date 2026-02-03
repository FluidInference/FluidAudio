@preconcurrency import CoreML
import FluidAudio
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
    private var constantsBundle: PocketTtsConstantsBundle?
    private var voiceCache: [String: PocketTtsVoiceData] = [:]
    private var repoDirectory: URL?

    public init() {}

    /// Load all four CoreML models and the constants bundle.
    public func loadIfNeeded() async throws {
        guard condStepModel == nil else { return }

        let repoDir = try await PocketTtsResourceDownloader.ensureModels()
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
            throw TTSError.modelNotFound("PocketTTS cond_step model not loaded")
        }
        return model
    }

    /// The autoregressive generation step model.
    public func flowlmStep() throws -> MLModel {
        guard let model = flowlmStepModel else {
            throw TTSError.modelNotFound("PocketTTS flowlm_step model not loaded")
        }
        return model
    }

    /// The LSD flow decoder model.
    public func flowDecoder() throws -> MLModel {
        guard let model = flowDecoderModel else {
            throw TTSError.modelNotFound("PocketTTS flow_decoder model not loaded")
        }
        return model
    }

    /// The Mimi streaming audio decoder model.
    public func mimiDecoder() throws -> MLModel {
        guard let model = mimiDecoderModel else {
            throw TTSError.modelNotFound("PocketTTS mimi_decoder model not loaded")
        }
        return model
    }

    /// The pre-loaded binary constants.
    public func constants() throws -> PocketTtsConstantsBundle {
        guard let bundle = constantsBundle else {
            throw TTSError.modelNotFound("PocketTTS constants not loaded")
        }
        return bundle
    }

    /// The repository directory containing models and constants.
    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else {
            throw TTSError.modelNotFound("PocketTTS repository not loaded")
        }
        return dir
    }

    /// Load and cache voice conditioning data.
    public func voiceData(for voice: String) throws -> PocketTtsVoiceData {
        if let cached = voiceCache[voice] {
            return cached
        }
        guard let repoDir = repoDirectory else {
            throw TTSError.modelNotFound("PocketTTS repository not loaded")
        }
        let data = try PocketTtsResourceDownloader.ensureVoice(voice, repoDirectory: repoDir)
        voiceCache[voice] = data
        return data
    }
}
