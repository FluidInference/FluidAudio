@preconcurrency import CoreML
import Foundation

/// Actor-based store for VoxCPM CoreML models and constants.
///
/// Manages loading and storing of the six CoreML models
/// (audio_vae_encoder, audio_vae_decoder, feat_encoder,
/// base_lm_step, residual_lm_step, locdit_step)
/// and the binary constants bundle.
public actor VoxCpmModelStore {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "VoxCpmModelStore")

    private var audioVaeEncoderModel: MLModel?
    private var audioVaeDecoderModel: MLModel?
    private var featEncoderModel: MLModel?
    private var baseLmStepModel: MLModel?
    private var residualLmStepModel: MLModel?
    private var locditStepModel: MLModel?
    private var constantsBundle: VoxCpmConstantsBundle?
    private var repoDirectory: URL?
    private let directory: URL?

    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public init(directory: URL? = nil) {
        self.directory = directory
    }

    /// Load all six CoreML models and the constants bundle.
    public func loadIfNeeded() async throws {
        guard audioVaeEncoderModel == nil else { return }

        let repoDir = try await VoxCpmResourceDownloader.ensureModels(directory: directory)
        self.repoDirectory = repoDir

        logger.info("Loading VoxCPM CoreML models...")

        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let loadStart = Date()

        let modelFiles = [
            ModelNames.VoxCPM.audioVaeEncoderFile,
            ModelNames.VoxCPM.audioVaeDecoderFile,
            ModelNames.VoxCPM.featEncoderFile,
            ModelNames.VoxCPM.baseLmStepFile,
            ModelNames.VoxCPM.residualLmStepFile,
            ModelNames.VoxCPM.locditStepFile,
        ]

        var loadedModels: [MLModel] = []
        for file in modelFiles {
            let modelURL = repoDir.appendingPathComponent(file)
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            loadedModels.append(model)
            logger.info("Loaded \(file)")
        }

        audioVaeEncoderModel = loadedModels[0]
        audioVaeDecoderModel = loadedModels[1]
        featEncoderModel = loadedModels[2]
        baseLmStepModel = loadedModels[3]
        residualLmStepModel = loadedModels[4]
        locditStepModel = loadedModels[5]

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All VoxCPM models loaded in \(String(format: "%.2f", elapsed))s")

        // Load constants
        constantsBundle = try await VoxCpmResourceDownloader.ensureConstants(
            repoDirectory: repoDir)
        logger.info("VoxCPM constants loaded")
    }

    /// The audio VAE encoder model.
    public func audioVaeEncoder() throws -> MLModel {
        guard let model = audioVaeEncoderModel else {
            throw VoxCpmError.modelNotFound("VoxCPM audio_vae_encoder not loaded")
        }
        return model
    }

    /// The audio VAE decoder model.
    public func audioVaeDecoder() throws -> MLModel {
        guard let model = audioVaeDecoderModel else {
            throw VoxCpmError.modelNotFound("VoxCPM audio_vae_decoder not loaded")
        }
        return model
    }

    /// The feature encoder model.
    public func featEncoder() throws -> MLModel {
        guard let model = featEncoderModel else {
            throw VoxCpmError.modelNotFound("VoxCPM feat_encoder not loaded")
        }
        return model
    }

    /// The base LM step model.
    public func baseLmStep() throws -> MLModel {
        guard let model = baseLmStepModel else {
            throw VoxCpmError.modelNotFound("VoxCPM base_lm_step not loaded")
        }
        return model
    }

    /// The residual LM step model.
    public func residualLmStep() throws -> MLModel {
        guard let model = residualLmStepModel else {
            throw VoxCpmError.modelNotFound("VoxCPM residual_lm_step not loaded")
        }
        return model
    }

    /// The LocDiT step model.
    public func locditStep() throws -> MLModel {
        guard let model = locditStepModel else {
            throw VoxCpmError.modelNotFound("VoxCPM locdit_step not loaded")
        }
        return model
    }

    /// The pre-loaded binary constants.
    public func constants() throws -> VoxCpmConstantsBundle {
        guard let bundle = constantsBundle else {
            throw VoxCpmError.modelNotFound("VoxCPM constants not loaded")
        }
        return bundle
    }

    /// The repository directory containing models and constants.
    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else {
            throw VoxCpmError.modelNotFound("VoxCPM repository not loaded")
        }
        return dir
    }
}
