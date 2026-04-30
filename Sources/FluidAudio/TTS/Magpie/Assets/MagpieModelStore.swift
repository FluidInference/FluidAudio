@preconcurrency import CoreML
import Foundation

/// Actor-based store for Magpie CoreML models + constants + LocalTransformer weights.
///
/// Manages loading of 3 required models (text_encoder, decoder_step, nanocodec_decoder)
/// and 1 optional model (decoder_prefill). Also holds the pre-loaded
/// `MagpieConstantsBundle` and `MagpieLocalTransformerWeights` so the synthesizer
/// can hit all assets from a single entry point.
public actor MagpieModelStore {

    private let logger = AppLogger(category: "MagpieModelStore")

    private var textEncoderModel: MLModel?
    private var decoderPrefillModel: MLModel?  // optional fast path
    private var decoderStepModel: MLModel?
    private var nanocodecDecoderModel: MLModel?

    private var constantsBundle: MagpieConstantsBundle?
    private var localTransformerWeights: MagpieLocalTransformerWeights?

    private var repoDirectory: URL?

    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private let preferredLanguages: Set<MagpieLanguage>

    /// - Parameters:
    ///   - directory: Optional override for the base cache directory.
    ///   - computeUnits: CoreML compute preference for all models.
    ///   - preferredLanguages: Set of languages whose tokenizer data should be fetched.
    public init(
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        preferredLanguages: Set<MagpieLanguage> = [.english]
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.preferredLanguages = preferredLanguages
    }

    /// Download (if missing) and load all Magpie CoreML models + constants.
    public func loadIfNeeded() async throws {
        if textEncoderModel != nil {
            return
        }

        let repoDir = try await MagpieResourceDownloader.ensureAssets(
            languages: preferredLanguages,
            directory: directory,
            includePrefill: true
        )
        self.repoDirectory = repoDir

        logger.info("Loading Magpie CoreML models from \(repoDir.path)…")

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // `decoder_step.mlmodelc` reliably fails ANE compilation
        // (`MILCompilerForANE error: ANECCompile() FAILED`) due to its rank-4
        // split-K/V scatter layout, then falls back to CPU at the cost of one
        // failed ANE compile attempt per call (~hundreds of ms each). Pin it
        // to `.cpuAndGPU` so CoreML skips the ANE attempt entirely and runs
        // on Metal MPS — verified end-to-end as the fastest path
        // (96s warm vs 103s warm on `.cpuAndNeuralEngine`).
        let gpuConfig = MLModelConfiguration()
        gpuConfig.computeUnits =
            computeUnits == .cpuOnly ? .cpuOnly : .cpuAndGPU

        // `nanocodec_decoder.mlmodelc` is fastest on **CPU only**. The model's
        // upsample stack (5 transposed convs + 96 sin/pow per-frame embedding
        // ops + 86 LeakyReLU) doesn't map well onto Metal MPS, and ANE compile
        // fails on its conv stack. Empirically (M-series, single fwd of 256
        // frames):
        //   .cpuOnly             ~2.87 s
        //   .cpuAndGPU           ~3.86 s
        //   .cpuAndNeuralEngine ~10.12 s   (ANE compile fail → CPU fallback dance)
        //   .all                 ~2.95 s
        // Putting it on `.cpuAndGPU` also makes `decoder_step` ~40 ms/step
        // because both contend for the same Metal queue. Pinning nanocodec to
        // CPU keeps Metal exclusive for decoder_step (25 ms/step) and saves a
        // full second on the nanocodec call → ~1.03x RTFx vs ~0.91x before.
        let cpuConfig = MLModelConfiguration()
        cpuConfig.computeUnits = .cpuOnly

        let loadStart = Date()

        textEncoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Magpie.textEncoderFile,
            config: config,
            required: true)

        decoderStepModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Magpie.decoderStepFile,
            config: gpuConfig,
            required: true)

        nanocodecDecoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Magpie.nanocodecDecoderFile,
            config: cpuConfig,
            required: true)

        decoderPrefillModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Magpie.decoderPrefillFile,
            config: config,
            required: false)

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info(
            "Magpie models loaded in \(String(format: "%.2f", elapsed))s (prefill \(decoderPrefillModel == nil ? "absent" : "present"))"
        )

        // Load constants + local transformer weights.
        let constantsDir = MagpieResourceDownloader.constantsDirectory(in: repoDir)
        let bundle = try MagpieConstantsLoader.load(from: constantsDir)
        constantsBundle = bundle
        localTransformerWeights = try MagpieLocalTransformerLoader.load(
            from: constantsDir, config: bundle.config)
    }

    public func textEncoder() throws -> MLModel {
        guard let model = textEncoderModel else {
            throw MagpieError.notInitialized
        }
        return model
    }

    public func decoderStep() throws -> MLModel {
        guard let model = decoderStepModel else {
            throw MagpieError.notInitialized
        }
        return model
    }

    public func nanocodecDecoder() throws -> MLModel {
        guard let model = nanocodecDecoderModel else {
            throw MagpieError.notInitialized
        }
        return model
    }

    public func decoderPrefill() throws -> MLModel {
        guard let model = decoderPrefillModel else {
            throw MagpieError.notInitialized
        }
        return model
    }

    public func hasDecoderPrefill() -> Bool {
        decoderPrefillModel != nil
    }

    public func constants() throws -> MagpieConstantsBundle {
        guard let bundle = constantsBundle else {
            throw MagpieError.notInitialized
        }
        return bundle
    }

    public func localTransformer() throws -> MagpieLocalTransformerWeights {
        guard let weights = localTransformerWeights else {
            throw MagpieError.notInitialized
        }
        return weights
    }

    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else {
            throw MagpieError.notInitialized
        }
        return dir
    }

    /// Release all loaded models + constants. Resource downloads on disk are kept.
    public func unload() {
        textEncoderModel = nil
        decoderPrefillModel = nil
        decoderStepModel = nil
        nanocodecDecoderModel = nil
        constantsBundle = nil
        localTransformerWeights = nil
    }

    // MARK: - Helpers

    private func loadModel(
        repoDir: URL, fileName: String, config: MLModelConfiguration, required: Bool
    ) throws -> MLModel? {
        let modelURL = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            if required {
                throw MagpieError.modelFileNotFound(fileName)
            } else {
                logger.notice("Optional model \(fileName) not present; skipping")
                return nil
            }
        }
        do {
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            logger.info("Loaded \(fileName)")
            return model
        } catch {
            if required {
                throw MagpieError.corruptedModel(fileName, underlying: "\(error)")
            } else {
                logger.warning("Failed to load optional \(fileName): \(error)")
                return nil
            }
        }
    }
}
