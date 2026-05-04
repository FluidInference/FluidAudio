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
    /// Either the chunked T_in=24 build (preferred, ANE) or the monolithic
    /// T=256 build (legacy, CPU-only). `MagpieNanocodec` reads the input
    /// shape and chunks accordingly.
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

        // `decoder_step.mlmodelc` runs cleanly on ANE: coreml-cli static
        // analysis reports 97.3 % ANE residency (765 / 773 ops; the 8 CPU
        // ops are trivial int32 casts / a `select` with ~0.14 ms total cost).
        // Single-call median predict is 15.2 ms on `.cpuAndNeuralEngine`
        // vs 22.5 ms on `.cpuAndGPU` (Apple M2). End-to-end on the ~110-char
        // bench sentence (seed=42, John, en):
        //   .cpuAndGPU             36.1 ms/step, 30.98 s synth, RTFx 0.62×,
        //                          and the AR loop never reaches EOS
        //                          (`EOS: false`, hits maxSteps with tail
        //                          garbage) — 651 codes / 19.23 s audio.
        //   .cpuAndNeuralEngine    23.9 ms/step bench / 14.3 ms/step
        //                          standalone, 9.25 s synth, RTFx 1.21×,
        //                          terminates correctly (`EOS: true`,
        //                          234 codes / 11.20 s audio).
        // ANE is both ~2× faster and produces correctly-terminated audio,
        // so pin to `.cpuAndNeuralEngine`. (Older comment claiming
        // `MILCompilerForANE error: ANECCompile() FAILED` no longer holds —
        // the rank-4 split-K/V scatter compiles cleanly.)
        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits =
            computeUnits == .cpuOnly ? .cpuOnly : .cpuAndNeuralEngine

        // Nanocodec is pinned to CPU regardless of `computeUnits`.
        //
        // The monolithic T=256 build can't use ANE anyway: its activation
        // tensor exceeds the W ≤ 16384 limit on space-to-batch lowering of
        // the dilated convs and ANECCompile() fails whole-graph.
        //
        // The chunked T_in=24 build *can* run ~43 %-resident on ANE, but
        // we ship it as fp32 weights (`compute_precision=FLOAT32` in
        // `convert_nanocodec.py`) because fp16 weight quantization
        // produces audible speech-correlated noise during voiced segments
        // — the silence-RMS metrics hide it, but A/B listening against
        // a PyTorch sin² reference makes it obvious. fp32 weights force
        // the CoreML runtime onto CPU (ANE is fp16-only), and the noise
        // floor matches the PyTorch reference within ~3.5 dB.
        //
        // Noise floor (quietest 0.3 s window, post-norm):
        //   PyTorch sin² (gold)              -77.4 dBFS
        //   T=24 chunked, fp32 weights, CPU  -73.6 dBFS  ← current
        //   T=24 chunked, fp16 weights, CPU  -73.9 dBFS  (audibly noisy)
        //   T=24 chunked, fp16 weights, ANE  -66.8 dBFS  (audibly noisy)
        //
        // fp32 trades throughput for fidelity: nanocodec wall is ~4×
        // slower than fp16 (8.5–9.7 s vs 2.2 s on a 12 s utterance, M2),
        // but the pipeline stays real-time at RTFx ~1.3× median.
        let nanocodecConfig = MLModelConfiguration()
        nanocodecConfig.computeUnits = .cpuOnly

        let loadStart = Date()

        textEncoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Magpie.textEncoderFile,
            config: config,
            required: true)

        decoderStepModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Magpie.decoderStepFile,
            config: aneConfig,
            required: true)

        // Prefer the chunked T_in=24 build (loads on ANE). Fall back to the
        // monolithic T=256 build if t24 isn't present in the repo. Exactly
        // one of the two must exist for synthesis to work.
        nanocodecDecoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Magpie.nanocodecDecoderT24File,
            config: nanocodecConfig,
            required: false)
        if nanocodecDecoderModel == nil {
            logger.notice(
                "T_in=24 nanocodec absent; falling back to monolithic CPU-only nanocodec_decoder.mlmodelc"
            )
            nanocodecDecoderModel = try loadModel(
                repoDir: repoDir,
                fileName: ModelNames.Magpie.nanocodecDecoderFile,
                config: nanocodecConfig,
                required: true)
        }

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
