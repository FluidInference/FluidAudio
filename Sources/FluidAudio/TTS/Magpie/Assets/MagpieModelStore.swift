@preconcurrency import CoreML
import Foundation

/// Selects which chunked nanocodec build the store loads.
///
/// - `.fp32` (v3): `nanocodec_decoder_v3.mlmodelc` — full fp32 weights,
///   pinned to CPU (~142.5 ms / 24-frame call on M2). Audibly clean,
///   matches the PyTorch sin² reference within the Snake-approximation
///   noise floor. Pipeline stays real-time at RTFx ~1.3× median.
/// - `.fp16` (v2): `nanocodec_decoder_v2.mlmodelc` — fp16 weights, runs
///   ~43 % ANE-resident at ~38.4 ms / 24-frame call. Faster but audibly
///   noisy on voiced speech (~27 dB SNR vs PyTorch reference, hidden by
///   silence-RMS metrics). Use only when throughput dominates quality.
public enum MagpieNanocodecPrecision: String, Sendable {
    case fp16
    case fp32
}

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
    /// One of:
    ///   - v3: chunked T_in=24 fp32 build (`nanocodec_decoder_v3`, default,
    ///     CPU-only, audibly clean)
    ///   - v2: chunked T_in=24 fp16 build (`nanocodec_decoder_v2`, fast/ANE,
    ///     audibly noisy)
    ///   - v1: monolithic T=256 fp16 build (`nanocodec_decoder`, legacy
    ///     fallback, CPU-only, audibly noisy)
    /// `MagpieNanocodec` reads the input shape and chunks accordingly.
    private var nanocodecDecoderModel: MLModel?

    private var constantsBundle: MagpieConstantsBundle?
    private var localTransformerWeights: MagpieLocalTransformerWeights?

    private var repoDirectory: URL?

    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private let preferredLanguages: Set<MagpieLanguage>
    private let nanocodecPrecision: MagpieNanocodecPrecision

    /// - Parameters:
    ///   - directory: Optional override for the base cache directory.
    ///   - computeUnits: CoreML compute preference for all models.
    ///   - preferredLanguages: Set of languages whose tokenizer data should be fetched.
    ///   - nanocodecPrecision: Which T_in=24 nanocodec build to load.
    ///     `.fp32` (default) is audibly clean but pinned to CPU
    ///     (~142.5 ms/call on M2). `.fp16` runs ~43 % ANE-resident
    ///     (~38.4 ms/call) but is audibly noisy on voiced speech due to
    ///     fp16 weight quantization. See Phase F write-up in
    ///     `mobius/.../per_module/results/STATUS.md`.
    public init(
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        preferredLanguages: Set<MagpieLanguage> = [.english],
        nanocodecPrecision: MagpieNanocodecPrecision = .fp32
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.preferredLanguages = preferredLanguages
        self.nanocodecPrecision = nanocodecPrecision
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

        // Nanocodec compute units are dictated by precision, not by the
        // caller's `computeUnits` setting:
        //
        //   .fp32  → `.cpuOnly`. fp32 weights force the CoreML runtime
        //            off ANE (ANE is fp16-only). Audibly clean.
        //   .fp16  → `.cpuAndNeuralEngine` (or `.cpuOnly` if the caller
        //            explicitly requested CPU-only). Runs ~43 %-resident
        //            on ANE at ~38.4 ms / 24-frame call but is audibly
        //            noisy on voiced speech.
        //
        // The monolithic T=256 fallback build can't use ANE anyway: its
        // activation tensor exceeds the W ≤ 16384 limit on space-to-batch
        // lowering of the dilated convs and ANECCompile() fails
        // whole-graph.
        //
        // Noise floor (quietest 0.3 s window, post-norm):
        //   PyTorch sin² (gold)              -77.4 dBFS
        //   T=24 chunked, fp32 weights, CPU  -73.6 dBFS  (clean, default)
        //   T=24 chunked, fp16 weights, CPU  -73.9 dBFS  (audibly noisy)
        //   T=24 chunked, fp16 weights, ANE  -66.8 dBFS  (audibly noisy)
        //
        // See Phase F in `mobius/.../per_module/results/STATUS.md` for
        // the full mixed-precision sweep that closed off any per-stage
        // or per-op-type fp16/fp32 island.
        let nanocodecConfig = MLModelConfiguration()
        switch nanocodecPrecision {
        case .fp32:
            nanocodecConfig.computeUnits = .cpuOnly
        case .fp16:
            nanocodecConfig.computeUnits =
                computeUnits == .cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
        }

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

        // Pick the requested precision's chunked build first. If it isn't
        // in the repo, fall back to the other precision's chunked build
        // (with a warning), and finally to the legacy monolithic v1
        // (`nanocodec_decoder.mlmodelc`). Exactly one of the three must
        // exist for synthesis to work.
        //
        // Naming:
        //   v1 → `nanocodec_decoder.mlmodelc`     (legacy mono, fp16, noisy)
        //   v2 → `nanocodec_decoder_v2.mlmodelc`  (chunked, fp16, noisy)
        //   v3 → `nanocodec_decoder_v3.mlmodelc`  (chunked, fp32, clean — default)
        let primaryName: String
        let secondaryName: String
        switch nanocodecPrecision {
        case .fp32:
            primaryName = ModelNames.Magpie.nanocodecDecoderV3File
            secondaryName = ModelNames.Magpie.nanocodecDecoderV2File
        case .fp16:
            primaryName = ModelNames.Magpie.nanocodecDecoderV2File
            secondaryName = ModelNames.Magpie.nanocodecDecoderV3File
        }

        nanocodecDecoderModel = try loadModel(
            repoDir: repoDir,
            fileName: primaryName,
            config: nanocodecConfig,
            required: false)
        if nanocodecDecoderModel == nil {
            logger.warning(
                "Requested \(nanocodecPrecision.rawValue) nanocodec (\(primaryName)) absent; trying alternate precision \(secondaryName)"
            )
            // Loading the alternate-precision build under the requested
            // compute config is fine: fp32 weights will force CPU at
            // runtime regardless, fp16 weights honour the chosen units.
            nanocodecDecoderModel = try loadModel(
                repoDir: repoDir,
                fileName: secondaryName,
                config: nanocodecConfig,
                required: false)
        }
        if nanocodecDecoderModel == nil {
            logger.notice(
                "No chunked nanocodec (v2/v3) present; falling back to legacy monolithic CPU-only nanocodec_decoder.mlmodelc (audibly noisy)"
            )
            let monolithicConfig = MLModelConfiguration()
            monolithicConfig.computeUnits = .cpuOnly
            nanocodecDecoderModel = try loadModel(
                repoDir: repoDir,
                fileName: ModelNames.Magpie.nanocodecDecoderFile,
                config: monolithicConfig,
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
