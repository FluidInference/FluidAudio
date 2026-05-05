@preconcurrency import CoreML
import Foundation

/// Which chunked nanocodec build to load.
/// `.fp32` (v3) is the audibly-clean default; `.fp16` (v2) is faster
/// (~4×, partial ANE) but noisy on voiced speech. `.fp32Pal` (v4) is
/// fp32 compute with 8-bit kmeans palettized weights — acoustically
/// transparent vs v3 at ~4× smaller on disk and ~11 % lower peak RSS.
public enum MagpieNanocodecPrecision: String, Sendable {
    case fp16
    case fp32
    case fp32Pal
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
    /// v3 / v2 / v1 nanocodec — `MagpieNanocodec` chunks based on the
    /// model's input shape.
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
    ///   - preferredLanguages: Languages whose tokenizer data is fetched.
    ///   - nanocodecPrecision: Chunked nanocodec build (default `.fp32`).
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

        // `decoder_step` is 97 % ANE-resident; pinning to ANE is ~2× faster
        // than CPU+GPU and the only path that terminates correctly via EOS.
        let aneConfig = MLModelConfiguration()
        aneConfig.computeUnits =
            computeUnits == .cpuOnly ? .cpuOnly : .cpuAndNeuralEngine

        // Nanocodec compute units track precision: fp32 / fp32Pal → CPU-only
        // (ANE is fp16-only; palettized weights dequantize to fp32 at runtime
        // so compute is still fp32); fp16 → ANE unless caller explicitly
        // pinned CPU.
        func nanocodecConfig(for precision: MagpieNanocodecPrecision) -> MLModelConfiguration {
            let cfg = MLModelConfiguration()
            switch precision {
            case .fp32, .fp32Pal:
                cfg.computeUnits = .cpuOnly
            case .fp16:
                cfg.computeUnits =
                    computeUnits == .cpuOnly ? .cpuOnly : .cpuAndNeuralEngine
            }
            return cfg
        }

        // Filename for a chunked nanocodec build.
        func nanocodecFile(for precision: MagpieNanocodecPrecision) -> String {
            switch precision {
            case .fp16: return ModelNames.Magpie.nanocodecDecoderV2File
            case .fp32: return ModelNames.Magpie.nanocodecDecoderV3File
            case .fp32Pal: return ModelNames.Magpie.nanocodecDecoderV4File
            }
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

        // Try the requested precision, then the other chunked builds in a
        // sensible audibility-preserving order, then the legacy monolithic
        // v1. Each candidate carries its own config so the fallback doesn't
        // inherit the primary's compute-unit selection. fp16 (v2) is only
        // reached when explicitly requested or when no other candidate is
        // present, since it's audibly noisy on voiced speech.
        let fallbackOrder: [MagpieNanocodecPrecision]
        switch nanocodecPrecision {
        case .fp32Pal:
            fallbackOrder = [.fp32Pal, .fp32, .fp16]
        case .fp32:
            fallbackOrder = [.fp32, .fp32Pal, .fp16]
        case .fp16:
            fallbackOrder = [.fp16, .fp32Pal, .fp32]
        }

        for (index, candidate) in fallbackOrder.enumerated() {
            let candidateName = nanocodecFile(for: candidate)
            if index > 0 {
                logger.warning(
                    "Requested \(fallbackOrder[0].rawValue) nanocodec absent; trying \(candidate.rawValue) (\(candidateName))"
                )
            }
            nanocodecDecoderModel = try loadModel(
                repoDir: repoDir,
                fileName: candidateName,
                config: nanocodecConfig(for: candidate),
                required: false)
            if nanocodecDecoderModel != nil { break }
        }
        if nanocodecDecoderModel == nil {
            logger.notice(
                "No chunked nanocodec (v2/v3/v4) present; falling back to legacy monolithic CPU-only nanocodec_decoder.mlmodelc (audibly noisy)"
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
