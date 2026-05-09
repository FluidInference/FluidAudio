@preconcurrency import CoreML
import Foundation

/// Actor-based store for the 8 StyleTTS2 default-path CoreML models plus
/// up-to-3 lazily-loaded bucket variants (T = 64 / 128 / 256) of `bert` and
/// `fused_diffusion_sampler`.
///
/// Per-stage compute units mirror the placement decisions documented in
/// `mobius/models/tts/styletts2/coreml/inference.py` (post Trials 4 + 6 + 8b):
/// `text_encoder`, `duration_predictor`, `fused_f0n_har_source`, and
/// `decoder_upsample` are CPU-only; `bert` and `fused_diffusion_sampler` go
/// to ANE+CPU+GPU; `ref_encoder` is CPU+GPU; `decoder_pre` is CPU+ANE.
///
/// Caller passing `cpuOnly` forces every stage to CPU regardless.
public actor StyleTTS2ModelStore {

    private let logger = AppLogger(category: "StyleTTS2ModelStore")

    // MARK: - Default-path models (always loaded on initialize)
    private var textEncoderModel: MLModel?
    private var bertModel: MLModel?  // T = 57 default
    private var refEncoderModel: MLModel?
    private var fusedDiffusionSamplerModel: MLModel?  // T = 57 default
    private var durationPredictorModel: MLModel?
    private var fusedF0nHarSourceModel: MLModel?
    private var decoderPreModel: MLModel?
    private var decoderUpsampleModel: MLModel?

    // MARK: - Bucket variants (lazy)
    /// Map from bucket token size (64 / 128 / 256) to bert MLModel.
    private var bucketBertModels: [Int: MLModel] = [:]
    /// Map from bucket token size to fused sampler MLModel.
    private var bucketSamplerModels: [Int: MLModel] = [:]

    private var repoDirectory: URL?

    private let directory: URL?
    private let computeUnits: MLComputeUnits

    public init(
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
    }

    // MARK: - Public API

    /// Download (if missing) and load all 8 default-path StyleTTS2 models.
    /// Bucket variants are loaded on demand via `bertModel(forBucket:)` /
    /// `samplerModel(forBucket:)`.
    public func loadIfNeeded() async throws {
        if textEncoderModel != nil {
            return
        }

        let repoDir = try await StyleTTS2ResourceDownloader.ensureDefaultModels(
            directory: directory)
        self.repoDirectory = repoDir

        logger.info("Loading StyleTTS2 CoreML models from \(repoDir.path)…")
        let loadStart = Date()

        textEncoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.textEncoderFile,
            config: config(for: .cpuOnly),
            required: true)

        bertModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.bertFile,
            config: config(for: .all),
            required: true)

        refEncoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.refEncoderFile,
            config: config(for: .cpuAndGPU),
            required: true)

        fusedDiffusionSamplerModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.fusedDiffusionSamplerFile,
            config: config(for: .all),
            required: true)

        durationPredictorModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.durationPredictorFile,
            config: config(for: .cpuOnly),
            required: true)

        fusedF0nHarSourceModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.fusedF0nHarSourceFile,
            config: config(for: .cpuOnly),
            required: true)

        decoderPreModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.decoderPreFile,
            config: config(for: .cpuAndNeuralEngine),
            required: true)

        decoderUpsampleModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.StyleTTS2.decoderUpsampleFile,
            config: config(for: .cpuOnly),
            required: true)

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info(
            "StyleTTS2 default models loaded in \(String(format: "%.2f", elapsed))s")
    }

    // MARK: - Default-path accessors

    public func textEncoder() throws -> MLModel { try unwrap(textEncoderModel, name: "text_encoder") }
    public func bert() throws -> MLModel { try unwrap(bertModel, name: "bert") }
    public func refEncoder() throws -> MLModel { try unwrap(refEncoderModel, name: "ref_encoder") }
    public func fusedDiffusionSampler() throws -> MLModel {
        try unwrap(fusedDiffusionSamplerModel, name: "fused_diffusion_sampler")
    }
    public func durationPredictor() throws -> MLModel {
        try unwrap(durationPredictorModel, name: "duration_predictor")
    }
    public func fusedF0nHarSource() throws -> MLModel {
        try unwrap(fusedF0nHarSourceModel, name: "fused_f0n_har_source")
    }
    public func decoderPre() throws -> MLModel { try unwrap(decoderPreModel, name: "decoder_pre") }
    public func decoderUpsample() throws -> MLModel {
        try unwrap(decoderUpsampleModel, name: "decoder_upsample")
    }

    // MARK: - Bucket accessors

    /// Return the `bert` model for the smallest bucket that fits `tokenCount`
    /// tokens. Returns the default T=57 model when `tokenCount <= 57`.
    /// Downloads + loads the bucket bundle if it isn't cached yet.
    public func bertModel(forTokenCount tokenCount: Int) async throws -> MLModel {
        if tokenCount <= StyleTTS2Constants.defaultBertTokens {
            return try bert()
        }
        let t = try resolveBucket(for: tokenCount)
        if let cached = bucketBertModels[t] {
            return cached
        }
        let repoDir = try repoDir()
        try await StyleTTS2ResourceDownloader.ensureBucket(forT: t, in: repoDir)
        let fileName: String
        switch t {
        case 64: fileName = ModelNames.StyleTTS2.bertT64File
        case 128: fileName = ModelNames.StyleTTS2.bertT128File
        case 256: fileName = ModelNames.StyleTTS2.bertT256File
        default: throw StyleTTS2Error.noBucketAvailable(tokenCount: tokenCount)
        }
        let model = try loadModel(
            repoDir: repoDir, fileName: fileName,
            config: config(for: .all), required: true)
        guard let model else {
            throw StyleTTS2Error.modelFileNotFound(fileName)
        }
        bucketBertModels[t] = model
        return model
    }

    /// Same as `bertModel(forTokenCount:)` for the fused diffusion sampler.
    public func samplerModel(forTokenCount tokenCount: Int) async throws -> MLModel {
        if tokenCount <= StyleTTS2Constants.defaultBertTokens {
            return try fusedDiffusionSampler()
        }
        let t = try resolveBucket(for: tokenCount)
        if let cached = bucketSamplerModels[t] {
            return cached
        }
        let repoDir = try repoDir()
        try await StyleTTS2ResourceDownloader.ensureBucket(forT: t, in: repoDir)
        let fileName: String
        switch t {
        case 64: fileName = ModelNames.StyleTTS2.fusedDiffusionSamplerT64File
        case 128: fileName = ModelNames.StyleTTS2.fusedDiffusionSamplerT128File
        case 256: fileName = ModelNames.StyleTTS2.fusedDiffusionSamplerT256File
        default: throw StyleTTS2Error.noBucketAvailable(tokenCount: tokenCount)
        }
        let model = try loadModel(
            repoDir: repoDir, fileName: fileName,
            config: config(for: .all), required: true)
        guard let model else {
            throw StyleTTS2Error.modelFileNotFound(fileName)
        }
        bucketSamplerModels[t] = model
        return model
    }

    /// Smallest bucket size from `bucketTokenSizes` that fits `tokenCount`.
    /// Throws `noBucketAvailable` if `tokenCount` exceeds the largest
    /// configured bucket.
    public func resolveBucket(for tokenCount: Int) throws -> Int {
        for size in StyleTTS2Constants.bucketTokenSizes where tokenCount <= size {
            return size
        }
        throw StyleTTS2Error.noBucketAvailable(tokenCount: tokenCount)
    }

    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else {
            throw StyleTTS2Error.notInitialized
        }
        return dir
    }

    /// Release all loaded models. Resource downloads on disk are kept.
    public func unload() {
        textEncoderModel = nil
        bertModel = nil
        refEncoderModel = nil
        fusedDiffusionSamplerModel = nil
        durationPredictorModel = nil
        fusedF0nHarSourceModel = nil
        decoderPreModel = nil
        decoderUpsampleModel = nil
        bucketBertModels.removeAll(keepingCapacity: false)
        bucketSamplerModels.removeAll(keepingCapacity: false)
    }

    // MARK: - Helpers

    /// Build an `MLModelConfiguration` honoring caller's CPU-only override.
    /// `desired` is the per-stage placement; if the caller forced
    /// `.cpuOnly` at init we pin everything to CPU regardless.
    private func config(for desired: MLComputeUnits) -> MLModelConfiguration {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = (computeUnits == .cpuOnly) ? .cpuOnly : desired
        return cfg
    }

    private func unwrap(_ model: MLModel?, name: String) throws -> MLModel {
        guard let model else { throw StyleTTS2Error.notInitialized }
        return model
    }

    private func loadModel(
        repoDir: URL, fileName: String, config: MLModelConfiguration, required: Bool
    ) throws -> MLModel? {
        let modelURL = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            if required {
                throw StyleTTS2Error.modelFileNotFound(fileName)
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
                throw StyleTTS2Error.corruptedModel(fileName, underlying: "\(error)")
            } else {
                logger.warning("Failed to load optional \(fileName): \(error)")
                return nil
            }
        }
    }
}
