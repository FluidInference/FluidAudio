@preconcurrency import CoreML
import Foundation

/// Actor-based store for the 12 StyleTTS2 CoreML models.
///
/// Stage layout & compute placement (mirrors `coreml/PRECISION.md` from the
/// upstream conversion repo):
///
/// | Stage          | Bucket axis     | Buckets                       | Precision | Compute     |
/// |----------------|-----------------|-------------------------------|-----------|-------------|
/// | text_predictor | input tokens    | 32, 64, 128, 256, 512         | fp16      | ANE         |
/// | diffusion_step | bert_dur frames | 512 only                      | fp16      | CPU+GPU     |
/// | f0n_energy     | dynamic shape   | n/a                           | fp16      | ANE         |
/// | decoder        | mel frames      | 256, 512, 1024, 2048, 4096    | fp32      | CPU+GPU     |
///
/// Lazy: only the buckets that are actually requested are loaded into
/// memory. Eager warmup of every bucket on init would cost ~1.4 GB of RAM
/// for the decoder alone.
public actor StyleTTS2ModelStore {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "StyleTTS2ModelStore")

    // text_predictor: lazy per-bucket cache.
    private var textPredictorModels: [Int: MLModel] = [:]
    // diffusion_step: single 512-bucket model, loaded on first use.
    private var diffusionStepModel: MLModel?
    // f0n_energy: dynamic shape, single model.
    private var f0nEnergyModel: MLModel?
    // decoder: lazy per-bucket cache (fp32, large).
    private var decoderModels: [Int: MLModel] = [:]

    private var repoRootDirectory: URL?
    private var cachedVocab: StyleTTS2Vocab?
    private var cachedBundleConfig: StyleTTS2BundleConfig?
    private let directory: URL?

    public init(directory: URL? = nil) {
        self.directory = directory
    }

    // MARK: - Bring-up

    /// Ensure the bundle is downloaded and resolve the repo root. Does **not**
    /// load any MLModel instances — those are loaded lazily per call site.
    public func ensureAssetsAvailable(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        if let dir = repoRootDirectory {
            return dir
        }
        let dir = try await StyleTTS2ResourceDownloader.ensureModels(
            directory: directory,
            progressHandler: progressHandler
        )
        repoRootDirectory = dir
        return dir
    }

    // MARK: - text_predictor (ANE, fp16, 5 buckets)

    /// Round `tokenLength` up to the nearest shipped text_predictor bucket.
    public nonisolated func textPredictorBucket(forTokenLength tokenLength: Int) -> Int {
        for bucket in StyleTTS2Constants.textPredictorBuckets where tokenLength <= bucket {
            return bucket
        }
        // Saturate at the largest shipped bucket; caller is responsible for
        // chunking inputs longer than 512 tokens.
        return StyleTTS2Constants.textPredictorBuckets.last ?? 512
    }

    public func textPredictor(bucket: Int) async throws -> MLModel {
        if let cached = textPredictorModels[bucket] {
            return cached
        }
        guard StyleTTS2Constants.textPredictorBuckets.contains(bucket) else {
            throw StyleTTS2Error.invalidConfiguration(
                "Unsupported text_predictor bucket: \(bucket)")
        }
        let model = try await loadModel(
            named: "compiled/styletts2_text_predictor_\(bucket).mlmodelc",
            computeUnits: .cpuAndNeuralEngine,
            label: "text_predictor_\(bucket)"
        )
        textPredictorModels[bucket] = model
        return model
    }

    // MARK: - diffusion_step (CPU+GPU, fp16, 512 only)

    public func diffusionStep() async throws -> MLModel {
        if let cached = diffusionStepModel {
            return cached
        }
        let model = try await loadModel(
            named: ModelNames.StyleTTS2.diffusionStep512File,
            computeUnits: .cpuAndGPU,
            label: "diffusion_step_512"
        )
        diffusionStepModel = model
        return model
    }

    // MARK: - f0n_energy (ANE, fp16, dynamic)

    public func f0nEnergy() async throws -> MLModel {
        if let cached = f0nEnergyModel {
            return cached
        }
        // f0n_energy ships with dynamic shape — the E5RT runtime rejects
        // MLMultiArrays with known strides on FlexibleShapeInfo models
        // when run on GPU/ANE, so we pin this stage to CPU. Per-utterance
        // cost is small (one call), the CPU fallback is acceptable.
        let model = try await loadModel(
            named: ModelNames.StyleTTS2.f0nEnergyFile,
            computeUnits: .cpuOnly,
            label: "f0n_energy"
        )
        f0nEnergyModel = model
        return model
    }

    // MARK: - decoder (CPU+GPU, fp32, 5 buckets)

    /// Round `melFrames` up to the nearest shipped decoder bucket.
    public nonisolated func decoderBucket(forMelFrames melFrames: Int) -> Int {
        for bucket in StyleTTS2Constants.decoderBuckets where melFrames <= bucket {
            return bucket
        }
        return StyleTTS2Constants.decoderBuckets.last ?? 4096
    }

    public func decoder(bucket: Int) async throws -> MLModel {
        if let cached = decoderModels[bucket] {
            return cached
        }
        guard StyleTTS2Constants.decoderBuckets.contains(bucket) else {
            throw StyleTTS2Error.invalidConfiguration(
                "Unsupported decoder bucket: \(bucket)")
        }
        let model = try await loadModel(
            named: "compiled/styletts2_decoder_\(bucket).mlmodelc",
            computeUnits: .cpuAndGPU,
            label: "decoder_\(bucket)"
        )
        decoderModels[bucket] = model
        return model
    }

    // MARK: - Bundle paths

    /// Path to the espeak-ng IPA vocabulary JSON.
    public func vocabularyURL() throws -> URL {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 repo not loaded")
        }
        return root.appendingPathComponent(ModelNames.StyleTTS2.vocabularyFile)
    }

    /// Path to the bundle `config.json`.
    public func configURL() throws -> URL {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 repo not loaded")
        }
        return root.appendingPathComponent(ModelNames.StyleTTS2.configFile)
    }

    /// Lazily-loaded 178-token IPA vocabulary.
    public func vocabulary() throws -> StyleTTS2Vocab {
        if let cached = cachedVocab {
            return cached
        }
        let url = try vocabularyURL()
        let vocab = try StyleTTS2Vocab.load(from: url)
        cachedVocab = vocab
        return vocab
    }

    /// Lazily-loaded `config.json`. Decoded once and cached for the lifetime
    /// of the store.
    public func bundleConfig() throws -> StyleTTS2BundleConfig {
        if let cached = cachedBundleConfig {
            return cached
        }
        let url = try configURL()
        let config = try StyleTTS2BundleConfig.load(from: url)
        cachedBundleConfig = config
        return config
    }

    public func repoRoot() throws -> URL {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 repo not loaded")
        }
        return root
    }

    // MARK: - Private

    private func loadModel(
        named filename: String,
        computeUnits: MLComputeUnits,
        label: String
    ) async throws -> MLModel {
        guard let root = repoRootDirectory else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 repo not loaded")
        }
        let url = root.appendingPathComponent(filename)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StyleTTS2Error.modelNotFound(filename)
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let start = Date()
        let model = try MLModel(contentsOf: url, configuration: config)
        let elapsed = Date().timeIntervalSince(start)
        logger.info(
            "Loaded \(label) in \(String(format: "%.2f", elapsed))s :: \(SystemInfo.summary())")
        return model
    }
}
