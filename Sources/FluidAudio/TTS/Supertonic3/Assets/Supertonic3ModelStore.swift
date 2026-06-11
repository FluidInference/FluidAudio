@preconcurrency import CoreML
import Foundation

/// Actor-based store for the four Supertonic-3 CoreML models plus the two
/// companion config files (`tts.json`, `unicode_indexer.json`).
///
/// The four stages are:
///   1. `text_encoder`        — text IDs + style → text embedding
///   2. `duration_predictor`  — text IDs + style → per-utterance duration
///   3. `vector_estimator`    — denoising loop input (called N times)
///   4. `vocoder`             — final latent → 44.1 kHz waveform
///
/// All four are intentionally loaded with `.cpuAndNeuralEngine` by default —
/// the converted graphs are FP16 and small enough to fit in ANE working
/// memory. Callers can override at init time (`.cpuOnly` is recommended for
/// Intel Macs and for the smoke tests).
public actor Supertonic3ModelStore {

    private let logger = AppLogger(category: "Supertonic3ModelStore")

    private let directory: URL?
    private let computeUnits: MLComputeUnits
    private let veOption: Supertonic3VectorEstimator

    private var repoDirectory: URL?
    private var textEncoderModel: MLModel?
    private var durationPredictorModel: MLModel?
    private var vocoderModel: MLModel?

    /// Single VectorEstimator for the non-bucketed (`.fp16Dynamic` / `.dynamic`)
    /// modes. `nil` in bucketed mode, where models are loaded lazily per bucket.
    private var vectorEstimatorModel: MLModel?
    /// Lazily-loaded fixed-length VectorEstimators keyed by latent bucket length
    /// (used only in `.aneBucketed` mode).
    private var bucketModels: [Int: MLModel] = [:]

    private(set) var config: Supertonic3Config = .defaults

    public init(
        directory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        vectorEstimator: Supertonic3VectorEstimator = .default
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.veOption = vectorEstimator
    }

    /// Compute units the VectorEstimator stage should use. Dynamic-shape builds
    /// cannot use the ANE (CoreML rejects data-dependent shapes), so they are
    /// pinned to CPU/GPU; bucketed builds target the ANE.
    private var veComputeUnits: MLComputeUnits {
        // An explicit .cpuOnly request always wins (it already excludes the ANE,
        // so no override is needed). Otherwise dynamic shapes avoid the ANE
        // (Core ML rejects them) and bucketed builds target it.
        if computeUnits == .cpuOnly { return .cpuOnly }
        switch veOption {
        case .fp16Dynamic: return computeUnits  // preserve historical behavior
        case .dynamic: return .cpuAndGPU
        case .aneBucketed: return .cpuAndNeuralEngine
        }
    }

    // MARK: - Public API

    /// Download (if missing) the four `.mlmodelc` bundles + `tts.json` +
    /// `unicode_indexer.json` and load the CoreML stages.
    public func loadIfNeeded() async throws {
        if textEncoderModel != nil { return }

        let repoDir = try await Supertonic3ResourceDownloader.ensureModels(
            directory: directory, veVariant: veOption.downloadVariant)
        self.repoDirectory = repoDir

        // tts.json — optional override of the compile-time defaults. If parsing
        // fails we keep the defaults and warn so callers can debug.
        let configURL = repoDir.appendingPathComponent(ModelNames.Supertonic3.configFile)
        if FileManager.default.fileExists(atPath: configURL.path) {
            do {
                let data = try Data(contentsOf: configURL)
                self.config = try JSONDecoder().decode(Supertonic3Config.self, from: data)
            } catch {
                logger.warning(
                    "Failed to decode tts.json (\(error)); using compile-time defaults")
            }
        }

        logger.info("Loading Supertonic-3 CoreML models from \(repoDir.path)…")
        let loadStart = Date()

        let cfg = MLModelConfiguration()
        cfg.computeUnits = computeUnits

        textEncoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Supertonic3.textEncoderFile, config: cfg)
        durationPredictorModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Supertonic3.durationPredictorFile, config: cfg)
        vocoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Supertonic3.vocoderFile, config: cfg)

        // VectorEstimator: dynamic builds load eagerly; bucketed builds load
        // each fixed-length model lazily on first use (see vectorEstimator(forLatentLength:)).
        if !veOption.isBucketed {
            let veCfg = MLModelConfiguration()
            veCfg.computeUnits = veComputeUnits
            vectorEstimatorModel = try loadModel(
                repoDir: repoDir,
                fileName: ModelNames.Supertonic3.vectorEstimatorFile(
                    precisionSuffix: veOption.precisionSuffix, bucket: nil),
                config: veCfg)
        }

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info(
            "Supertonic-3 models loaded in \(String(format: "%.2f", elapsed))s")
    }

    // MARK: - Accessors

    public func textEncoder() throws -> MLModel { try unwrap(textEncoderModel, name: "text_encoder") }
    public func durationPredictor() throws -> MLModel {
        try unwrap(durationPredictorModel, name: "duration_predictor")
    }
    /// Resolve the VectorEstimator for a chunk of `latentLength` TTL slots.
    ///
    /// Returns the model plus the latent length the caller must feed it: in
    /// dynamic modes that equals `latentLength` (no padding); in bucketed mode
    /// it is the smallest published bucket ≥ `latentLength`, and the caller pads
    /// the latent/mask up to that length. Bucket models are loaded lazily and
    /// cached.
    public func vectorEstimator(forLatentLength latentLength: Int) throws -> (model: MLModel, paddedLength: Int) {
        guard veOption.isBucketed else {
            return (try unwrap(vectorEstimatorModel, name: "vector_estimator"), latentLength)
        }
        guard let bucket = ModelNames.Supertonic3.aneBuckets.first(where: { $0 >= latentLength })
        else {
            throw Supertonic3Error.inferenceFailed(
                stage: "vector_estimator",
                underlying:
                    "latent length \(latentLength) exceeds largest ANE bucket "
                    + "\(ModelNames.Supertonic3.aneBuckets.last ?? 0); use a dynamic VectorEstimator")
        }
        if let cached = bucketModels[bucket] { return (cached, bucket) }

        guard let repoDir = repoDirectory else { throw Supertonic3Error.notInitialized }
        let veCfg = MLModelConfiguration()
        veCfg.computeUnits = veComputeUnits
        let model = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.Supertonic3.vectorEstimatorFile(
                precisionSuffix: veOption.precisionSuffix, bucket: bucket),
            config: veCfg)
        bucketModels[bucket] = model
        return (model, bucket)
    }
    public func vocoder() throws -> MLModel { try unwrap(vocoderModel, name: "vocoder") }

    public func repoDir() throws -> URL {
        guard let dir = repoDirectory else { throw Supertonic3Error.notInitialized }
        return dir
    }

    /// Path to `unicode_indexer.json` after `loadIfNeeded()` has succeeded.
    public func unicodeIndexerURL() throws -> URL {
        try repoDir().appendingPathComponent(ModelNames.Supertonic3.unicodeIndexerFile)
    }

    public func unload() {
        textEncoderModel = nil
        durationPredictorModel = nil
        vectorEstimatorModel = nil
        bucketModels.removeAll()
        vocoderModel = nil
    }

    // MARK: - Helpers

    private func unwrap(_ model: MLModel?, name: String) throws -> MLModel {
        guard let model else { throw Supertonic3Error.notInitialized }
        return model
    }

    private func loadModel(
        repoDir: URL, fileName: String, config: MLModelConfiguration
    ) throws -> MLModel {
        let modelURL = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw Supertonic3Error.modelFileNotFound(fileName)
        }
        do {
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            logger.info("Loaded \(fileName)")
            return model
        } catch {
            throw Supertonic3Error.corruptedModel(fileName, underlying: "\(error)")
        }
    }
}
