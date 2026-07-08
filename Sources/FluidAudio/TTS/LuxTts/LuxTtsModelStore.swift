@preconcurrency import CoreML
import Foundation

/// Downloads the LuxTTS CoreML assets from HuggingFace
/// (`FluidInference/luxtts-coreml`) into the shared TTS cache.
public enum LuxTtsResourceDownloader {

    private static let logger = AppLogger(category: "LuxTtsResourceDownloader")

    /// Ensure all required LuxTTS files for the given graph variant are
    /// present locally. Returns the resolved repo directory.
    @discardableResult
    public static func ensureModels(
        directory: URL? = nil,
        variant: String = ModelNames.LuxTts.defaultVariant,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.luxtts.folderName)

        let required = ModelNames.LuxTts.requiredFiles(variant: variant)
        let allPresent = required.allSatisfy { file in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(file).path)
        }

        if !allPresent {
            logger.info("Downloading LuxTTS CoreML assets (\(variant)/) from HuggingFace…")
            do {
                try await ModelHub.download(
                    .luxtts, to: modelsRoot, variant: variant,
                    progressHandler: progressHandler)
            } catch {
                throw LuxTtsError.downloadFailed("\(error)")
            }
        } else {
            logger.info("LuxTTS assets found in cache at \(repoDir.path)")
        }

        return repoDir
    }

    private static func defaultCacheRoot() throws -> URL {
        let root = try TtsCacheDirectory.ensure().appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}

/// Actor-based store for the LuxTTS CoreML models: text encoder +
/// flow-matching decoder (per-platform graph variant) and the two
/// fixed-shape vocoders (loaded lazily by generated-frame budget).
///
/// Compute-unit placement is deliberate:
///   - macOS loads the `gpu/` graphs with `.cpuAndGPU`. The original graph
///     must NOT run on the ANE (rel-pos attention loses precision there and
///     corrupts audio).
///   - iOS loads the `ane/` graphs with `.cpuAndNeuralEngine` (FmDecoder
///     100% ANE-resident, TextEncoder 99%; ~25 MB jetsam-visible footprint).
///   - The vocoder runs with `.cpuAndGPU` everywhere (ISTFT + resample +
///     crossover in-graph, ~2.3 ms on GPU). It is only ~72% ANE-placeable
///     and CPU_AND_NE compilation is flaky, so it never goes on the ANE.
///
/// The `ane/` and `gpu/` graphs share an IDENTICAL external I/O contract —
/// same input/output names, shapes and fp32 dtype (TextEncoder:
/// `tokens (1,256) i32` + `padding_mask (1,256)` → `token_embeds (1,256,100)`;
/// FmDecoder: `x`/`text_condition`/`speech_condition (1,1024,100)` +
/// `t`/`guidance_scale (1)` + `padding_mask (1,1024)` → `v (1,1024,100)`).
/// The "ANE-canonical" rewrite (channel-axis pre-concat, `(1,C,1,S)` working
/// layout) lives entirely INSIDE the MIL and transposes back before output,
/// so `LuxTtsSynthesizer` drives both variants with the same tensor packing —
/// no host-side adapter is required. Select the variant via `variant` (and
/// the `FLUIDAUDIO_LUXTTS_VARIANT` override folded into `defaultVariant`).
public actor LuxTtsModelStore {

    private let logger = AppLogger(category: "LuxTtsModelStore")

    private let directory: URL?
    private let computeUnitsOverride: MLComputeUnits?
    let variant: String

    private var repoDirectory: URL?
    private var textEncoderModel: MLModel?
    private var fmDecoderModel: MLModel?
    private var vocoderModels: [Int: MLModel] = [:]
    private var loadedTokenizer: LuxTtsTokenizer?

    public init(
        directory: URL? = nil,
        variant: String = ModelNames.LuxTts.defaultVariant,
        computeUnitsOverride: MLComputeUnits? = nil
    ) {
        self.directory = directory
        self.variant = variant
        self.computeUnitsOverride = computeUnitsOverride
    }

    private var encoderDecoderComputeUnits: MLComputeUnits {
        Self.encoderDecoderComputeUnits(variant: variant, override: computeUnitsOverride)
    }

    private var vocoderComputeUnits: MLComputeUnits {
        computeUnitsOverride ?? .cpuAndGPU
    }

    /// Encoder/decoder compute-unit policy (pure; unit-tested). `ane/` → ANE,
    /// everything else → GPU. An explicit `computeUnitsOverride` wins.
    static func encoderDecoderComputeUnits(
        variant: String, override: MLComputeUnits?
    ) -> MLComputeUnits {
        if let override { return override }
        return variant == ModelNames.LuxTts.aneVariant ? .cpuAndNeuralEngine : .cpuAndGPU
    }

    // MARK: - Loading

    public func loadIfNeeded(progressHandler: ProgressHandler? = nil) async throws {
        if textEncoderModel != nil { return }

        let repoDir = try await LuxTtsResourceDownloader.ensureModels(
            directory: directory, variant: variant, progressHandler: progressHandler)
        self.repoDirectory = repoDir

        logger.info("Loading LuxTTS CoreML models (\(variant)/) from \(repoDir.path)…")
        let loadStart = Date()

        let config = MLModelConfiguration()
        config.computeUnits = encoderDecoderComputeUnits

        textEncoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.LuxTts.textEncoderFile(variant: variant), config: config)
        fmDecoderModel = try loadModel(
            repoDir: repoDir,
            fileName: ModelNames.LuxTts.fmDecoderFile(variant: variant), config: config)

        loadedTokenizer = try LuxTtsTokenizer(
            tokensFileURL: repoDir.appendingPathComponent(ModelNames.LuxTts.tokensFile))

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("LuxTTS models loaded in \(String(format: "%.2f", elapsed))s")
    }

    // MARK: - Accessors

    public func textEncoder() throws -> MLModel {
        guard let model = textEncoderModel else { throw LuxTtsError.notInitialized }
        return model
    }

    public func fmDecoder() throws -> MLModel {
        guard let model = fmDecoderModel else { throw LuxTtsError.notInitialized }
        return model
    }

    public func tokenizer() throws -> LuxTtsTokenizer {
        guard let tokenizer = loadedTokenizer else { throw LuxTtsError.notInitialized }
        return tokenizer
    }

    /// Vocoder for a window of `bucket` generated frames (282 or 555).
    /// Loaded lazily and cached — short utterances never pay for the 555
    /// bucket, long ones never pay for the 282 one.
    public func vocoder(bucket: Int) throws -> MLModel {
        if let cached = vocoderModels[bucket] { return cached }
        guard let repoDir = repoDirectory else { throw LuxTtsError.notInitialized }
        let fileName =
            bucket == 282
            ? ModelNames.LuxTts.vocoder282File : ModelNames.LuxTts.vocoder555File
        let config = MLModelConfiguration()
        config.computeUnits = vocoderComputeUnits
        let model = try loadModel(repoDir: repoDir, fileName: fileName, config: config)
        vocoderModels[bucket] = model
        return model
    }

    public func unload() {
        textEncoderModel = nil
        fmDecoderModel = nil
        vocoderModels.removeAll()
        loadedTokenizer = nil
    }

    // MARK: - Helpers

    private func loadModel(
        repoDir: URL, fileName: String, config: MLModelConfiguration
    ) throws -> MLModel {
        let modelURL = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw LuxTtsError.modelFileNotFound(fileName)
        }
        do {
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            logger.info("Loaded \(fileName)")
            return model
        } catch {
            throw LuxTtsError.corruptedModel(fileName, underlying: "\(error)")
        }
    }
}
