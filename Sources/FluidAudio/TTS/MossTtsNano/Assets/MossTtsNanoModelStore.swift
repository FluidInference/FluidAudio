@preconcurrency import CoreML
import Foundation

/// Holds the four streaming-path CoreML models plus config and tokenizer.
///
/// Prefill and Step are pinned to CPU+GPU: their graphs fail ANE compilation
/// (`ANECCompile FAILED`) and would only fall back after a slow compile attempt.
/// Frame and CodecStep run on any unit; the caller's `computeUnits` applies to
/// them. The fp32 codec encoder (voice cloning) is loaded on demand.
public actor MossTtsNanoModelStore {

    private let logger = AppLogger(category: "MossTtsNanoModelStore")

    private let directory: URL?
    private let computeUnits: MLComputeUnits

    private var repoDirectory: URL?
    private var prefillModel: MLModel?
    private var stepModel: MLModel?
    private var frameModel: MLModel?
    private var codecStepModel: MLModel?
    private var encoderModel: MLModel?
    private var loadedConfig: MossTtsNanoConfig?
    private var loadedTokenizer: MossTtsNanoTokenizer?

    public init(directory: URL? = nil, computeUnits: MLComputeUnits = .cpuAndGPU) {
        self.directory = directory
        self.computeUnits = computeUnits
    }

    private var lmComputeUnits: MLComputeUnits {
        computeUnits == .cpuOnly ? .cpuOnly : .cpuAndGPU
    }

    public func loadIfNeeded() async throws {
        if prefillModel != nil { return }
        let repoDir = try await MossTtsNanoResourceDownloader.ensureModels(directory: directory)
        repoDirectory = repoDir

        do {
            loadedConfig = try MossTtsNanoConfig.load(
                from: repoDir.appendingPathComponent(ModelNames.MossTtsNano.configFile))
        } catch {
            throw MossTtsNanoError.configLoadFailed("\(error)")
        }
        loadedTokenizer = try MossTtsNanoTokenizer(
            modelURL: repoDir.appendingPathComponent(ModelNames.MossTtsNano.tokenizerFile))

        logger.info("Loading MOSS-TTS-Nano CoreML models from \(repoDir.path)…")
        let start = Date()
        let lmConfig = MLModelConfiguration()
        lmConfig.computeUnits = lmComputeUnits
        let anyConfig = MLModelConfiguration()
        anyConfig.computeUnits = computeUnits

        prefillModel = try load(repoDir: repoDir, fileName: ModelNames.MossTtsNano.prefillFile, config: lmConfig)
        stepModel = try load(repoDir: repoDir, fileName: ModelNames.MossTtsNano.stepFile, config: lmConfig)
        frameModel = try load(repoDir: repoDir, fileName: ModelNames.MossTtsNano.frameFile, config: anyConfig)
        codecStepModel = try load(repoDir: repoDir, fileName: ModelNames.MossTtsNano.codecStepFile, config: anyConfig)
        logger.info("MOSS-TTS-Nano models loaded in \(String(format: "%.2f", Date().timeIntervalSince(start)))s")
    }

    // MARK: - Accessors

    public func prefill() throws -> MLModel { try unwrap(prefillModel) }
    public func step() throws -> MLModel { try unwrap(stepModel) }
    public func frame() throws -> MLModel { try unwrap(frameModel) }
    public func codecStep() throws -> MLModel { try unwrap(codecStepModel) }
    public func config() throws -> MossTtsNanoConfig { try unwrap(loadedConfig) }
    public func tokenizer() throws -> MossTtsNanoTokenizer { try unwrap(loadedTokenizer) }

    /// fp32 codec encoder, downloaded and loaded on first call.
    public func encoder() async throws -> MLModel {
        if let encoderModel { return encoderModel }
        let url = try await MossTtsNanoResourceDownloader.ensureEncoder(directory: directory)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = lmComputeUnits
        let model = try load(repoDir: url.deletingLastPathComponent(), fileName: url.lastPathComponent, config: cfg)
        encoderModel = model
        return model
    }

    public func unload() {
        prefillModel = nil
        stepModel = nil
        frameModel = nil
        codecStepModel = nil
        encoderModel = nil
    }

    // MARK: - Helpers

    private func unwrap<T>(_ value: T?) throws -> T {
        guard let value else { throw MossTtsNanoError.notInitialized }
        return value
    }

    private func load(repoDir: URL, fileName: String, config: MLModelConfiguration) throws -> MLModel {
        let url = repoDir.appendingPathComponent(fileName)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MossTtsNanoError.modelFileNotFound(fileName)
        }
        do {
            let model = try MLModel(contentsOf: url, configuration: config)
            logger.info("Loaded \(fileName)")
            return model
        } catch {
            throw MossTtsNanoError.corruptedModel(fileName, underlying: "\(error)")
        }
    }
}
