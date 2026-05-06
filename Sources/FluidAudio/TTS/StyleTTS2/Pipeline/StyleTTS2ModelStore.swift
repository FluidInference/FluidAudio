@preconcurrency import CoreML
import Foundation

/// Per-stage compute-unit assignment for the StyleTTS2-ANE 7-stage chain.
///
/// Mirrors `KokoroAneComputeUnits` exactly except `.tail` is folded into
/// `.vocoder` (StyleTTS2's HiFi-GAN is iSTFT-free) and a new
/// `.diffusionStep` is added for the StyleTTS2-only diffusion sampler.
///
/// Empirical defaults match the conversion script's compute-unit hints:
///   - PLBert / PostBert / Alignment / DiffusionStep / Prosody / Vocoder
///     → `.cpuAndNeuralEngine`
///   - Noise → `.all` (fp32 phase precision; ANE may decline silently)
public struct StyleTTS2ComputeUnits: Sendable, Equatable {
    public var plBert: MLComputeUnits
    public var postBert: MLComputeUnits
    public var alignment: MLComputeUnits
    public var diffusionStep: MLComputeUnits
    public var prosody: MLComputeUnits
    public var noise: MLComputeUnits
    public var vocoder: MLComputeUnits

    public init(
        plBert: MLComputeUnits = .cpuAndNeuralEngine,
        postBert: MLComputeUnits = .cpuAndNeuralEngine,
        alignment: MLComputeUnits = .cpuAndNeuralEngine,
        diffusionStep: MLComputeUnits = .cpuAndNeuralEngine,
        prosody: MLComputeUnits = .cpuAndNeuralEngine,
        noise: MLComputeUnits = .all,
        vocoder: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.plBert = plBert
        self.postBert = postBert
        self.alignment = alignment
        self.diffusionStep = diffusionStep
        self.prosody = prosody
        self.noise = noise
        self.vocoder = vocoder
    }

    /// Empirical default — see file header for rationale.
    public static let `default` = StyleTTS2ComputeUnits()

    /// CPU+GPU only (skip ANE entirely). Useful for a baseline.
    public static let cpuAndGpu = StyleTTS2ComputeUnits(
        plBert: .cpuAndGPU, postBert: .cpuAndGPU, alignment: .cpuAndGPU,
        diffusionStep: .cpuAndGPU, prosody: .cpuAndGPU,
        noise: .cpuAndGPU, vocoder: .cpuAndGPU
    )

    /// Force every stage onto `.cpuAndNeuralEngine`. Used by the
    /// `tts-benchmark` ANE-vs-CPU sweep.
    public static let allAne = StyleTTS2ComputeUnits(
        plBert: .cpuAndNeuralEngine, postBert: .cpuAndNeuralEngine,
        alignment: .cpuAndNeuralEngine, diffusionStep: .cpuAndNeuralEngine,
        prosody: .cpuAndNeuralEngine, noise: .cpuAndNeuralEngine,
        vocoder: .cpuAndNeuralEngine
    )

    /// CPU-only — slowest but most predictable; debugging baseline.
    public static let cpuOnly = StyleTTS2ComputeUnits(
        plBert: .cpuOnly, postBert: .cpuOnly, alignment: .cpuOnly,
        diffusionStep: .cpuOnly, prosody: .cpuOnly,
        noise: .cpuOnly, vocoder: .cpuOnly
    )

    public init(preset: TtsComputeUnitPreset) {
        switch preset {
        case .default: self = .default
        case .allAne: self = .allAne
        case .cpuAndGpu: self = .cpuAndGpu
        case .cpuOnly: self = .cpuOnly
        }
    }

    func units(for stage: StyleTTS2Stage) -> MLComputeUnits {
        switch stage {
        case .plBert: return plBert
        case .postBert: return postBert
        case .alignment: return alignment
        case .diffusionStep: return diffusionStep
        case .prosody: return prosody
        case .noise: return noise
        case .vocoder: return vocoder
        }
    }
}

/// Resident store for the 7 StyleTTS2-ANE `.mlmodelc` bundles.
///
/// Lazily loads each model on first call; afterwards keeps strong refs
/// so the actor-isolated synthesizer can issue back-to-back predictions
/// without re-paying the (multi-second) ANE compile cost.
///
/// Mirrors `KokoroAneModelStore` 1:1 except for the seven stages it
/// holds.
public actor StyleTTS2ModelStore {

    private let logger = AppLogger(category: "StyleTTS2ModelStore")
    private let directory: URL?
    private let computeUnits: StyleTTS2ComputeUnits

    private var models: [StyleTTS2Stage: MLModel] = [:]
    private var loaded: Bool = false
    private var loadTask: Task<Void, Error>?

    public init(
        directory: URL? = nil,
        computeUnits: StyleTTS2ComputeUnits = .default
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
    }

    public var isLoaded: Bool { loaded }

    /// Resolve or download the bundle, then load every `.mlmodelc`.
    /// Idempotent: subsequent calls are no-ops. Concurrent first-callers
    /// join a shared `Task` so the download + multi-second ANE compile is
    /// paid exactly once. Only the first call's `progressHandler` fires.
    public func loadIfNeeded(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        if loaded { return }
        if let task = loadTask {
            try await task.value
            return
        }
        let task = Task<Void, Error> { [self] in
            try await loadAllStages(progressHandler: progressHandler)
        }
        loadTask = task
        do {
            try await task.value
            loadTask = nil
        } catch {
            loadTask = nil
            throw error
        }
    }

    private func loadAllStages(
        progressHandler: DownloadUtils.ProgressHandler?
    ) async throws {
        let bundleRoot =
            try await StyleTTS2CoreMLDownloader
            .ensureAssets(directory: directory, progressHandler: progressHandler)

        for stage in StyleTTS2Stage.allCases {
            let url = bundleRoot.appendingPathComponent(stage.bundleName)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw StyleTTS2Error.modelNotLoaded(stage.bundleName)
            }
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits.units(for: stage)
            do {
                let model = try MLModel(contentsOf: url, configuration: config)
                models[stage] = model
            } catch {
                throw StyleTTS2Error.predictionFailed(
                    stage: stage.rawValue, underlying: error)
            }
        }
        loaded = true
        logger.notice("StyleTTS2-ANE: 7 mlmodelcs loaded.")
    }

    public func model(for stage: StyleTTS2Stage) throws -> MLModel {
        guard let model = models[stage] else {
            throw StyleTTS2Error.modelNotLoaded(stage.bundleName)
        }
        return model
    }

    public func cleanup() {
        models.removeAll()
        loaded = false
    }
}
