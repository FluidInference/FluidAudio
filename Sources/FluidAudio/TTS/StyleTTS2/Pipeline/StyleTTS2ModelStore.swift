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
        // Drop any stragglers from a previous failed bring-up so the state
        // machine is "fully loaded or empty" rather than "fully loaded,
        // empty, or some-arbitrary-subset". Without this guard a mid-load
        // failure would leave `models` populated with whatever succeeded;
        // subsequent `model(for:)` calls would happily hand those entries
        // back and only fail on the missing stages, masking the original
        // error and producing confusing partial-pipeline runs.
        models.removeAll()
        loaded = false

        let bundleRoot =
            try await StyleTTS2CoreMLDownloader
            .ensureAssets(directory: directory, progressHandler: progressHandler)

        // Load each stage concurrently. ANE compilation funnels through
        // `anecompilerservice` (a system singleton), so the speedup is
        // bounded by what that service can pipeline; the CPU-side
        // `.mlmodelc` unpack does parallelize cleanly. The first stage
        // throw cancels the rest via TaskGroup teardown.
        //
        // Accumulator pattern (mirrors
        // `KokoroAneModelStore.loadIfNeeded`): stage models land in
        // `pending` first; we only commit to `self.models` after the
        // cancellation check below passes — a partial failure or a
        // racing `cleanup()` therefore can never leave the store
        // half-loaded.
        let units = computeUnits
        var pending: [StyleTTS2Stage: MLModel] = [:]
        try await withThrowingTaskGroup(
            of: (StyleTTS2Stage, MLModel).self
        ) { group in
            for stage in StyleTTS2Stage.allCases {
                group.addTask {
                    let url = bundleRoot.appendingPathComponent(stage.bundleName)
                    guard FileManager.default.fileExists(atPath: url.path) else {
                        throw StyleTTS2Error.modelNotLoaded(stage.bundleName)
                    }
                    let config = MLModelConfiguration()
                    config.computeUnits = units.units(for: stage)
                    do {
                        let model = try MLModel(contentsOf: url, configuration: config)
                        return (stage, model)
                    } catch {
                        throw StyleTTS2Error.loadFailed(
                            stage: stage.rawValue, underlying: error)
                    }
                }
            }
            for try await (stage, model) in group {
                pending[stage] = model
            }
        }

        // `cleanup()` cancels this task. Honour that here so a cleanup
        // call that races a successful bring-up doesn't have its state
        // wipe immediately overwritten by the late commit below.
        try Task.checkCancellation()

        models = pending
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
        // Cancel any in-flight bring-up. The load body calls
        // `Task.checkCancellation()` before committing the loaded
        // models dict, so cleanup() winning the actor turn keeps the
        // store empty without needing to await the task's settlement.
        loadTask?.cancel()
        loadTask = nil
        models.removeAll()
        loaded = false
    }
}
