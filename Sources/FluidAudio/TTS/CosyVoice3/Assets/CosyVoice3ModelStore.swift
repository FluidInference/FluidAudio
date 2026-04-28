@preconcurrency import CoreML
import Foundation

/// Actor-based store for the four CosyVoice3 CoreML models.
///
/// Two on-disk layouts are accepted:
///
/// 1. **HuggingFace cache** (flat): `<dir>/<ModelName>.mlmodelc` (or
///    `.mlpackage`) at repo root, with `<dir>/embeddings/speech_embedding-fp16.safetensors`.
///    This is what `CosyVoice3ResourceDownloader` produces.
///
/// 2. **Local mobius build dir**: `<dir>/<subdir>/<ModelName>.mlpackage` as
///    emitted by `models/tts/cosyvoice3/coreml/convert-coreml.py` (with
///    `llm-fp16/`, `flow-fp16-n250/`, `hift-fp16-t500/` subdirs).
///
/// The store probes layout (1) first, then falls back to (2). CoreML
/// auto-compiles `.mlpackage` on first load and caches the compiled bundle on
/// disk.
public actor CosyVoice3ModelStore {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "CosyVoice3ModelStore")

    public nonisolated let directory: URL
    private let computeUnits: MLComputeUnits

    private var loadedModels: CosyVoice3Models?
    private var speechEmbeddingsURL: URL?

    /// - Parameters:
    ///   - directory: Base build directory that contains
    ///     `llm-fp16/`, `llm-fp16-stateful/`, `flow-fp16-n250/`,
    ///     `hift-fp16-t500/`, `embeddings/`.
    ///   - computeUnits: Defaults to `.cpuAndNeuralEngine`. Applied to
    ///     LLM-Prefill + HiFT models only. LLM-Decode (stateful) and Flow
    ///     both force `.cpuAndGPU` regardless (see `loadIfNeeded()`).
    public init(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
        self.directory = directory
        self.computeUnits = computeUnits
    }

    /// Load all four CoreML models. Idempotent.
    public func loadIfNeeded() async throws {
        guard loadedModels == nil else { return }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        let loadStart = Date()
        logger.info("Loading CosyVoice3 CoreML models from \(directory.path)...")

        let prefillURL = try resolveModel(
            subdir: CosyVoice3Constants.Files.llmPrefillSubdir,
            baseName: ModelNames.CosyVoice3.llmPrefill)
        let decodeURL = try resolveModel(
            subdir: CosyVoice3Constants.Files.llmDecodeSubdir,
            baseName: ModelNames.CosyVoice3.llmDecode)
        let flowURL = try resolveModel(
            subdir: CosyVoice3Constants.Files.flowSubdir,
            baseName: ModelNames.CosyVoice3.flow)
        let hiftURL = try resolveModel(
            subdir: CosyVoice3Constants.Files.hiftSubdir,
            baseName: ModelNames.CosyVoice3.hift)
        let embeddingsURL = try resolveAsset(
            subdir: CosyVoice3Constants.Files.speechEmbeddingsSubdir,
            file: CosyVoice3Constants.Files.speechEmbeddings)

        let prefill = try await compileAndLoad(prefillURL, configuration: config)
        logger.info("Loaded \(CosyVoice3Constants.Files.llmPrefill)")

        // Stateful decode MUST run on `.cpuAndGPU`:
        //   - ANE refuses to compile the stateful graph (same failure mode
        //     as Flow: `MILCompilerForANE ANECCompile() FAILED`), so
        //     `.cpuAndNE` / `.all` deadlock load
        //   - CPU-only works but is ~2× slower than the GPU path
        // Ignore the user-supplied `computeUnits` for decode.
        let decodeConfig = MLModelConfiguration()
        decodeConfig.computeUnits = .cpuAndGPU
        let decode = try await compileAndLoad(decodeURL, configuration: decodeConfig)
        logger.info("Loaded \(CosyVoice3Constants.Files.llmDecode)")

        // Flow runs on `.cpuAndGPU` (fp16). An ANE-port attempt (BC1S
        // rewrite: Linear→Conv2d(1×1), LayerNorm on axis=1, manual SDPA,
        // pre-baked rotary sin/cos) produced a Flow that *compiled* and
        // ran ~3× faster, but numerically broken: on the parity
        // fixture the ANE graph collapses the mel dynamic range from
        // [-12.5, +5.2] to [-10.1, -0.8] (MAE 2.58 vs PyTorch fp32;
        // plan required <1e-3), yielding HiFT audio at ~40× lower peak
        // amplitude — unintelligible to both CTC-ZH and Qwen3 ASR.
        // Reverted to the cpuAndGPU fp16 baseline. See
        // `coreml/TRIALS_AND_ERRORS.md` "Flow ANE port" for the full
        // journey including the residual 77-op `conv_pos_embed` CPU
        // island that may have been masking the dynamic-range
        // compression introduced elsewhere in the BC1S rewrite.
        // Ignore the user-supplied `computeUnits` for Flow; apply it to
        // the LLM + HiFT models only.
        let flowConfig = MLModelConfiguration()
        flowConfig.computeUnits = .cpuAndGPU
        let flow = try await compileAndLoad(flowURL, configuration: flowConfig)
        logger.info("Loaded \(CosyVoice3Constants.Files.flow)")

        let hift = try await compileAndLoad(hiftURL, configuration: config)
        logger.info("Loaded \(CosyVoice3Constants.Files.hift)")

        loadedModels = CosyVoice3Models(prefill: prefill, decode: decode, flow: flow, hift: hift)
        speechEmbeddingsURL = embeddingsURL

        let elapsed = Date().timeIntervalSince(loadStart)
        logger.info("All CosyVoice3 models loaded in \(String(format: "%.2f", elapsed))s")
    }

    public func models() throws -> CosyVoice3Models {
        guard let models = loadedModels else {
            throw CosyVoice3Error.notInitialized
        }
        return models
    }

    public func speechEmbeddingsFileURL() throws -> URL {
        guard let url = speechEmbeddingsURL else {
            throw CosyVoice3Error.notInitialized
        }
        return url
    }

    // MARK: - Helpers

    /// Resolve a CoreML model accepting either `.mlmodelc` or `.mlpackage`
    /// extensions and both layouts: flat (HF) or subdir (local build).
    private func resolveModel(subdir: String, baseName: String) throws -> URL {
        let candidates: [URL] = [
            // HF flat layout prefers the precompiled .mlmodelc.
            directory.appendingPathComponent("\(baseName).mlmodelc"),
            directory.appendingPathComponent("\(baseName).mlpackage"),
            // Local build layout (mobius convert-coreml.py output).
            directory.appendingPathComponent(subdir).appendingPathComponent("\(baseName).mlmodelc"),
            directory.appendingPathComponent(subdir).appendingPathComponent("\(baseName).mlpackage"),
        ]
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        let probed = candidates.map { $0.path }.joined(separator: ", ")
        throw CosyVoice3Error.modelFileNotFound(probed)
    }

    /// Resolve a plain sidecar file (e.g. `speech_embedding-fp16.safetensors`).
    /// Probes `<dir>/<subdir>/<file>` then `<dir>/<file>`.
    private func resolveAsset(subdir: String, file: String) throws -> URL {
        let candidates: [URL] = [
            directory.appendingPathComponent(subdir).appendingPathComponent(file),
            directory.appendingPathComponent(file),
        ]
        for url in candidates where FileManager.default.fileExists(atPath: url.path) {
            return url
        }
        let probed = candidates.map { $0.path }.joined(separator: ", ")
        throw CosyVoice3Error.modelFileNotFound(probed)
    }

    /// Compile an .mlpackage to .mlmodelc (cached in a persistent temp dir
    /// next to the original package) and load it. Skips compilation if an
    /// already-compiled .mlmodelc exists next to the package.
    private func compileAndLoad(
        _ url: URL,
        configuration: MLModelConfiguration
    ) async throws -> MLModel {
        if url.pathExtension == "mlmodelc" {
            return try MLModel(contentsOf: url, configuration: configuration)
        }
        let base = url.deletingPathExtension().lastPathComponent
        let compiledName = base + ".mlmodelc"
        let cached = url.deletingLastPathComponent().appendingPathComponent(compiledName)
        if FileManager.default.fileExists(atPath: cached.path) {
            return try MLModel(contentsOf: cached, configuration: configuration)
        }
        let compiledURL = try await MLModel.compileModel(at: url)
        // Move into place next to the package so subsequent loads are fast.
        try? FileManager.default.removeItem(at: cached)
        do {
            try FileManager.default.moveItem(at: compiledURL, to: cached)
            return try MLModel(contentsOf: cached, configuration: configuration)
        } catch {
            // If the move fails (e.g. cross-device), load from the temp URL.
            return try MLModel(contentsOf: compiledURL, configuration: configuration)
        }
    }
}
