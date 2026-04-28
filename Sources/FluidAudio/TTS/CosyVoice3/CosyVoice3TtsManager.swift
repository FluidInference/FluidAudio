@preconcurrency import CoreML
import Foundation

/// Public entry point for the CosyVoice3 (Mandarin) TTS pipeline.
///
/// > Important: **Experimental / beta.** This backend ships as an early port
/// > and end-to-end synthesis is currently **slow** on Apple Silicon —
/// > expect well below real-time (RTFx < 1.0) on M-series GPUs and several
/// > seconds of latency for short Mandarin utterances. The slowdown is
/// > primarily in the Flow CFM stage, which is fp32/CPU-or-GPU only because
/// > fp16 + ANE produces NaNs through the fused `layer_norm` (CoreMLTools
/// > limitation; tracked upstream). The HiFT vocoder also has ~12 sinegen /
/// > windowing ops that fall back to CPU. We do not yet know whether the
/// > residual cost is fundamental to the model or recoverable through better
/// > conversion — treat performance numbers as preliminary. The Swift API,
/// > model layout, and prompt-asset format may change in subsequent
/// > releases without deprecation aliases.
///
/// Two synthesis paths are exposed:
///
/// 1. `synthesizeFromFixture` — Phase 1 parity harness that replays a
///    Python-generated fixture against the Swift CoreML pipeline.
///
/// 2. `synthesize(text:promptAssets:)` — Phase 2 text-driven synthesis. The
///    user supplies a Mandarin `text` plus a `CosyVoice3PromptAssets` bundle
///    (precomputed `llm_prompt_speech_ids`, `prompt_mel`, `spk_embedding`,
///    plus the prompt text containing `<|endofprompt|>`). The manager
///    tokenizes with the on-device Qwen2 BPE tokenizer, assembles
///    `lm_input_embeds` from the mmap'd runtime embedding tables, and runs
///    prefill → decode → Flow → HiFT exactly like the fixture path.
///
/// Text-mode requires three extra resources that must be provided at init:
/// - `tokenizerDirectory`: HuggingFace Qwen2 assets (`vocab.json` + `merges.txt`).
/// - `textEmbeddingsFile`: `embeddings-runtime-fp32.safetensors` produced by
///   `mobius/.../verify/export_runtime_embeddings.py`. Contains Qwen2
///   `text_embedding` and CosyVoice3 `speech_embedding` rows at runtime dtype.
/// - `specialTokensFile`: JSON map `{"<|endofprompt|>": 151646, ...}` covering
///   the 281 runtime-added special tokens (CosyVoice3Tokenizer). Same format
///   that `tokenizer_fixture.json` dumps under its `special_tokens` key.
///
/// > Note: Gated to macOS 15 / iOS 18 because the underlying
/// > `CosyVoice3Synthesizer` uses CoreML `MLState` for the decode KV cache.
/// > Other FluidAudio modules (ASR, Diarization, VAD, Kokoro, PocketTTS)
/// > remain available on macOS 14 / iOS 17.
@available(macOS 15, iOS 18, *)
public actor CosyVoice3TtsManager {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "CosyVoice3TtsManager")

    private let store: CosyVoice3ModelStore
    private let tokenizerDirectory: URL?
    private let textEmbeddingsFile: URL?
    private let specialTokensFile: URL?

    private var synthesizer: CosyVoice3Synthesizer?
    private var textFrontend: CosyVoice3TextFrontend?

    /// Fixture-only (Phase 1) constructor.
    public init(directory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) {
        self.store = CosyVoice3ModelStore(directory: directory, computeUnits: computeUnits)
        self.tokenizerDirectory = nil
        self.textEmbeddingsFile = nil
        self.specialTokensFile = nil
    }

    /// Text-mode (Phase 2) constructor. Pass `modelsDirectory` plus the three
    /// tokenizer-frontend resources. `synthesizeFromFixture` still works
    /// without initializing the frontend.
    public init(
        modelsDirectory: URL,
        tokenizerDirectory: URL,
        textEmbeddingsFile: URL,
        specialTokensFile: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.store = CosyVoice3ModelStore(directory: modelsDirectory, computeUnits: computeUnits)
        self.tokenizerDirectory = tokenizerDirectory
        self.textEmbeddingsFile = textEmbeddingsFile
        self.specialTokensFile = specialTokensFile
    }

    /// Convenience factory that downloads all required assets from HuggingFace
    /// (`FluidInference/CosyVoice3-0.5B-coreml`) into the shared FluidAudio
    /// cache, then returns a text-mode–ready manager.
    ///
    /// - Parameters:
    ///   - cacheDirectory: Optional override for the base cache root. When
    ///     `nil`, uses `~/.cache/fluidaudio` (macOS) or the app Caches dir
    ///     (iOS) — the same location every other FluidAudio TTS backend uses.
    ///   - includeDefaultVoice: When `true` (default), also fetches the
    ///     upstream `cosyvoice3-default-zh` voice bundle so the first
    ///     `synthesize(...)` call works without any additional downloads.
    ///   - computeUnits: CoreML compute units for LLM + HiFT. Flow is forced
    ///     to CPU+GPU regardless (fp32 graph, ANE would NaN on fused LN).
    ///   - progressHandler: Forwarded to the HF downloader for UI updates.
    /// - Returns: An uninitialized manager; the caller must still invoke
    ///   `initialize()` to compile + load models. A download of ~5.8 GB occurs
    ///   on first run; subsequent runs are cache hits.
    public static func downloadAndCreate(
        cacheDirectory: URL? = nil,
        includeDefaultVoice: Bool = true,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> CosyVoice3TtsManager {
        let repoDir = try await CosyVoice3ResourceDownloader.ensureCoreModels(
            directory: cacheDirectory, progressHandler: progressHandler)
        let frontend = try await CosyVoice3ResourceDownloader.ensureTextFrontendAssets(
            repoDirectory: repoDir)
        if includeDefaultVoice {
            _ = try await CosyVoice3ResourceDownloader.ensureVoice(
                repoDirectory: repoDir)
        }
        return CosyVoice3TtsManager(
            modelsDirectory: repoDir,
            tokenizerDirectory: frontend.tokenizerDirectory,
            textEmbeddingsFile: frontend.runtimeEmbeddingsFile,
            specialTokensFile: frontend.specialTokensFile,
            computeUnits: computeUnits)
    }

    /// Ensure the given voice id (e.g. `"cosyvoice3-default-zh"` or an
    /// `aishell3-zh-SSB####-{female|male}` id) is cached locally, and return
    /// the loaded prompt bundle ready to pass into `synthesize(text:promptAssets:)`.
    public func loadVoice(
        _ voiceId: String = ModelNames.CosyVoice3.Sidecar.defaultVoiceId
    ) async throws -> CosyVoice3PromptAssets {
        let tensorsURL = try await CosyVoice3ResourceDownloader.ensureVoice(
            voiceId: voiceId,
            repoDirectory: modelsDirectory)
        return try CosyVoice3PromptAssets.load(from: tensorsURL)
    }

    /// Repo root directory (cache location after `downloadAndCreate(...)`).
    /// Pass this to `CosyVoice3ResourceDownloader.ensureVoice(voiceId:repoDirectory:)`
    /// when fetching additional voice bundles on demand.
    public nonisolated var modelsDirectory: URL {
        store.directory
    }

    /// Load all four CoreML models + (if configured) the text frontend.
    /// Idempotent.
    public func initialize() async throws {
        if synthesizer == nil {
            logger.warning(
                "CosyVoice3 is experimental / beta. Synthesis is currently slow "
                    + "(RTFx < 1.0 typical) — see CosyVoice3TtsManager docs.")
            try await store.loadIfNeeded()
            let models = try await store.models()
            let embeddingsURL = try await store.speechEmbeddingsFileURL()
            let embeddings = try CosyVoice3SpeechEmbeddings(url: embeddingsURL)
            self.synthesizer = CosyVoice3Synthesizer(models: models, embeddings: embeddings)
            logger.info("CosyVoice3 synthesizer ready")
        }
        if textFrontend == nil,
            let tokDir = tokenizerDirectory,
            let embURL = textEmbeddingsFile,
            let specURL = specialTokensFile
        {
            let tokStart = Date()
            let specialTokens = try Self.loadSpecialTokens(url: specURL)
            let tokenizer = try Qwen2BpeTokenizer.load(
                directory: tokDir, specialTokens: specialTokens)
            let textEmbeddings = try CosyVoice3TextEmbeddings(url: embURL)
            self.textFrontend = CosyVoice3TextFrontend(
                tokenizer: tokenizer, embeddings: textEmbeddings)
            logger.info(
                "CosyVoice3 text frontend ready in \(String(format: "%.2fs", Date().timeIntervalSince(tokStart)))"
            )
        }
    }

    /// Phase 1 parity entry point.
    public func synthesizeFromFixture(
        fixtureURL: URL,
        options: CosyVoice3ParityOptions = CosyVoice3ParityOptions()
    ) async throws -> CosyVoice3SynthesisResult {
        guard let synthesizer = synthesizer else {
            throw CosyVoice3Error.notInitialized
        }
        let fixture = try CosyVoice3FrontendFixture.load(from: fixtureURL)
        return try await synthesizer.synthesize(fixture: fixture, options: options)
    }

    /// Phase 2 text-driven synthesis.
    ///
    /// - Parameters:
    ///   - text: Mandarin (or mixed) input text.
    ///   - promptAssets: Bundle with prompt text + precomputed speech prompt
    ///     tokens + prompt mel + speaker embedding.
    ///   - options: Sampling / seed controls. `replayDecodedTokens` must be
    ///     `false` in text mode (the default here).
    ///   - prenormalized: When `true`, skip the built-in minimal Chinese
    ///     normalizer and feed `text` straight to the tokenizer. Set this if
    ///     you've already run wetext (or equivalent) server-side.
    public func synthesize(
        text: String,
        promptAssets: CosyVoice3PromptAssets,
        options: CosyVoice3SynthesisOptions = CosyVoice3SynthesisOptions(),
        prenormalized: Bool = false
    ) async throws -> CosyVoice3SynthesisResult {
        guard let synthesizer = synthesizer else {
            throw CosyVoice3Error.notInitialized
        }
        guard let frontend = textFrontend else {
            throw CosyVoice3Error.notInitialized
        }

        // Skip normalization if the caller set `prenormalized`, if the input
        // contains SSML-ish markers (mirrors Python's `'<|' in text and '|>'`
        // bypass), or if there are no CJK characters at all.
        let ssmlLike = text.contains("<|") && text.contains("|>")
        let normalized: String
        if prenormalized || ssmlLike || !CosyVoice3ChineseNormalizer.containsChinese(text) {
            normalized = text
        } else {
            normalized = CosyVoice3ChineseNormalizer.normalize(text)
        }

        let assembled = try frontend.assemble(
            promptText: promptAssets.promptText,
            ttsText: normalized,
            promptSpeechIds: promptAssets.promptSpeechIds)

        let lmInputEmbedsFlat = try Self.flattenLmEmbeds(
            assembled.lmInputEmbeds, tPre: assembled.tPre)

        // Build an in-memory fixture adapter so we can reuse the Phase 1
        // synthesize(fixture:) path without a second code path.
        let fixture = CosyVoice3FrontendFixture(
            lmInputEmbeds: lmInputEmbedsFlat,
            tPre: assembled.tPre,
            promptSpeechIds: promptAssets.promptSpeechIds,
            promptMel: promptAssets.promptMel,
            promptMelFrames: promptAssets.promptMelFrames,
            spkEmbedding: promptAssets.spkEmbedding,
            decodedTokens: [],
            seed: Int32(truncatingIfNeeded: options.seed),
            numPromptMel: 0,
            audioLengthSamples: 0)

        let parityOptions = CosyVoice3ParityOptions(
            maxNewTokens: options.maxNewTokens,
            seed: options.seed,
            replayDecodedTokens: false)

        return try await synthesizer.synthesize(fixture: fixture, options: parityOptions)
    }

    // MARK: - Helpers

    /// Flatten `[1, tPre, 896]` MLMultiArray fp32 into `[tPre * 896]` Float,
    /// honoring non-compact strides.
    private static func flattenLmEmbeds(
        _ array: MLMultiArray, tPre: Int
    ) throws -> [Float] {
        guard
            array.dataType == .float32,
            array.shape.count == 3,
            array.shape[0].intValue == 1,
            array.shape[1].intValue == tPre,
            array.shape[2].intValue == CosyVoice3Constants.embedDim
        else {
            throw CosyVoice3Error.invalidShape(
                "lmInputEmbeds expects [1, \(tPre), \(CosyVoice3Constants.embedDim)] fp32, got shape=\(array.shape) dtype=\(array.dataType.rawValue)"
            )
        }
        let dim = CosyVoice3Constants.embedDim
        let strides = array.strides.map { $0.intValue }
        let src = array.dataPointer.bindMemory(to: Float.self, capacity: array.count)
        var out = [Float](repeating: 0, count: tPre * dim)
        out.withUnsafeMutableBufferPointer { dst in
            for t in 0..<tPre {
                let srcRow = src.advanced(by: t * strides[1])
                let dstRow = dst.baseAddress!.advanced(by: t * dim)
                if strides[2] == 1 {
                    memcpy(dstRow, srcRow, dim * MemoryLayout<Float>.size)
                } else {
                    for d in 0..<dim { dstRow[d] = srcRow[d * strides[2]] }
                }
            }
        }
        return out
    }

    private static func loadSpecialTokens(url: URL) throws -> [String: Int32] {
        let data = try Data(contentsOf: url)
        // Accept either the tokenizer_fixture.json shape
        // ({"special_tokens": {...}, "cases": [...]}) or a flat map.
        let json = try JSONSerialization.jsonObject(with: data)
        let raw: [String: Any]
        if let obj = json as? [String: Any], let nested = obj["special_tokens"] as? [String: Any] {
            raw = nested
        } else if let obj = json as? [String: Any] {
            raw = obj
        } else {
            throw CosyVoice3Error.invalidShape(
                "special tokens file must be a JSON object, got \(type(of: json))")
        }
        var out: [String: Int32] = [:]
        out.reserveCapacity(raw.count)
        for (k, v) in raw {
            if let n = v as? Int {
                out[k] = Int32(n)
            } else if let n = v as? NSNumber {
                out[k] = n.int32Value
            }
        }
        guard !out.isEmpty else {
            throw CosyVoice3Error.invalidShape(
                "special tokens file parsed to an empty map at \(url.path)")
        }
        return out
    }
}
