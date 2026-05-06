import Foundation

/// Manages text-to-speech synthesis using the StyleTTS2-ANE 7-stage CoreML
/// chain. Mirrors `KokoroAneManager` 1:1 — public surface, error handling,
/// G2P bootstrap, voice loading, and timing reporting are all identical so
/// that downstream callers can switch backends with a single line change.
///
/// Pipeline (per utterance):
///   1. `plbert.mlmodelc`         — RangeDim(2..512), fp16, ANE
///   2. `postbert.mlmodelc`       — RangeDim(2..512), fp16, ANE  (BiLSTM-unrolled)
///   3. `alignment.mlmodelc`      — RangeDim, fp16, ANE  (cumsum+broadcast hard alignment)
///   4. `diffusion_step.mlmodelc` — STATIC shapes, fp16, ANE  (ADPM2 sampler in Swift,
///                                  invoked 11× = 5 midpoint steps × 2 + 1 final)
///   5. `prosody.mlmodelc`        — STATIC T_a=2000, fp16, ANE
///   6. `noise.mlmodelc`          — STATIC T_a=2000, fp32, ALL  (SineGen phase precision)
///   7. `vocoder.mlmodelc`        — STATIC T_a=2000, fp16, ANE  (cos-Snake patched HiFi-GAN)
///
/// The Swift host is responsible for:
///   - misaki / espeak-ng IPA phonemization + vocab lookup (reused from
///     the legacy `StyleTTS2/` module — same 178-token vocab),
///   - the ADPM2 + Karras sampler loop around `diffusion_step` (also reused
///     from the legacy module via `StyleTTS2Sampler`).
public actor StyleTTS2Manager {

    private let logger = AppLogger(category: "StyleTTS2Manager")
    private let modelStore: StyleTTS2ModelStore
    private let directory: URL?
    private let computeUnits: StyleTTS2ComputeUnits

    /// Lazily-loaded shared vocab + bundle config from the legacy 4-graph
    /// repo. The ANE re-cut deliberately reuses these instead of duplicating
    /// the JSON because they are identical across both checkpoints (same
    /// LibriTTS espeak-ng IPA). Reusing also means an existing user who
    /// has already downloaded `StyleTTS-2-coreml/` doesn't pay the asset
    /// cost again.
    private let legacyAssetStore: StyleTTS2AssetStore

    private var isInitialized = false
    private var initializeTask: Task<Void, Error>?

    /// - Parameters:
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    ///   - computeUnits: Per-stage compute-unit assignment. Defaults match
    ///     the empirical sweep documented in `StyleTTS2ComputeUnits`.
    public init(
        directory: URL? = nil,
        computeUnits: StyleTTS2ComputeUnits = .default
    ) {
        self.directory = directory
        self.computeUnits = computeUnits
        self.modelStore = StyleTTS2ModelStore(
            directory: directory, computeUnits: computeUnits)
        self.legacyAssetStore = StyleTTS2AssetStore(directory: directory)
    }

    /// Public-API alias for `isInitialized`. Both names exist intentionally:
    /// `isAvailable` reads naturally at call sites that gate UI / feature
    /// availability (`if await manager.isAvailable { ... }`), while the
    /// internal `isInitialized` flag mirrors the verb in `initialize()`
    /// and reads more naturally inside the implementation. They are always
    /// the same value — `isAvailable` is just a thin alias and intentionally
    /// not stored separately so they can never drift.
    public var isAvailable: Bool { isInitialized }

    /// Download the 7 mlmodelcs, eagerly compile them onto the requested
    /// compute units, and bootstrap the G2P + vocab assets. Idempotent.
    /// Concurrent first-callers join a shared `Task` so the multi-second
    /// download + ANE compile is paid exactly once. Only the first call's
    /// `progressHandler` fires; later joiners just await the same result.
    public func initialize(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        if isInitialized { return }
        if let task = initializeTask {
            try await task.value
            return
        }
        let task = Task<Void, Error> { [self] in
            try await runInitialize(progressHandler: progressHandler)
        }
        initializeTask = task
        do {
            try await task.value
            initializeTask = nil
        } catch {
            initializeTask = nil
            throw error
        }
    }

    private func runInitialize(
        progressHandler: DownloadUtils.ProgressHandler?
    ) async throws {
        // 1. Fetch + compile the 7 ANE mlmodelcs.
        try await modelStore.loadIfNeeded(progressHandler: progressHandler)

        // 2. Reuse the legacy 4-graph repo *only* for its vocab + config.
        //    `ensureAssetsAvailable` downloads the 12 mlmodelcs too, which
        //    is wasteful — but vocab/config aren't bundled separately on
        //    HF and the marginal disk cost is acceptable. A cleaner split
        //    is a follow-up (extract vocab.json into the ANE repo).
        _ = try await legacyAssetStore.ensureAssetsAvailable(
            progressHandler: progressHandler)

        // 3. The English G2P CoreML assets ship in the kokoro repo and are
        //    loaded from `~/.cache/fluidaudio/Models/kokoro/`. Mirror
        //    `KokoroAneManager.initialize` and fetch them explicitly so a
        //    first-time user who has never run kokoro doesn't hit a cryptic
        //    `G2PModelError.vocabLoadFailed` deep inside `synthesize`.
        //
        //    NOTE: pass nil (not `directory`) — `G2PModel.shared` is a
        //    singleton hardcoded to the default cache path; honouring a
        //    custom override here would write to a path the singleton can't
        //    read.
        try await KokoroAneResourceDownloader.ensureG2PAssets(
            directory: nil, progressHandler: progressHandler)
        try await G2PModel.shared.ensureModelsAvailable()

        isInitialized = true
        logger.notice("StyleTTS2Manager initialized")
    }

    /// The shipped LibriTTS checkpoint is English-only. Other
    /// `MultilingualG2PLanguage` cases would route through misaki/espeak-ng
    /// for phonemization but produce gibberish acoustically because the
    /// acoustic model has never seen those phoneme distributions. Reject
    /// them at the API boundary instead of silently producing wrong-language
    /// audio.
    private static func validateLanguage(_ language: MultilingualG2PLanguage) throws {
        switch language {
        case .americanEnglish, .britishEnglish:
            return
        default:
            throw StyleTTS2Error.invalidConfiguration(
                "StyleTTS2-ANE checkpoint is English-only; got \(language). "
                    + "Pass `.americanEnglish` or `.britishEnglish`."
            )
        }
    }

    /// Synthesize text to a WAV blob at 24 kHz.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize.
    ///   - voiceStyleURL: Path to a precomputed `ref_s.bin` (256 fp32 LE,
    ///     1024 bytes). Same blob format as the legacy 4-graph backend —
    ///     style-encoder export is a follow-up that hasn't shipped yet.
    ///   - language: G2P language for phonemization. The shipped LibriTTS
    ///     checkpoint is English-only.
    ///   - diffusionSteps: ADPM2 sampler iterations (default 5, matching
    ///     upstream Python).
    ///   - alpha: Acoustic style mix weight (default 0.3).
    ///   - beta: Prosody style mix weight (default 0.7).
    ///   - randomSeed: Seed for the diffusion noise RNG. `nil` → use the
    ///     system RNG (non-reproducible).
    /// - Returns: 24 kHz mono 16-bit PCM WAV data.
    public func synthesize(
        text: String,
        voiceStyleURL: URL,
        language: MultilingualG2PLanguage = .americanEnglish,
        diffusionSteps: Int = StyleTTS2Constants.defaultDiffusionSteps,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        randomSeed: UInt64? = nil
    ) async throws -> Data {
        let samples = try await synthesizeSamples(
            text: text,
            voiceStyleURL: voiceStyleURL,
            language: language,
            diffusionSteps: diffusionSteps,
            alpha: alpha,
            beta: beta,
            randomSeed: randomSeed
        ).samples
        // Scale to a target peak of -7 dBFS to match the upstream PyTorch
        // and Python CoreML reference loudness. Without this, the raw
        // HiFi-GAN output regularly exceeds ±1.0 and AudioWAV.data either
        // clips at 0 dBFS (peakNormalize:false) or normalizes back up to
        // 0 dBFS (peakNormalize:true) — both ~7 dB hotter than the
        // reference (-7 dBFS peak / ~-27 dB RMS).
        let targetPeak: Float = 0.4467  // 10^(-7.0 / 20)
        let maxAbs = samples.reduce(Float(0)) { Swift.max($0, abs($1)) }
        let scaledSamples: [Float] =
            maxAbs > 0
            ? samples.map { $0 * (targetPeak / maxAbs) }
            : samples
        return try AudioWAV.data(
            from: scaledSamples,
            sampleRate: Double(StyleTTS2Constants.sampleRate),
            peakNormalize: false
        )
    }

    /// Same as `synthesize` but returns raw fp32 PCM samples + sample rate.
    /// Used by the tts-benchmark harness and ASR pairing — avoids the WAV
    /// encode/decode round trip.
    public func synthesizeSamples(
        text: String,
        voiceStyleURL: URL,
        language: MultilingualG2PLanguage = .americanEnglish,
        diffusionSteps: Int = StyleTTS2Constants.defaultDiffusionSteps,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        randomSeed: UInt64? = nil
    ) async throws -> (samples: [Float], sampleRate: Int) {
        let result = try await synthesizeDetailed(
            text: text,
            voiceStyleURL: voiceStyleURL,
            language: language,
            diffusionSteps: diffusionSteps,
            alpha: alpha,
            beta: beta,
            randomSeed: randomSeed
        )
        return (result.samples, result.sampleRate)
    }

    /// Detailed variant that surfaces per-stage timings and the final
    /// `T_tok` / `T_a` shapes. Used by the per-stage benchmark sweep.
    public func synthesizeDetailed(
        text: String,
        voiceStyleURL: URL,
        language: MultilingualG2PLanguage = .americanEnglish,
        diffusionSteps: Int = StyleTTS2Constants.defaultDiffusionSteps,
        alpha: Float = StyleTTS2Constants.defaultAlpha,
        beta: Float = StyleTTS2Constants.defaultBeta,
        randomSeed: UInt64? = nil
    ) async throws -> StyleTTS2SynthesisResult {
        guard isInitialized else {
            throw StyleTTS2Error.modelNotLoaded("StyleTTS2Manager.initialize() not called")
        }
        try Self.validateLanguage(language)
        let voice = try StyleTTS2VoiceStyle.load(from: voiceStyleURL)
        let (_, ids) = try await tokenize(text: text, language: language)
        let options = StyleTTS2Synthesizer.Options(
            diffusionSteps: diffusionSteps,
            alpha: alpha,
            beta: beta,
            randomSeed: randomSeed
        )
        return try await StyleTTS2Synthesizer.synthesize(
            ids: ids, voice: voice, options: options, store: modelStore)
    }

    /// Run the text frontend (preprocess → G2P → vocab encode) end-to-end.
    /// Available before synthesis is wired so callers can validate the
    /// bundle, vocab, and G2P installation.
    public func tokenize(
        text: String,
        language: MultilingualG2PLanguage = .americanEnglish
    ) async throws -> (phonemes: String, ids: [Int32]) {
        guard isInitialized else {
            throw StyleTTS2Error.modelNotLoaded("StyleTTS2Manager.initialize() not called")
        }
        try Self.validateLanguage(language)
        let phonemes = try await StyleTTS2Phonemizer.phonemize(
            text: text, language: language)
        let vocab = try await legacyAssetStore.vocabulary()
        let ids = vocab.encode(phonemes)
        return (phonemes, ids)
    }

    /// Diagnostic tokenize: same as `tokenize(text:language:)` but also
    /// returns the per-scalar drop frequency from
    /// `StyleTTS2Vocab.encodeWithReport`. Used by the CLI to quantify
    /// how much of the misaki BART G2P output the espeak-ng-trained
    /// 178-token vocab can actually consume.
    public func tokenizeWithReport(
        text: String,
        language: MultilingualG2PLanguage = .americanEnglish
    ) async throws -> (
        phonemes: String, ids: [Int32], dropped: [Unicode.Scalar: Int]
    ) {
        guard isInitialized else {
            throw StyleTTS2Error.modelNotLoaded("StyleTTS2Manager.initialize() not called")
        }
        try Self.validateLanguage(language)
        let phonemes = try await StyleTTS2Phonemizer.phonemize(
            text: text, language: language)
        let vocab = try await legacyAssetStore.vocabulary()
        let (ids, dropped) = vocab.encodeWithReport(phonemes)
        return (phonemes, ids, dropped)
    }

    /// Resolve the repo root for the bundled `voices/` directory. Used by
    /// the CLI to map `--voice-name <id>` → `voices/ref_s_<id>.bin`.
    /// Requires `initialize()` to have completed.
    public func voicesRepoRoot() async throws -> URL {
        guard isInitialized else {
            throw StyleTTS2Error.modelNotLoaded("StyleTTS2Manager.initialize() not called")
        }
        return try await legacyAssetStore.repoRoot()
    }

    /// Drop strong references to the loaded models. Call this when the app
    /// is going to background or under memory pressure. Re-call
    /// `initialize()` to reload. Cancels any in-flight `initialize()` so a
    /// late-arriving success can't flip `isInitialized` back to true after
    /// the cleanup has already run.
    public func cleanup() async {
        initializeTask?.cancel()
        initializeTask = nil
        await modelStore.cleanup()
        isInitialized = false
    }
}
