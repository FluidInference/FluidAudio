import Foundation

/// Manages text-to-speech synthesis using the StyleTTS2 4-stage diffusion
/// pipeline (LibriTTS multi-speaker checkpoint).
///
/// Pipeline (per utterance):
///  1. `text_predictor` (fp16, ANE) → `bert_dur` features + duration logits.
///  2. `diffusion_step_512` (fp16, CPU+GPU) — ADPM2 sampler, 5× per utt + CFG.
///  3. `f0n_energy` (fp16, ANE) → F0 + energy regression.
///  4. `decoder` (fp32, CPU+GPU) — HiFi-GAN waveform synthesis.
///
/// The Swift host is responsible for:
///   - espeak-ng IPA phonemization + vocab lookup,
///   - the ADPM2 + Karras sampler loop around `diffusion_step`,
///   - cumsum-of-durations → one-hot → matmul hard-alignment,
///   - bucket selection (round token length → text_predictor; round
///     mel frames → decoder).
///
/// **Status:** scaffold only. Synthesis is not yet implemented; calls to
/// `synthesize` throw `processingFailed`. The asset bring-up (download +
/// model store) is wired up so dependent layers can land incrementally.
public actor StyleTTS2Manager {

    private let logger = AppLogger(category: "StyleTTS2Manager")
    private let modelStore: StyleTTS2ModelStore
    private let synthesizer: StyleTTS2Synthesizer
    private var isInitialized = false

    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public init(directory: URL? = nil) {
        let store = StyleTTS2ModelStore(directory: directory)
        self.modelStore = store
        self.synthesizer = StyleTTS2Synthesizer(modelStore: store)
    }

    public var isAvailable: Bool {
        isInitialized
    }

    /// Download the bundle (mlpackages + config + vocab) and resolve the
    /// repo root. Models are loaded lazily on first synthesis call.
    ///
    /// Also decodes and validates `config.json` against `StyleTTS2Constants`
    /// so wrong-bundle / partial-download / version-mismatch errors surface
    /// here rather than as cryptic CoreML shape errors at synthesis time.
    public func initialize(
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        _ = try await modelStore.ensureAssetsAvailable(progressHandler: progressHandler)
        let config = try await modelStore.bundleConfig()
        try config.validate()

        // The English G2P CoreML assets ship in the kokoro repo and are loaded
        // from `~/.cache/fluidaudio/Models/kokoro/`. `G2PModel.loadIfNeeded`
        // only reads from cache (it never downloads), so a first-time
        // StyleTTS2 user who has never run kokoro/kokoroAne would otherwise
        // hit a cryptic `G2PModelError.vocabLoadFailed` deep inside
        // `synthesize`. Mirror `KokoroAneManager.initialize` and fetch the
        // shared G2P assets explicitly here.
        //
        // NOTE: pass nil (not the caller's `directory`) — `G2PModel.shared`
        // is a singleton that hardcodes the default cache path
        // (`TtsModels.cacheDirectoryURL()/Models/kokoro`). Honouring a
        // custom override here would write to a path the singleton can't
        // read and we'd still hit `vocabLoadFailed`.
        try await KokoroAneResourceDownloader.ensureG2PAssets(
            directory: nil, progressHandler: progressHandler)
        try await G2PModel.shared.ensureModelsAvailable()

        isInitialized = true
        logger.notice("StyleTTS2Manager initialized")
    }

    /// Synthesize text to a WAV blob at 24 kHz.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize.
    ///   - voiceStyleURL: Path to a precomputed `ref_s.bin` (256 fp32 LE,
    ///     1024 bytes) produced offline by
    ///     `mobius-styletts2/scripts/06_dump_ref_s.py`. The on-device style
    ///     encoder export is a follow-up; until it lands, voices ship as
    ///     these blobs.
    ///   - language: G2P language for phonemization.
    ///   - diffusionSteps: Number of ADPM2 sampler iterations (default 5).
    ///   - alpha: Acoustic style mix weight (default 0.3).
    ///   - beta: Prosody style mix weight (default 0.7).
    ///   - randomSeed: Seed for the diffusion noise RNG. `nil` → use the
    ///     system RNG (non-reproducible).
    /// - Returns: WAV audio data (24 kHz, mono, 16-bit PCM).
    public func synthesize(
        text: String,
        voiceStyleURL: URL,
        language: MultilingualG2PLanguage = .americanEnglish,
        diffusionSteps: Int = StyleTTS2Constants.defaultDiffusionSteps,
        alpha: Float = 0.3,
        beta: Float = 0.7,
        randomSeed: UInt64? = nil
    ) async throws -> Data {
        guard isInitialized else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 model not initialized")
        }
        let voice = try StyleTTS2VoiceStyle.load(from: voiceStyleURL)
        let (_, ids) = try await tokenize(text: text, language: language)
        let options = StyleTTS2Synthesizer.Options(
            diffusionSteps: diffusionSteps,
            alpha: alpha,
            beta: beta,
            randomSeed: randomSeed
        )
        return try await synthesizer.synthesize(ids: ids, voice: voice, options: options)
    }

    /// Run the text frontend (preprocess → G2P → vocab encode) end-to-end.
    ///
    /// Available before the diffusion synthesizer is wired so callers can
    /// validate the bundle, vocab, and G2P installation. The returned ids
    /// are exactly what the `text_predictor` model would consume after
    /// padding to a bucket length.
    ///
    /// - Parameters:
    ///   - text: Source text to phonemize.
    ///   - language: G2P language. Defaults to `.americanEnglish` because
    ///     the shipped LibriTTS checkpoint is English-only.
    /// - Returns: A tuple of the IPA phoneme string and its `[Int32]` token
    ///   id encoding under the 178-token espeak-ng vocab.
    public func tokenize(
        text: String,
        language: MultilingualG2PLanguage = .americanEnglish
    ) async throws -> (phonemes: String, ids: [Int32]) {
        guard isInitialized else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 model not initialized")
        }
        let phonemes = try await StyleTTS2Phonemizer.phonemize(
            text: text, language: language)
        let vocab = try await modelStore.vocabulary()
        let ids = vocab.encode(phonemes)
        return (phonemes, ids)
    }

    public func cleanup() {
        isInitialized = false
    }

    // MARK: - Internal accessors (for the synthesizer once it lands)

    /// Expose the model store to the not-yet-written synthesizer module.
    public func underlyingModelStore() -> StyleTTS2ModelStore {
        modelStore
    }
}
