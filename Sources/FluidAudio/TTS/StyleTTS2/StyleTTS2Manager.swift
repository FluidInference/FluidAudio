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
    private var isInitialized = false

    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public init(directory: URL? = nil) {
        self.modelStore = StyleTTS2ModelStore(directory: directory)
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
        isInitialized = true
        logger.notice("StyleTTS2Manager initialized")
    }

    /// Synthesize text to a WAV blob at 24 kHz.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize.
    ///   - referenceAudioURL: Reference clip used by the diffusion sampler to
    ///     extract `ref_s` (style + prosody, 256-dim concat). LibriTTS clones
    ///     are robust to ~5 s of speech.
    ///   - diffusionSteps: Number of ADPM2 sampler iterations (default 5).
    ///   - cfgScale: Classifier-free guidance scale (default 1.0).
    /// - Returns: WAV audio data (24 kHz, mono, fp32 PCM).
    public func synthesize(
        text: String,
        referenceAudioURL: URL,
        diffusionSteps: Int = StyleTTS2Constants.defaultDiffusionSteps,
        cfgScale: Float = StyleTTS2Constants.cfgScale
    ) async throws -> Data {
        guard isInitialized else {
            throw StyleTTS2Error.modelNotFound("StyleTTS2 model not initialized")
        }
        // Scaffold: synthesizer is not yet implemented. The pieces will land
        // in follow-up commits — phonemizer wiring, sampler loop, hard-align,
        // decoder driver. Throwing here is intentional so callers can wire
        // up the surface today and fill in the implementation incrementally.
        _ = text
        _ = referenceAudioURL
        _ = diffusionSteps
        _ = cfgScale
        throw StyleTTS2Error.processingFailed(
            "StyleTTS2 synthesis is not yet implemented")
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
