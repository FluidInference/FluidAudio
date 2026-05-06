import Foundation

/// Compile-time constants for the StyleTTS2 7-stage ANE CoreML chain.
///
/// Source of truth:
///   `mobius/models/tts/styletts2/scripts/ane/_styletts2_ane_lib.py`
///   (`MAX_T_TOK`, `MAX_T_A`, `UPSAMPLE_SCALE`) and the per-stage I/O
///   contracts in the 01..07 export scripts. The companion `config.json`
///   shipped with the `FluidInference/StyleTTS-2-coreml` repo carries the
///   audio/tokenizer numbers in machine-readable form; values here mirror
///   that contract so the framework can run without parsing the bundle
///   config before model load.
///
/// Stage layout (mirrors Kokoro-ANE exactly except DiffusionStep is
/// StyleTTS2-only and Kokoro's separate Tail collapses into Vocoder):
///
///   1. PLBert        — RangeDim(2..512), fp16, ANE
///   2. PostBert      — RangeDim(2..512), fp16, ANE  (BiLSTM unrolled)
///   3. Alignment     — RangeDim(2..512), fp16, ANE
///   4. DiffusionStep — STATIC shapes,    fp16, ANE  (StyleTTS2-only)
///   5. Prosody       — STATIC T_a=2000,  fp16, ANE
///   6. Noise         — STATIC T_a=2000,  fp32, ALL  (phase precision)
///   7. Vocoder       — STATIC T_a=2000,  fp16, ANE  (cos-Snake)
public enum StyleTTS2Constants {

    // MARK: - Audio

    public static let sampleRate: Int = 24_000
    /// HiFi-GAN hop size — `mel_frames * hopSize == samples`.
    public static let hopSize: Int = 300
    public static let upsampleScale: Int = 300
    public static let melChannels: Int = 80
    public static let nFFT: Int = 2_048
    public static let winLength: Int = 1_200

    // MARK: - Tokenizer

    /// 178-token espeak-ng IPA + stress vocabulary (mirrors
    /// `text_utils.TextCleaner` from upstream StyleTTS2).
    public static let vocabSize: Int = 178
    /// Pad token id (id 0 in the upstream cleaner table).
    public static let padTokenId: Int = 0

    // MARK: - Bucketing bounds (mirror MAX_T_TOK / MAX_T_A in Python lib)

    /// PLBERT max_position_embeddings — phoneme tokens cannot exceed this.
    public static let maxInputTokens: Int = 512
    /// Static shape used by Prosody / Noise / Vocoder. Sentences whose
    /// `T_a` (= sum of predicted durations) exceeds this must be chunked.
    public static let maxAcousticFrames: Int = 2_000

    // MARK: - Model dimensions

    /// Style/reference vector channels per branch (acoustic, prosody).
    public static let styleDim: Int = 128
    /// Concat of acoustic + prosody style vectors (`ref_s` input dim).
    public static let refStyleDim: Int = 256
    /// BERT/text predictor hidden size.
    public static let hiddenDim: Int = 512

    // MARK: - Sampler (ADPM2 + Karras schedule + CFG)

    /// Default number of diffusion sampler steps (5× per utterance).
    public static let defaultDiffusionSteps: Int = 5
    /// Default acoustic style mix weight (alpha) — 30 % diffusion-predicted,
    /// 70 % reference. Matches the upstream Python e2e reference
    /// (`99b_e2e_coreml.py`).
    public static let defaultAlpha: Float = 0.3
    /// Default prosody style mix weight (beta) — 70 % diffusion-predicted,
    /// 30 % reference. Matches the upstream Python e2e reference.
    public static let defaultBeta: Float = 0.7
    /// Karras schedule rho (controls sigma curvature). Matches the upstream
    /// e2e reference (`99b_e2e_coreml.py`) which uses rho=9.0 — *not* the
    /// k-diffusion default of 7.0.
    public static let karrasRho: Float = 9.0
    public static let karrasSigmaMin: Float = 0.0001
    public static let karrasSigmaMax: Float = 3.0
    /// Classifier-free guidance scale applied during the diffusion step.
    public static let cfgScale: Float = 1.0

    // MARK: - Repository

    public static let defaultModelsSubdirectory: String = "Models"
    /// HuggingFace subdirectory (mirrors the Kokoro `ANE/` precedent).
    public static let huggingFaceSubdirectory: String = "ANE"
}
