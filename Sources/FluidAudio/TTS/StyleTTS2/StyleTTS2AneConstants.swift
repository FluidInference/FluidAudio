import Foundation

/// Compile-time constants for the StyleTTS2-ANE 7-stage CoreML chain.
///
/// Source of truth:
///   `mobius/models/tts/styletts2/scripts/ane/_styletts2_ane_lib.py`
///   (`MAX_T_TOK`, `MAX_T_A`, `UPSAMPLE_SCALE`) and the per-stage I/O
///   contracts in the 01..07 export scripts.
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
public enum StyleTTS2AneConstants {

    // MARK: - Audio

    public static let sampleRate: Int = 24_000
    public static let hopSize: Int = 300
    public static let upsampleScale: Int = 300
    public static let melChannels: Int = 80

    // MARK: - Tokenizer

    /// 178-token espeak-ng IPA + stress vocab (same as legacy StyleTTS2).
    public static let vocabSize: Int = 178
    public static let padTokenId: Int = 0

    // MARK: - Bucketing bounds (mirror MAX_T_TOK / MAX_T_A in Python lib)

    /// PLBERT max_position_embeddings — phoneme tokens cannot exceed this.
    public static let maxInputTokens: Int = 512
    /// Static shape used by Prosody / Noise / Vocoder. Sentences whose
    /// `T_a` (= sum of predicted durations) exceeds this must be chunked.
    public static let maxAcousticFrames: Int = 2_000

    // MARK: - Style vector layout (unchanged from legacy StyleTTS2)

    /// Per-half style channels (acoustic, prosody).
    public static let styleDim: Int = 128
    /// Concat of acoustic + prosody style vectors (`ref_s` input dim).
    public static let refStyleDim: Int = 256
    public static let hiddenDim: Int = 512

    // MARK: - Sampler (ADPM2 + Karras + CFG, reused from legacy)

    public static let defaultDiffusionSteps: Int = 5
    public static let karrasRho: Float = 9.0
    public static let karrasSigmaMin: Float = 0.0001
    public static let karrasSigmaMax: Float = 3.0
    public static let cfgScale: Float = 1.0

    // MARK: - Repository

    public static let defaultModelsSubdirectory: String = "Models"
    /// HuggingFace subdirectory (mirrors the Kokoro `ANE/` precedent).
    public static let huggingFaceSubdirectory: String = "ANE"
}
