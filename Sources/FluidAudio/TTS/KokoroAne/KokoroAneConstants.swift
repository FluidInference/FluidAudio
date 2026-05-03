import Foundation

/// Compile-time constants for the laishere/kokoro 7-stage CoreML chain.
///
/// Source of truth: mobius/models/tts/kokoro/laishere-coreml/convert-coreml.py
/// (specifically `compute_shape_bounds(max_frames=2000)` and the per-stage
/// I/O contracts).
public enum KokoroAneConstants {

    /// Default voice id for the English (`ANE/`) variant.
    public static let defaultVoice = "af_heart"

    /// Default voice id for the Mandarin (`ANE-zh/`) variant.
    public static let defaultVoiceMandarin = "zf_001"

    /// Output sample rate of the iSTFT in `KokoroTail.mlpackage`.
    public static let sampleRate = 24_000

    /// BOS / EOS token id used by both `convert-coreml.py` and the iOS demo.
    public static let bosTokenId: Int32 = 0
    public static let eosTokenId: Int32 = 0

    /// ALBERT context window — input_ids cannot exceed this, so the IPA
    /// phoneme sequence (excluding BOS/EOS) must be ≤ 510.
    public static let maxInputTokens = 512
    public static let maxPhonemeLength = 510

    /// Voice pack rows × columns. The pack is stored flat as `[510, 256]` fp32:
    ///   * row index = `min(max(T_enc - 1, 0), 509)` (utterance-length bucket)
    ///   * cols `[0..<128]`   = `style_timbre` (→ Noise + Vocoder)
    ///   * cols `[128..<256]` = `style_s`      (→ PostAlbert + Prosody)
    public static let voicePackRows = 510
    public static let voicePackCols = 256

    /// `--max-frames` baked into the converted models. Sentences whose `T_a`
    /// exceeds this must be skipped or chunked.
    public static let maxAcousticFrames = 2_000

    /// Default playback speed factor for PostAlbert.
    public static let defaultSpeed: Float = 1.0
}

/// Language variant of the laishere/kokoro 7-stage CoreML chain.
///
/// The 7-stage chain is language-agnostic by construction (input ids, voice
/// slices, and per-stage I/O contracts are identical across variants). Only
/// the embedding vocab, HF subdirectory, voice-file layout, and the default
/// voice id differ.
///
/// | Variant      | HF subdir | Vocab | Default voice | Voice layout       |
/// |--------------|-----------|-------|---------------|--------------------|
/// | `.english`   | `ANE/`    | 177   | `af_heart`    | flat (`<voice>.bin`)            |
/// | `.mandarin`  | `ANE-zh/` | 171   | `zf_001`      | nested (`voices/<voice>.bin`)   |
public enum KokoroAneVariant: String, CaseIterable, Sendable {
    case english
    case mandarin

    /// Default voice id shipped with the variant's HF bundle.
    public var defaultVoice: String {
        switch self {
        case .english: return KokoroAneConstants.defaultVoice
        case .mandarin: return KokoroAneConstants.defaultVoiceMandarin
        }
    }

    /// True if voice packs live under a `voices/` subdirectory inside the
    /// repo bundle (Mandarin); false if they sit at the bundle root (English).
    public var useVoicesSubdir: Bool {
        switch self {
        case .english: return false
        case .mandarin: return true
        }
    }

    /// HuggingFace repo case for this variant.
    public var repo: Repo {
        switch self {
        case .english: return .kokoroAne
        case .mandarin: return .kokoroAneZh
        }
    }
}
