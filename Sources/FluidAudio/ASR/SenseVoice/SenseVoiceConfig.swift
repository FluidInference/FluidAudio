import Foundation

/// Configuration constants for the SenseVoiceSmall CoreML pipeline.
///
/// SenseVoiceSmall is non-autoregressive: a SANM encoder + single CTC head.
/// The CoreML export is a 3-stage pipeline (preprocessor → encoder+CTC → host
/// greedy decode). These constants mirror the conversion in
/// `FluidInference/sensevoice-small-coreml`.
public enum SenseVoiceConfig {
    /// LFR feature dimension (80-bin fbank × LFR m=7).
    public static let featureDim = 560

    /// Enumerated encoder sequence-length buckets (post-LFR frames). The host
    /// pads the preprocessor's feature output up to the smallest bucket ≥ T.
    public static let buckets = [128, 256, 512, 1024, 1800]

    /// Number of query tokens the encoder prepends (language + 2 event/emotion
    /// + text-norm). The first 4 logit positions are these special tokens.
    public static let numQueryTokens = 4

    /// CTC blank token id (SenseVoice uses `<unk>` = 0 as blank).
    public static let blankId = 0

    /// Default language embed index (`0` = auto-detect).
    public static let defaultLanguage: Int32 = 0

    /// Default text-norm embed index (`15` = woitn, no inverse text-norm).
    public static let defaultTextNorm: Int32 = 15

    public static let sampleRate = 16_000

    /// Kaldi feeds waveforms in int16 range; AudioConverter yields [-1, 1], so
    /// the preprocessor input is scaled by this factor.
    public static let waveformScale: Float = 32_768.0

    /// Largest feature length we can serve (= last bucket). ~108 s of audio.
    public static var maxFrames: Int { buckets.last ?? 1800 }

    /// Smallest bucket ≥ `frames` (clamped to the largest bucket).
    public static func pickBucket(forFrames frames: Int) -> Int {
        for b in buckets where b >= frames { return b }
        return buckets.last ?? 1800
    }
}
