import Foundation

/// Configuration for the Paraformer-large (zh) CoreML pipeline.
///
/// Non-autoregressive: SANM encoder -> CIF predictor (host integrate-and-fire on
/// alphas from `ParaformerCifAlphas`) -> parallel decoder. Mirrors the conversion
/// in `FluidInference/paraformer-large-zh-coreml`.
public enum ParaformerConfig {
    /// LFR feature dim (80-bin fbank x LFR m=7).
    public static let featureDim = 560
    /// Encoder hidden dim.
    public static let encoderDim = 512

    /// Encoder enumerated sequence-length buckets (post-LFR frames). Host pads
    /// features up to the smallest bucket >= T.
    public static let encoderBuckets = [128, 256, 512, 1024, 1800]

    /// Decoder fixed shapes: cross-attention memory length and token budget.
    public static let decoderEncFrames = 512
    public static let decoderMaxTokens = 128

    /// Special token ids (CharTokenizer): <blank>=0, <s>=1, </s>=2.
    public static let blankId = 0
    public static let sosId = 1
    public static let eosId = 2

    /// CIF hyperparameters (CifPredictorV2).
    public static let cifThreshold: Float = 1.0
    public static let cifTailThreshold: Float = 0.45

    public static let sampleRate = 16_000
    /// Kaldi feeds int16-range waveforms; AudioConverter yields [-1, 1].
    public static let waveformScale: Float = 32_768.0

    public static func pickEncoderBucket(forFrames frames: Int) -> Int {
        for b in encoderBuckets where b >= frames { return b }
        return encoderBuckets.last ?? 1800
    }
}
