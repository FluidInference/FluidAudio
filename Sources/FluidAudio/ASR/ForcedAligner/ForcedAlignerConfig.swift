import Foundation

/// Configuration constants for the Qwen3-ForcedAligner-0.6B CoreML model.
///
/// Architecture: audio_encoder -> embedding + merge -> 28 decoder layers + lm_head (fused)
/// NAR (non-autoregressive): single prefill pass, no KV-cache decode loop.
/// Outputs per-word timestamps via argmax on a 5000-class timestamp vocabulary.
public enum ForcedAlignerConfig {
    // MARK: Audio

    public static let sampleRate = 16000
    public static let hopLength = 160
    public static let numMelBins = 128

    /// Audio encoder window size in mel frames (n_window * 2 = 50 * 2 = 100).
    public static let melWindowSize = 100

    /// Conv2D downsampling factor: 3 layers of stride-2 -> 8x reduction.
    public static let convDownsampleFactor = 8

    // MARK: Encoder

    public static let encoderOutputDim = 1024

    // MARK: Decoder (LLM)

    public static let hiddenSize = 1024
    public static let numDecoderLayers = 28
    public static let numAttentionHeads = 16
    public static let numKVHeads = 8
    public static let headDim = 128
    public static let ropeTheta: Double = 1_000_000
    public static let mropeSection = [24, 20, 20]

    /// Fixed prefill sequence length (all inputs padded to this).
    public static let prefillSeqLen = 1024

    /// LM head output vocabulary size (timestamp token IDs 0-4999).
    public static let lmHeadOutputDim = 5000

    // MARK: Special Tokens

    public static let audioStartTokenId = 151_669
    public static let audioEndTokenId = 151_670
    public static let audioPadTokenId = 151_676
    public static let timestampTokenId = 151_705

    /// Milliseconds per timestamp segment (each output class = 80ms).
    public static let timestampSegmentTimeMs = 80
}
