import Foundation

/// Constants for the VoxCPM 1.5 diffusion autoregressive TTS backend.
public enum VoxCpmConstants {

    // MARK: - Audio

    /// Output sample rate (44.1 kHz).
    public static let audioSampleRate: Int = 44_100
    /// Hop length for the AudioVAE encoder/decoder.
    public static let hopLength: Int = 1_764
    /// Token rate in Hz (6.25 Hz = 160ms per token patch).
    public static let tokenRateHz: Float = 6.25

    // MARK: - Model dimensions

    /// Transformer hidden state size.
    public static let hiddenSize: Int = 1024
    /// Feature dimension (latent channels in AudioVAE).
    public static let featDim: Int = 64
    /// Patch size for latent patches (4 frames per patch).
    public static let patchSize: Int = 4
    /// Vocabulary size for the MiniCPM tokenizer.
    public static let vocabSize: Int = 73_448
    /// Embedding scale factor (MuP scaling).
    public static let scaleEmb: Float = 12.0

    // MARK: - LM architecture

    /// Number of layers in the base LM (MiniCPM-4).
    public static let baseLmLayers: Int = 24
    /// Number of layers in the residual LM.
    public static let residualLmLayers: Int = 8
    /// Number of KV heads per layer.
    public static let numKvHeads: Int = 2
    /// Dimension per attention head.
    public static let headDim: Int = 64

    // MARK: - KV cache

    /// Total KV cache tensors for base LM (24 layers × 2 for k,v).
    public static let baseCacheCount: Int = 48
    /// Total KV cache tensors for residual LM (8 layers × 2 for k,v).
    public static let residualCacheCount: Int = 16
    /// Maximum sequence length for KV caches.
    public static let maxSeqLen: Int = 512

    // MARK: - Generation parameters

    /// Number of Euler integration steps for LocDiT diffusion.
    public static let defaultTimesteps: Int = 10
    /// Classifier-free guidance scale.
    public static let defaultCfgValue: Float = 2.0
    /// Maximum generation steps (patches).
    public static let defaultMaxLen: Int = 200
    /// Minimum steps before stop head can trigger.
    public static let defaultMinLen: Int = 5

    // MARK: - Special tokens

    /// Audio start token ID appended after text tokens.
    public static let audioStartToken: Int = 101

    // MARK: - AudioVAE

    /// Fixed encoder input length: 5 seconds at 44.1 kHz.
    public static let encoderSamples: Int = 5 * 44_100  // 220500

    // MARK: - Repository

    public static let defaultModelsSubdirectory: String = "Models"
}
