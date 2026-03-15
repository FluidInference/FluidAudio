import Foundation

/// Constants for the PocketTTS flow-matching language model TTS backend.
public enum PocketTtsConstants {

    // MARK: - Audio

    public static let audioSampleRate: Int = 24_000
    /// Each generation step produces one frame of audio: 1920 samples = 80ms at 24kHz.
    public static let samplesPerFrame: Int = 1_920

    // MARK: - Model dimensions

    /// Audio code dimensionality — output of flow_decoder, input to mimi_decoder.
    public static let latentDim: Int = 32
    /// Transformer hidden state size — shared by flowlm_step output and flow_decoder input.
    public static let transformerDim: Int = 1024
    /// SentencePiece vocabulary size for text tokenization.
    public static let vocabSize: Int = 4001
    /// Embedding dimension for voice and text tokens (matches transformerDim).
    public static let embeddingDim: Int = 1024

    // MARK: - Generation parameters

    /// Number of Euler integration steps in flow_decoder (noise → audio code).
    public static let numLsdSteps: Int = 8
    /// Controls randomness in flow_decoder: scales initial noise by sqrt(temperature).
    public static let temperature: Float = 0.7
    /// flowlm_step EOS logit threshold — above this means the model is done speaking.
    public static let eosThreshold: Float = -4.0
    public static let shortTextPadFrames: Int = 3
    public static let longTextExtraFrames: Int = 1
    public static let extraFramesAfterDetection: Int = 2
    public static let shortTextWordThreshold: Int = 5
    /// Max text tokens per chunk — keeps total KV cache usage under kvCacheMaxLen.
    public static let maxTokensPerChunk: Int = 50

    // MARK: - KV cache

    /// Number of transformer layers, each with its own KV cache.
    public static let kvCacheLayers: Int = 6
    /// Max KV cache positions: voice (~125) + text (≤50) + generated frames.
    public static let kvCacheMaxLen: Int = 512

    // MARK: - Voice

    public static let defaultVoice: String = "alba"
    /// Default voice prompt length in frames. Cloned voices may differ (up to 250).
    public static let voicePromptLength: Int = 125

    // MARK: - Repository

    public static let defaultModelsSubdirectory: String = "Models"
}
