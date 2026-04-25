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

    /// Default number of transformer layers for the legacy English (6L) pack.
    /// For multi-language packs, prefer `PocketTtsLanguage.transformerLayers`.
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

/// Supported PocketTTS language packs (matches upstream
/// `kyutai/pocket-tts/languages/<id>/` folder names exactly).
///
/// File layout on `FluidInference/pocket-tts-coreml`:
/// - `english`: legacy root layout (`mimi_decoder_v2.mlmodelc` etc.)
/// - other languages: `v2/<id>/` subtree with `mimi_decoder.mlmodelc`
public enum PocketTtsLanguage: String, Sendable, CaseIterable {
    case english
    case french24L = "french_24l"
    case german
    case german24L = "german_24l"
    case italian
    case italian24L = "italian_24l"
    case portuguese
    case portuguese24L = "portuguese_24l"
    case spanish
    case spanish24L = "spanish_24l"

    /// Number of transformer layers in this language pack (6 or 24).
    public var transformerLayers: Int {
        switch self {
        case .english, .german, .italian, .portuguese, .spanish:
            return 6
        case .french24L, .german24L, .italian24L, .portuguese24L, .spanish24L:
            return 24
        }
    }

    /// HF subdirectory under the pocket-tts-coreml repo root.
    /// English returns `nil` to preserve the legacy root-level layout
    /// (avoids forcing existing caches to re-download).
    public var repoSubdirectory: String? {
        self == .english ? nil : "v2/\(rawValue)"
    }
}
