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
/// - int8 submodels (any language): `languages/<id>/int8/<basename>.mlmodelc`
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

    /// HF subdirectory under the pocket-tts-coreml repo root for the FP16
    /// language pack. English returns `nil` to preserve the legacy
    /// root-level layout (avoids forcing existing caches to re-download).
    public var repoSubdirectory: String? {
        self == .english ? nil : "v2/\(rawValue)"
    }

    /// HF subdirectory under the pocket-tts-coreml repo root that contains
    /// the int8 weight-only quantized variants of this language's submodels.
    /// Uniform across all languages: `languages/<id>/int8/`.
    public var int8RepoSubdirectory: String {
        "languages/\(rawValue)/int8"
    }
}

/// Precision (and thus on-disk asset) of an individual PocketTTS submodel.
///
/// All four submodels (cond_step, flowlm_step, flow_decoder, mimi_decoder)
/// have identical I/O signatures regardless of precision, so any subset
/// can be mixed at runtime.
///
/// Quality summary (English 6L A/B, prompt-relative):
/// - cond_step int8: speaker_sim 0.984, Pearson 0.94 — safe
/// - flowlm_step int8: speaker_sim 0.989, Pearson 0.94 — safe
/// - flow_decoder int8: speaker_sim 0.981, Pearson 0.78 — risky (audible drift
///   from the LSD 8-step inner loop compounding the quantization error)
/// - mimi_decoder int8: speaker_sim 0.998, Pearson 1.00 — transparent
public enum PocketTtsModelPrecision: String, Sendable, CaseIterable {
    case fp16
    case int8
}

/// Per-submodel quantization configuration for PocketTTS.
///
/// Each of the four submodels can be loaded independently as fp16 or int8.
/// Defaults to all-fp16 for backward compatibility.
public struct PocketTtsQuantization: Sendable, Equatable {
    public var condStep: PocketTtsModelPrecision
    public var flowlmStep: PocketTtsModelPrecision
    public var flowDecoder: PocketTtsModelPrecision
    public var mimiDecoder: PocketTtsModelPrecision

    public init(
        condStep: PocketTtsModelPrecision = .fp16,
        flowlmStep: PocketTtsModelPrecision = .fp16,
        flowDecoder: PocketTtsModelPrecision = .fp16,
        mimiDecoder: PocketTtsModelPrecision = .fp16
    ) {
        self.condStep = condStep
        self.flowlmStep = flowlmStep
        self.flowDecoder = flowDecoder
        self.mimiDecoder = mimiDecoder
    }

    /// All four submodels in fp16. Default; matches legacy behavior.
    public static let allFp16 = PocketTtsQuantization()

    /// cond_step + flowlm_step + mimi_decoder in int8; flow_decoder kept fp16.
    /// Recommended balance: ~75% size reduction with no audible quality loss.
    public static let safeInt8 = PocketTtsQuantization(
        condStep: .int8,
        flowlmStep: .int8,
        flowDecoder: .fp16,
        mimiDecoder: .int8
    )

    /// All four submodels in int8. Maximum size reduction (~74%); audible
    /// drift from flow_decoder int8 but speaker similarity remains > 0.94.
    public static let aggressiveInt8 = PocketTtsQuantization(
        condStep: .int8,
        flowlmStep: .int8,
        flowDecoder: .int8,
        mimiDecoder: .int8
    )

    /// `true` if any submodel is set to int8 (and thus int8 assets are needed).
    public var hasAnyInt8: Bool {
        condStep == .int8 || flowlmStep == .int8 || flowDecoder == .int8 || mimiDecoder == .int8
    }

    /// `true` if every submodel is fp16 (legacy fast path).
    public var isAllFp16: Bool { !hasAnyInt8 }
}
