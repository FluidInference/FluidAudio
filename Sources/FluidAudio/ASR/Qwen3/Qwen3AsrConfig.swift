import Foundation

// MARK: - Qwen3-ASR Model Configuration

/// Configuration constants for the Qwen3-ASR-0.6B CoreML model.
///
/// Architecture: audio_encoder -> embedding + merge -> 28 decoder layers -> lm_head
/// The audio encoder processes mel spectrograms in fixed-size windows.
/// The LLM decoder generates text autoregressively with KV-cache.
public enum Qwen3AsrConfig {
    // MARK: Audio

    public static let sampleRate = 16000
    public static let numMelBins = 128

    /// Audio encoder window size in mel frames (n_window * 2).
    /// The encoder processes chunks of this size.
    public static let melWindowSize = 100

    /// Conv2D downsampling factor: 3 layers of stride-2 -> 8x reduction.
    public static let convDownsampleFactor = 8

    /// Output frames per mel window: ceil(100/8) = 13.
    public static let outputFramesPerWindow = (melWindowSize + convDownsampleFactor - 1) / convDownsampleFactor

    // MARK: Encoder

    public static let encoderDModel = 896
    public static let encoderOutputDim = 1024
    public static let encoderNumLayers = 18
    public static let encoderNumHeads = 14

    // MARK: Decoder (LLM)

    public static let hiddenSize = 1024
    public static let intermediateSize = 3072
    public static let numDecoderLayers = 28
    public static let numAttentionHeads = 16
    public static let numKVHeads = 8
    public static let headDim = 128
    public static let vocabSize = 151_936
    public static let ropeTheta: Double = 1_000_000
    public static let mropeSection = [24, 20, 20]

    // MARK: Special Tokens

    public static let audioStartTokenId = 151_669
    public static let audioEndTokenId = 151_670
    public static let audioTokenId = 151_676
    public static let asrTextTokenId = 151_704
    public static let eosTokenIds: Set<Int> = [151_645, 151_643]

    // MARK: Chat Template Tokens

    public static let imStartTokenId = 151_644
    public static let imEndTokenId = 151_645
    public static let systemTokenId = 8948
    public static let userTokenId = 872
    public static let assistantTokenId = 77_091
    public static let newlineTokenId = 198

    // MARK: Generation

    public static let maxSequenceLength = 4096
    public static let maxAudioSeconds: Double = 30.0

    /// Maximum KV cache sequence length for the stateful decoder model.
    /// Must match the value used during CoreML conversion.
    public static let maxCacheSeqLen = 512

    // MARK: - Supported Languages

    /// Supported languages for Qwen3-ASR (30 languages + 22 Chinese dialects).
    /// Use ISO 639-1 codes or English names. Pass nil for automatic detection.
    public enum Language: String, CaseIterable, Sendable {
        case chinese = "zh"
        case english = "en"
        case cantonese = "yue"
        case arabic = "ar"
        case german = "de"
        case french = "fr"
        case spanish = "es"
        case portuguese = "pt"
        case indonesian = "id"
        case italian = "it"
        case korean = "ko"
        case russian = "ru"
        case thai = "th"
        case vietnamese = "vi"
        case japanese = "ja"
        case turkish = "tr"
        case hindi = "hi"
        case malay = "ms"
        case dutch = "nl"
        case swedish = "sv"
        case danish = "da"
        case finnish = "fi"
        case polish = "pl"
        case czech = "cs"
        case filipino = "fil"
        case persian = "fa"
        case greek = "el"
        case hungarian = "hu"
        case macedonian = "mk"
        case romanian = "ro"

        /// English name for the language (used in Qwen3-ASR prompts).
        public var englishName: String {
            switch self {
            case .chinese: return "Chinese"
            case .english: return "English"
            case .cantonese: return "Cantonese"
            case .arabic: return "Arabic"
            case .german: return "German"
            case .french: return "French"
            case .spanish: return "Spanish"
            case .portuguese: return "Portuguese"
            case .indonesian: return "Indonesian"
            case .italian: return "Italian"
            case .korean: return "Korean"
            case .russian: return "Russian"
            case .thai: return "Thai"
            case .vietnamese: return "Vietnamese"
            case .japanese: return "Japanese"
            case .turkish: return "Turkish"
            case .hindi: return "Hindi"
            case .malay: return "Malay"
            case .dutch: return "Dutch"
            case .swedish: return "Swedish"
            case .danish: return "Danish"
            case .finnish: return "Finnish"
            case .polish: return "Polish"
            case .czech: return "Czech"
            case .filipino: return "Filipino"
            case .persian: return "Persian"
            case .greek: return "Greek"
            case .hungarian: return "Hungarian"
            case .macedonian: return "Macedonian"
            case .romanian: return "Romanian"
            }
        }

        /// Initialize from ISO code or English name.
        public init?(from string: String) {
            let lowercased = string.lowercased()
            // Try ISO code first
            if let lang = Language(rawValue: lowercased) {
                self = lang
                return
            }
            // Try English name
            if let lang = Language.allCases.first(where: { $0.englishName.lowercased() == lowercased }) {
                self = lang
                return
            }
            return nil
        }
    }
}
