import Foundation

/// Configuration for Cohere Transcribe CoreML ASR model.
public enum CohereAsrConfig {
    /// Sample rate expected by the model (16kHz).
    public static let sampleRate: Int = 16000

    /// Maximum audio duration in seconds (30s).
    public static let maxAudioSeconds: Float = 30.0

    /// Maximum number of audio samples (480,000 at 16kHz = 30 seconds).
    public static let maxSamples: Int = 480_000

    /// Vocabulary size.
    public static let vocabSize: Int = 16_384

    /// Encoder hidden size (Conformer blocks).
    public static let encoderHiddenSize: Int = 1280

    /// Decoder hidden size.
    public static let decoderHiddenSize: Int = 1024

    /// Number of encoder layers.
    public static let numEncoderLayers: Int = 48

    /// Number of decoder layers.
    public static let numDecoderLayers: Int = 8

    /// Number of attention heads in decoder.
    public static let numDecoderHeads: Int = 8

    /// Head dimension (1024 / 8).
    public static let headDim: Int = 128

    /// Maximum sequence length for decoder KV cache.
    public static let maxSeqLen: Int = 108

    /// Number of mel bins.
    public static let numMelBins: Int = 128

    /// Mel spectrogram parameters.
    public enum MelSpec {
        public static let nFFT: Int = 1024
        public static let hopLength: Int = 160
        public static let nMels: Int = 128
        public static let fMin: Float = 0.0
        public static let fMax: Float = 8000.0
        public static let preemphasis: Float = 0.97
    }

    /// Special tokens.
    public enum SpecialTokens {
        /// Unknown token.
        public static let unkToken: Int = 0
        /// No speech token.
        public static let noSpeechToken: Int = 1
        /// Padding token.
        public static let padToken: Int = 2
        /// End of text / End of sequence token.
        public static let eosToken: Int = 3
        /// Start of transcript token.
        public static let startToken: Int = 4
    }

    /// Supported languages.
    public enum Language: String, CaseIterable {
        case english = "en"
        case french = "fr"
        case german = "de"
        case spanish = "es"
        case italian = "it"
        case portuguese = "pt"
        case dutch = "nl"
        case polish = "pl"
        case greek = "el"
        case arabic = "ar"
        case japanese = "ja"
        case chinese = "zh"
        case vietnamese = "vi"
        case korean = "ko"

        public var englishName: String {
            switch self {
            case .english: return "English"
            case .french: return "French"
            case .german: return "German"
            case .spanish: return "Spanish"
            case .italian: return "Italian"
            case .portuguese: return "Portuguese"
            case .dutch: return "Dutch"
            case .polish: return "Polish"
            case .greek: return "Greek"
            case .arabic: return "Arabic"
            case .japanese: return "Japanese"
            case .chinese: return "Chinese"
            case .vietnamese: return "Vietnamese"
            case .korean: return "Korean"
            }
        }
    }
}
