import Foundation

/// Configuration for Cohere Transcribe CoreML ASR model.
public enum CohereAsrConfig {
    /// Sample rate expected by the model (16kHz).
    public static let sampleRate: Int = 16000

    /// Maximum audio duration in seconds (35s).
    ///
    /// Matches the encoder mel input `[1, 128, 3500]` (3500 frames * 160 hop
    /// / 16000 sr = 35s).
    public static let maxAudioSeconds: Float = 35.0

    /// Maximum number of audio samples (560,000 at 16kHz = 35 seconds).
    public static let maxSamples: Int = 560_000

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
        /// FFT size used by `CohereMelSpectrogram`: `nextPowerOfTwo(winLength=400) = 512`.
        public static let nFFT: Int = 512
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
        /// Start of context token.
        public static let startOfContext: Int = 7
        /// Emotion undefined token.
        public static let emoUndefined: Int = 16
        /// Punctuation token.
        public static let pnc: Int = 5
        /// No inverse text normalization.
        public static let noitn: Int = 9
        /// No timestamp token.
        public static let notimestamp: Int = 11
        /// No diarization token.
        public static let nodiarize: Int = 13
        /// Word boundary marker.
        public static let wordBoundary: Int = 13764
    }

    /// Supported languages.
    public enum Language: String, CaseIterable, Sendable {
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

        /// Language token ID (used as start token for conditioned generation).
        public var tokenId: Int {
            switch self {
            case .english: return 62
            case .french: return 69
            case .german: return 76
            case .spanish: return 169
            case .italian: return 97
            case .portuguese: return 149
            case .dutch: return 60
            case .polish: return 148
            case .greek: return 77
            case .arabic: return 28
            case .japanese: return 98
            case .chinese: return 50
            case .vietnamese: return 194
            case .korean: return 110
            }
        }

        /// Build the prompt sequence for this language.
        ///
        /// Cohere models expect a specific prompt sequence:
        /// 1. Word boundary marker
        /// 2. Start of context
        /// 3. Start of transcript
        /// 4. Emotion undefined
        /// 5-6. Language token (repeated twice)
        /// 7. Punctuation
        /// 8. No inverse text normalization
        /// 9. No timestamp
        /// 10. No diarization
        public var promptSequence: [Int] {
            let langToken = tokenId
            return [
                SpecialTokens.wordBoundary,  // ▁
                SpecialTokens.startOfContext,  // <|startofcontext|>
                SpecialTokens.startToken,  // <|startoftranscript|>
                SpecialTokens.emoUndefined,  // <|emo:undefined|>
                langToken,  // <|en|> (or other language)
                langToken,  // <|en|> (repeated)
                SpecialTokens.pnc,  // <|pnc|>
                SpecialTokens.noitn,  // <|noitn|>
                SpecialTokens.notimestamp,  // <|notimestamp|>
                SpecialTokens.nodiarize,  // <|nodiarize|>
            ]
        }
    }
}
