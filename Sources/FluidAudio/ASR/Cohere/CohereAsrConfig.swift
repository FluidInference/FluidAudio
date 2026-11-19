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

    /// Sliding-window overlap (seconds) when chunking audio > `maxAudioSeconds`.
    ///
    /// Matches upstream `cohere-pytorch/config.json` `overlap_chunk_second: 5`.
    public static let chunkOverlapSeconds: Float = 5.0

    /// Sliding-window hop (seconds) for long-form audio: 35 − 5 = 30 s.
    public static var chunkHopSeconds: Float { maxAudioSeconds - chunkOverlapSeconds }

    /// Longest chunk `transcribeLong` decodes for a language written without
    /// spaces (Japanese, Chinese). There the 35 s sliding window loses whole
    /// sentences: a window that starts a few seconds into an utterance makes
    /// the model skip ahead and end early, and windows holding several
    /// sentences drop some. Such audio is cut into non-overlapping chunks that
    /// start and end at pauses instead (#1140).
    public static let pauseChunkSeconds: Float = 20.0

    /// How far before each chunk limit `transcribeLong` looks for the pause to
    /// cut at, so pause chunks run 10–20 s. Upstream `split_audio_chunks_energy`
    /// searches only the last 5 s of a 35 s chunk.
    public static let pauseSearchSeconds: Float = 10.0

    /// Window over which the pause search compares energy (100 ms), upstream
    /// `min_energy_window_samples: 1600`.
    public static let pauseEnergyWindowSamples: Int = 1_600

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

    /// KV-cache length of the 108-slot decoder exports (v1 dynamic, v2 static).
    ///
    /// The runtime reads the real cache length from the loaded decoder's
    /// `k_cache_0` input shape (`CoherePipeline.LoadedModels.cacheLength`);
    /// this constant only documents the legacy exports. The default v3
    /// decoder ships a 256-slot cache — the 108-slot cache was shared by the
    /// 10-token prompt and the output, so a dense 35 s window (up to ~225
    /// tokens) hit the cap and silently dropped the rest of the window.
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

        /// Whether the language is written without spaces between words
        /// (upstream `NO_SPACE_LANGS`). `transcribeLong` cuts such audio at
        /// pauses and joins the chunk texts with no separator.
        public var writesWithoutSpaces: Bool {
            self == .japanese || self == .chinese
        }

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
