import Foundation

/// Constants for the Qwen3-TTS 6-model CoreML pipeline.
public enum Qwen3TtsConstants {

    // MARK: - Audio

    public static let audioSampleRate: Int = 24_000

    /// Audio samples per codec frame (80ms at 24kHz).
    public static let samplesPerFrame: Int = 1_920

    // MARK: - Model dimensions

    public static let hiddenSize: Int = 1024
    public static let numCodebooks: Int = 16
    public static let codecVocabSize: Int = 2048

    // MARK: - CodeDecoder KV cache

    /// Fixed KV cache sequence length for CodeDecoder.
    /// key_cache / value_cache shape: [1, 28672, 1, 256] float16
    public static let cdKvLen: Int = 256

    /// Consolidated KV dimension for CodeDecoder (28 layers).
    public static let cdKvDim: Int = 28_672

    // MARK: - MultiCodeDecoder KV cache

    /// Fixed KV cache sequence length for MultiCodeDecoder.
    /// key_cache / value_cache shape: [1, 5120, 1, 16] float16
    public static let mcdKvLen: Int = 16

    /// Consolidated KV dimension for MultiCodeDecoder (5 layers).
    public static let mcdKvDim: Int = 5_120

    // MARK: - Codec special token IDs

    public static let codecPadId: Int = 2148
    public static let codecBosId: Int = 2149
    public static let codecEosId: Int = 2150
    public static let codecThinkId: Int = 2154
    public static let codecNoThinkId: Int = 2155
    public static let codecThinkBosId: Int = 2156
    public static let codecThinkEosId: Int = 2157

    // MARK: - Language IDs

    public static let languageIds: [String: Int] = [
        "english": 2050,
        "chinese": 2055,
        "german": 2053,
        "italian": 2070,
        "portuguese": 2071,
        "spanish": 2054,
        "japanese": 2058,
        "korean": 2064,
        "french": 2061,
        "russian": 2069,
    ]

    // MARK: - TTS special token IDs

    public static let ttsPadTokenId: Int = 151_671
    public static let ttsBosTokenId: Int = 151_672
    public static let ttsEosTokenId: Int = 151_673

    // MARK: - Role prefix tokens

    /// [im_start, assistant, newline]
    public static let rolePrefixTokens: [Int] = [151_644, 77_091, 198]

    // MARK: - Generation parameters

    public static let maxCodecTokens: Int = 125
    public static let temperature: Float = 0.9
    public static let topK: Int = 50
    public static let repetitionPenalty: Float = 1.05
    public static let minNewTokens: Int = 2

    // MARK: - SpeechDecoder

    /// Fixed input time dimension for SpeechDecoder: [1, 16, 125].
    public static let speechDecoderFrames: Int = 125

    // MARK: - Defaults

    public static let defaultVoice: String = "default"
    public static let defaultLanguage: String = "english"
}
