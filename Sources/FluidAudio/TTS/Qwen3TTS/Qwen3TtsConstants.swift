import Foundation

/// Constants for the Qwen3-TTS language model TTS backend.
public enum Qwen3TtsConstants {

    // MARK: - Audio

    public static let audioSampleRate: Int = 24_000

    // MARK: - Model dimensions

    public static let hiddenSize: Int = 1024
    public static let numHeads: Int = 16
    public static let numKvHeads: Int = 8
    public static let headDim: Int = 128
    public static let numLayers: Int = 28
    public static let vocabSize: Int = 152064
    public static let numCodebooks: Int = 16
    public static let numCodeGroups: Int = 16

    // MARK: - Special token IDs

    public static let ttsBosTokenId: Int = 151672
    public static let ttsPadTokenId: Int = 151671
    public static let ttsEosTokenId: Int = 151673
    public static let codecBosTokenId: Int = 2149
    public static let codecEosTokenId: Int = 2150
    public static let codecPadTokenId: Int = 2050

    // MARK: - Language IDs

    public static let languageEnglish: Int = 2050
    public static let languageChinese: Int = 2055

    // MARK: - Role prefix tokens

    public static let rolePrefixTokens: [Int] = [151644, 77091, 198]

    // MARK: - Generation parameters

    public static let maxTextLength: Int = 128
    public static let maxCodecTokens: Int = 125

    /// CB0 (outer LM) sampling parameters
    public static let temperature: Float = 0.7
    public static let topK: Int = 30
    public static let repetitionPenalty: Float = 1.05
    public static let minNewTokens: Int = 2

    /// CB1-15 (code predictor) sampling parameters
    public static let cpTemperature: Float = 0.9
    public static let cpTopK: Int = 50

    // MARK: - KV cache

    /// Maximum KV cache length (prefill + generated tokens)
    public static let maxKvLength: Int = 200

    /// Number of KV cache entries (2 per layer: key + value)
    public static let kvCacheEntries: Int = 56

    // MARK: - Default voice

    public static let defaultVoice: String = "default"
}
