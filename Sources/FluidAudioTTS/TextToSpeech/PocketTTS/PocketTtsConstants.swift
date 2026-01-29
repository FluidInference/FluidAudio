import Foundation

/// Constants for the PocketTTS flow-matching language model TTS backend.
public enum PocketTtsConstants {

    // MARK: - Audio

    public static let audioSampleRate: Int = 24_000
    public static let samplesPerFrame: Int = 1_920
    public static let frameDurationMs: Double = 80.0

    // MARK: - Model dimensions

    public static let latentDim: Int = 32
    public static let transformerDim: Int = 1024
    public static let vocabSize: Int = 4001
    public static let embeddingDim: Int = 1024

    // MARK: - Generation parameters

    public static let numLsdSteps: Int = 8
    public static let temperature: Float = 0.7
    public static let eosThreshold: Float = -4.0
    public static let shortTextPadFrames: Int = 3
    public static let longTextExtraFrames: Int = 1
    public static let extraFramesAfterDetection: Int = 2
    public static let shortTextWordThreshold: Int = 5
    public static let maxTokensPerChunk: Int = 50

    // MARK: - KV cache

    public static let kvCacheLayers: Int = 6
    public static let kvCacheMaxLen: Int = 512
    public static let kvCacheHeads: Int = 16
    public static let kvCacheHeadDim: Int = 64

    // MARK: - Voice

    public static let defaultVoice: String = "alba"
    public static let voicePromptLength: Int = 125

    // MARK: - Repository

    public static let defaultRepository: String = "alexwengg/pocket-tts-coreml"
    public static let defaultModelsSubdirectory: String = "Models"
}
