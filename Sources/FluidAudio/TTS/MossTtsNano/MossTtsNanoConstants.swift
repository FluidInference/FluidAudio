import Foundation

/// Compile-time constants for the MOSS-TTS-Nano backend.
///
/// Geometry that is baked into the published CoreML graphs
/// (`FluidInference/moss-tts-nano-coreml`). Tokenizer-derived values (prompt
/// template ids, special token ids) are read from the repo's `config.json`
/// at load time instead — see `MossTtsNanoConfig`.
public enum MossTtsNanoConstants {

    // MARK: - Audio

    /// Codec output rate. Native 48 kHz stereo.
    public static let sampleRate: Int = 48_000
    public static let channels: Int = 2
    /// One LM frame = 80 ms = 3840 samples per channel.
    public static let samplesPerFramePerChannel: Int = 3840
    public static let frameDuration: Double = 0.08

    // MARK: - Language model

    /// RVQ codebooks per frame (row width is `numCodebooks + 1`).
    public static let numCodebooks: Int = 16
    public static let rowWidth: Int = 17
    public static let hiddenSize: Int = 768
    public static let audioCodebookSize: Int = 1024
    /// Fixed prompt capacity of the published prefill graph.
    public static let prefillRows: Int = 512
    /// KV-cache capacity of the published step graph (prompt + generated frames).
    public static let maxLen: Int = 1024

    // MARK: - Generation defaults (upstream `inference()` defaults)

    public static let defaultTextTemperature: Float = 1.5
    public static let defaultAudioTemperature: Float = 1.7
    public static let defaultAudioTopP: Float = 0.8
    public static let defaultRepetitionPenalty: Float = 1.0
    public static let defaultMaxNewFrames: Int = 375
    /// Sentence-chunk token budget for voice-clone synthesis
    /// (`DEFAULT_VOICE_CLONE_MAX_TEXT_TOKENS`).
    public static let defaultMaxTextTokens: Int = 50
    /// Silence inserted between text chunks: short for ≤ 4-word chunks, long otherwise.
    public static let interChunkPauseShort: Float = 0.40
    public static let interChunkPauseLong: Float = 0.24
}
