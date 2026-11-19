import Foundation

/// Constants for Nemotron 3.5 Transcribe Lite: NVIDIA Nemotron 3.5 Transcribe's ASR encoder and
/// CTC timestamp head, without the LLM.
///
/// The encoder runs NeMo's long-form layout: 40 s cores with 4 s of context on each side, one
/// 48 s window per model call. Encoder frames are 80 ms (8 stacked 10 ms mel frames); the CTC head
/// upsamples them to 40 ms.
public enum NemotronTranscribeLiteConfig {
    public static let sampleRate = 16_000
    public static let hopLength = 160
    public static let nFFT = 512
    public static let winLength = 400
    public static let nMels = 128
    /// Mel frames stacked into one encoder frame.
    public static let stack = 8

    /// Mel frames per window: 4000 core + 400 context on each side.
    public static let coreMelFrames = 4000
    public static let contextMelFrames = 400
    public static let windowMelFrames = coreMelFrames + 2 * contextMelFrames

    /// Encoder frames per CTC-head window: 500 core + 50 context on each side.
    public static let coreEncoderFrames = 500
    public static let contextEncoderFrames = 50
    public static let windowEncoderFrames = coreEncoderFrames + 2 * contextEncoderFrames

    public static let encoderChannels = 1280
    public static let encoderFrameSeconds = 0.08
    public static let ctcFrameSeconds = 0.04
    /// CTC blank: the class after the 23,862 SentencePiece pieces.
    public static let blankId = 23_862
    /// SentencePiece word-start marker.
    public static let wordStart: Character = "\u{2581}"

    /// Speaker attribution (with Nemotron-3-Diarization): activity threshold, the minimum total speech
    /// (in 80 ms frames, 2 s) for a diarizer slot to count as a speaker, and how far (in 80 ms frames,
    /// ~1 s) to look around a word for an active speaker.
    public static let speakerActivityThreshold: Float = 0.5
    public static let minSpeakerFrames = 25
    public static let speakerSearchFrames = 12
}
