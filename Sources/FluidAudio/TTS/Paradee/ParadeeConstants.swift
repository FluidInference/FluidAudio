import Foundation

/// Fixed pipeline parameters for the Paradee-8M CoreML backend. Values mirror
/// `FluidInference/paradee-8m-coreml/config.json` and the upstream
/// `paradee/tts.py`.
public enum ParadeeConstants {

    /// Output sample rate (24 kHz mono).
    public static let sampleRate = 24_000

    /// Output samples per aligned frame: each frame carries two F0 steps of
    /// 300 samples (`prod(upsample_rates) * istft_hop`).
    public static let samplesPerFrame = 600

    /// Upper bound of the acoustic model's frame axis (100 s at speed 1).
    public static let maxFrames = 4_000

    /// Phonemes per synthesis call; the text side has 512 positions, two of
    /// which hold the pad token at each end.
    public static let maxPhonemeLength = 510

    /// Channels of the prosody features `d` (hidden 192 + style 32).
    public static let prosodyChannels = 224

    /// Channels of the text features `asr_tok` (the teacher decoder's width).
    public static let textChannels = 512

    /// Speech-rate multiplier. Durations scale by `1 / speed`.
    public static let defaultSpeed: Float = 1.0
}
