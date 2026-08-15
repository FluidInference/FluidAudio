import Foundation

/// Compile-time constants for the LuxTTS (ZipVoice-Distill) backend.
///
/// Values mirror the upstream LuxTTS inference defaults and the fixed-shape
/// buckets baked into the CoreML graphs at
/// `FluidInference/luxtts-coreml` (see the repo README for the bucket table).
public enum LuxTtsConstants {

    /// Sample rate of the mel frontend / prompt conditioning (Hz).
    public static let melSampleRate = 24000
    /// Sample rate of the generated waveform (Hz) — the vocoder upsamples
    /// 24 kHz mel frames to 48 kHz audio in-graph.
    public static let outputSampleRate = 48000

    // Mel frontend (upstream VocosFbank: torchaudio MelSpectrogram, power=1).
    public static let nFFT = 1024
    public static let hopLength = 256
    public static let nMels = 100
    /// Log floor: `mel.clamp(min: 1e-7).log()`.
    public static let logMelFloor: Float = 1e-7
    /// Features are scaled by 0.1 before conditioning (`feat_scale`).
    public static let featScale: Float = 0.1

    // Fixed CoreML shape buckets (gpu/ + ane/ graphs).
    public static let maxTokens = 256
    public static let maxFrames = 1024
    public static let featDim = 100

    // Flow-matching solver.
    public static let numSteps = 4
    public static let tShift = 0.5
    public static let guidanceScale: Float = 3.0

    /// Default speech-rate divisor. Upstream `generate()` silently multiplies
    /// speed by 1.3, which squeezes the ratio-based duration estimate and
    /// clips sentence onsets; 1.0 synthesizes complete sentences.
    public static let defaultSpeed: Float = 1.0

    /// Prompt RMS normalization target (upstream `rms_norm`). Prompts quieter
    /// than this are boosted before mel extraction and the generated waveform
    /// is scaled back down by the same factor.
    public static let targetRms: Float = 0.1

    /// Prompt duration cap in seconds. Frames beyond this would eat too much
    /// of the 1024-frame bucket (~10.9 s total at 93.75 frames/s).
    public static let maxPromptSeconds: Double = 5.0

    /// Published fixed-shape vocoder buckets (generated frames).
    public static let vocoderBuckets = [282, 555]
    /// Vocoder hop at 48 kHz (256 at 24 kHz × 2). The vocoder emits
    /// `(bucket - 1) * hop48k` samples.
    public static let hop48k = 512

    /// Default synthesis noise seed (matches the Python reference scripts).
    public static let defaultSeed: UInt64 = 42
}
