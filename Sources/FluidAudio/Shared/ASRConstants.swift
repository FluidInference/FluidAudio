import Foundation

/// Constants for ASR audio processing and frame calculations
public enum ASRConstants {
    /// Audio sample rate expected by ASR models
    public static let sampleRate: Int = 16_000

    /// Maximum audio duration supported by CoreML encoder (seconds)
    public static let maxDurationSeconds: Double = 15.0

    /// Maximum audio samples supported by CoreML encoder (sampleRate × maxDurationSeconds)
    public static let maxModelSamples: Int = 240_000

    /// Minimum audio duration accepted by the ASR guard (seconds).
    public static let minimumAudioDurationSeconds: Double = 0.3

    /// Mel-spectrogram hop size in samples (10ms at 16kHz)
    public static let melHopSize: Int = 160

    /// Encoder subsampling factor (8x downsampling from mel frames to encoder frames)
    public static let encoderSubsampling: Int = 8

    /// Size of encoder hidden representation for Parakeet-TDT models
    public static let encoderHiddenSize: Int = 1024

    /// Size of decoder hidden state for Parakeet-TDT models
    public static let decoderHiddenSize: Int = 640

    /// Samples per encoder frame (melHopSize * encoderSubsampling)
    /// Each encoder frame represents ~80ms of audio at 16kHz
    public static let samplesPerEncoderFrame: Int = melHopSize * encoderSubsampling  // 1280

    /// Duration of one encoder frame in seconds (80ms)
    public static let secondsPerEncoderFrame: Double = Double(samplesPerEncoderFrame) / Double(sampleRate)  // 0.08

    /// WER threshold for detailed error analysis in benchmarks
    public static let highWERThreshold: Double = 0.15

    /// Punctuation token IDs (period, question mark, exclamation mark)
    public static let punctuationTokens: [Int] = [7883, 7952, 7948]

    /// Standard overlap in encoder frames (2.0s = 25 frames at 0.08s per frame)
    public static let standardOverlapFrames: Int = 25

    /// Minimum confidence score (for empty or very uncertain transcriptions)
    public static let minConfidence: Float = 0.1

    /// Maximum confidence score (perfect confidence)
    public static let maxConfidence: Float = 1.0

    /// Calculate encoder frames from audio samples using proper ceiling division
    /// - Parameter samples: Number of audio samples
    /// - Returns: Number of encoder frames
    public static func calculateEncoderFrames(from samples: Int) -> Int {
        return Int(ceil(Double(samples) / Double(samplesPerEncoderFrame)))
    }

    /// Minimum number of samples required by the ASR guard for a given sample rate.
    /// - Parameter sampleRate: Audio sample rate in Hz
    /// - Returns: Sample count corresponding to `minimumAudioDurationSeconds`
    public static func minimumRequiredSamples(forSampleRate sampleRate: Int) -> Int {
        return Int(Double(sampleRate) * minimumAudioDurationSeconds)
    }
}
