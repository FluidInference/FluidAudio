import Foundation

/// Constants for ASR audio processing and frame calculations
public enum ASRConstants {
    /// Audio sample rate expected by ASR models
    public static let sampleRate: Int = 16_000

    /// Maximum audio duration supported by CoreML encoder (seconds)
    public static let maxDurationSeconds: Double = 15.0

    /// Maximum audio samples supported by CoreML encoder (sampleRate × maxDurationSeconds)
    public static let maxModelSamples: Int = 240_000

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

    /// WER threshold for detailed error analysis in benchmarks
    public static let highWERThreshold: Double = 0.15

    /// Calculate encoder frames from audio samples using proper ceiling division
    /// - Parameter samples: Number of audio samples
    /// - Returns: Number of encoder frames
    public static func calculateEncoderFrames(from samples: Int) -> Int {
        return Int(ceil(Double(samples) / Double(samplesPerEncoderFrame)))
    }
}
