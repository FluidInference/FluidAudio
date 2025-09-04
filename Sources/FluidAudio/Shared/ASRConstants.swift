import Foundation

/// Constants for ASR audio processing and frame calculations
public enum ASRConstants {
    /// Mel-spectrogram hop size in samples (10ms at 16kHz)
    public static let melHopSize: Int = 160

    /// Encoder subsampling factor (8x downsampling from mel frames to encoder frames)
    public static let encoderSubsampling: Int = 8

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
