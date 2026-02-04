@preconcurrency import AVFoundation
@preconcurrency import CoreML
import FluidAudio
import Foundation
import OSLog

/// Voice cloning for PocketTTS using the Mimi encoder.
///
/// Converts audio samples to voice conditioning embeddings that can be used
/// for text-to-speech synthesis with a cloned voice.
public enum PocketTtsVoiceCloner {

    private static let logger = AppLogger(category: "PocketTtsVoiceCloner")

    // MARK: - Constants

    /// Sample rate expected by the encoder (24kHz).
    public static let sampleRate: Int = PocketTtsConstants.audioSampleRate

    /// Frame size for the encoder (1920 samples = 80ms).
    public static let frameSize: Int = PocketTtsConstants.samplesPerFrame

    /// Standard voice prompt length (125 frames).
    public static let voicePromptLength: Int = PocketTtsConstants.voicePromptLength

    /// Minimum audio duration in seconds for voice cloning.
    public static let minDurationSeconds: Double = 1.0

    /// Maximum audio duration in seconds for voice cloning.
    public static let maxDurationSeconds: Double = 30.0

    // MARK: - Voice Cloning

    /// Clone a voice from audio samples.
    ///
    /// - Parameters:
    ///   - samples: Audio samples at 24kHz mono float32.
    ///   - encoder: The Mimi encoder CoreML model.
    /// - Returns: Voice conditioning data ready for TTS.
    /// - Throws: `TTSError.processingFailed` if samples are too short or too long.
    public static func cloneVoice(
        from samples: [Float],
        using encoder: MLModel
    ) throws -> PocketTtsVoiceData {
        // Validate input
        let durationSeconds = Double(samples.count) / Double(sampleRate)
        guard durationSeconds >= minDurationSeconds else {
            throw TTSError.processingFailed(
                "Audio too short for voice cloning: \(String(format: "%.1f", durationSeconds))s "
                    + "(minimum \(minDurationSeconds)s required)"
            )
        }
        guard durationSeconds <= maxDurationSeconds else {
            throw TTSError.processingFailed(
                "Audio too long for voice cloning: \(String(format: "%.1f", durationSeconds))s "
                    + "(maximum \(maxDurationSeconds)s allowed)"
            )
        }

        // Pad audio to frame boundary
        let paddedSamples = padToFrameBoundary(samples)

        logger.info("Encoding \(paddedSamples.count) samples (\(String(format: "%.1f", durationSeconds))s)")

        // Create input tensor [1, 1, T]
        let audioArray = try MLMultiArray(shape: [1, 1, NSNumber(value: paddedSamples.count)], dataType: .float32)
        for (i, sample) in paddedSamples.enumerated() {
            audioArray[[0, 0, NSNumber(value: i)]] = NSNumber(value: sample)
        }

        // Run encoder
        let input = try MLDictionaryFeatureProvider(dictionary: ["audio": audioArray])
        let output = try encoder.prediction(from: input)

        // Get conditioning output [1, num_frames, 1024]
        guard let conditioning = output.featureValue(for: "conditioning")?.multiArrayValue else {
            throw TTSError.processingFailed("Failed to get conditioning output from encoder")
        }

        let numFrames = conditioning.shape[1].intValue
        logger.info("Encoded to \(numFrames) frames")

        // Convert to voice data format [voicePromptLength, 1024]
        let voiceData = padOrTruncate(conditioning, targetFrames: voicePromptLength)

        return PocketTtsVoiceData(audioPrompt: voiceData, promptLength: voicePromptLength)
    }

    /// Clone a voice from an audio file.
    ///
    /// Supports any audio format that AVFoundation can read (WAV, MP3, M4A, etc.).
    /// Audio is automatically converted to 24kHz mono.
    ///
    /// - Parameters:
    ///   - url: URL to the audio file.
    ///   - encoder: The Mimi encoder CoreML model.
    /// - Returns: Voice conditioning data ready for TTS.
    /// - Throws: `TTSError.processingFailed` if the file cannot be read or audio is invalid.
    public static func cloneVoice(
        from url: URL,
        using encoder: MLModel
    ) throws -> PocketTtsVoiceData {
        let samples = try loadAudio(from: url)
        return try cloneVoice(from: samples, using: encoder)
    }

    /// Save voice conditioning data to a binary file.
    ///
    /// - Parameters:
    ///   - voiceData: The voice conditioning data.
    ///   - url: Destination URL for the .bin file.
    public static func saveVoice(_ voiceData: PocketTtsVoiceData, to url: URL) throws {
        // Write as raw Float32 binary (little-endian)
        var data = Data()
        data.reserveCapacity(voiceData.audioPrompt.count * MemoryLayout<Float>.size)
        for value in voiceData.audioPrompt {
            var floatValue = value
            withUnsafeBytes(of: &floatValue) { data.append(contentsOf: $0) }
        }
        try data.write(to: url)
        logger.info("Saved voice to \(url.lastPathComponent) (\(data.count / 1024) KB)")
    }

    /// Load voice conditioning data from a binary file.
    ///
    /// - Parameters:
    ///   - url: Path to the .bin file containing voice data.
    /// - Returns: Voice conditioning data ready for TTS.
    /// - Throws: `TTSError.processingFailed` if the file cannot be read or has invalid size.
    public static func loadVoice(from url: URL) throws -> PocketTtsVoiceData {
        let data = try Data(contentsOf: url)
        let expectedSize = voicePromptLength * PocketTtsConstants.embeddingDim * MemoryLayout<Float>.size

        guard data.count == expectedSize else {
            throw TTSError.processingFailed(
                "Invalid voice file size: \(data.count) bytes (expected \(expectedSize))"
            )
        }

        var audioPrompt = [Float](repeating: 0, count: voicePromptLength * PocketTtsConstants.embeddingDim)
        data.withUnsafeBytes { buffer in
            let floatBuffer = buffer.bindMemory(to: Float.self)
            for i in 0..<audioPrompt.count {
                audioPrompt[i] = floatBuffer[i]
            }
        }

        logger.info("Loaded voice from \(url.lastPathComponent) (\(data.count / 1024) KB)")
        return PocketTtsVoiceData(audioPrompt: audioPrompt, promptLength: voicePromptLength)
    }

    // MARK: - Private Helpers

    private static func padToFrameBoundary(_ samples: [Float]) -> [Float] {
        let length = samples.count
        let padLength = (frameSize - (length % frameSize)) % frameSize
        if padLength > 0 {
            return samples + [Float](repeating: 0, count: padLength)
        }
        return samples
    }

    private static func padOrTruncate(_ conditioning: MLMultiArray, targetFrames: Int) -> [Float] {
        let numFrames = conditioning.shape[1].intValue
        let embDim = conditioning.shape[2].intValue

        var result = [Float](repeating: 0, count: targetFrames * embDim)

        let framesToCopy = min(numFrames, targetFrames)
        for frame in 0..<framesToCopy {
            for dim in 0..<embDim {
                let idx = frame * embDim + dim
                let value = conditioning[[0, NSNumber(value: frame), NSNumber(value: dim)]].floatValue
                result[idx] = value
            }
        }

        if numFrames > targetFrames {
            logger.info("Truncated from \(numFrames) to \(targetFrames) frames")
        } else if numFrames < targetFrames {
            logger.info("Padded from \(numFrames) to \(targetFrames) frames")
        }

        return result
    }

    /// Load audio from a file and convert to 24kHz mono Float32.
    ///
    /// Uses AudioConverter for high-quality resampling via AVAudioConverter.
    private static func loadAudio(from url: URL) throws -> [Float] {
        // Create AudioConverter targeting 24kHz mono (PocketTTS requirement)
        guard
            let targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(sampleRate),
                channels: 1,
                interleaved: false
            )
        else {
            throw TTSError.processingFailed("Failed to create target audio format")
        }

        let converter = AudioConverter(targetFormat: targetFormat)

        do {
            let samples = try converter.resampleAudioFile(url)

            guard !samples.isEmpty else {
                throw TTSError.processingFailed("Audio file contains no samples")
            }

            return samples
        } catch {
            throw TTSError.processingFailed("Failed to load audio file: \(error.localizedDescription)")
        }
    }
}
