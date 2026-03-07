@preconcurrency import AVFoundation
@preconcurrency import CoreML
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

    /// Maximum voice prompt frames (caps at ~20s to leave KV cache room for text tokens).
    public static let maxVoiceFrames: Int = 250

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
    /// - Throws: `PocketTTSError.processingFailed` if samples are too short or too long.
    public static func cloneVoice(
        from samples: [Float],
        using encoder: MLModel
    ) throws -> PocketTtsVoiceData {
        // Validate input
        let durationSeconds = Double(samples.count) / Double(sampleRate)
        guard durationSeconds >= minDurationSeconds else {
            throw PocketTTSError.processingFailed(
                "Audio too short for voice cloning: \(String(format: "%.1f", durationSeconds))s "
                    + "(minimum \(minDurationSeconds)s required)"
            )
        }
        guard durationSeconds <= maxDurationSeconds else {
            throw PocketTTSError.processingFailed(
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
            throw PocketTTSError.processingFailed("Failed to get conditioning output from encoder")
        }

        let numFrames = conditioning.shape[1].intValue
        let embDim = conditioning.shape[2].intValue
        let usableFrames = min(numFrames, maxVoiceFrames)
        logger.info("Encoded to \(numFrames) frames, using \(usableFrames)")

        // Extract conditioning with bulk memory copy (no zero-padding)
        let totalFloats = usableFrames * embDim
        let voiceData = extractConditioning(conditioning, frames: usableFrames, embDim: embDim)

        guard voiceData.count == totalFloats else {
            throw PocketTTSError.processingFailed(
                "Conditioning extraction mismatch: got \(voiceData.count), expected \(totalFloats)")
        }

        return PocketTtsVoiceData(audioPrompt: voiceData, promptLength: usableFrames)
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
    /// - Throws: `PocketTTSError.processingFailed` if the file cannot be read or audio is invalid.
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
    /// Supports variable-length voice prompts — the prompt length is derived
    /// from the file size (`floatCount / embeddingDim`).
    ///
    /// - Parameters:
    ///   - url: Path to the .bin file containing voice data.
    /// - Returns: Voice conditioning data ready for TTS.
    /// - Throws: `PocketTTSError.processingFailed` if the file cannot be read or has invalid size.
    public static func loadVoice(from url: URL) throws -> PocketTtsVoiceData {
        let data = try Data(contentsOf: url)
        let embDim = PocketTtsConstants.embeddingDim
        let floatCount = data.count / MemoryLayout<Float>.size

        guard floatCount > 0, floatCount % embDim == 0 else {
            throw PocketTTSError.processingFailed(
                "Invalid voice file size: \(data.count) bytes (not divisible by embedding dim \(embDim))"
            )
        }

        let promptLength = floatCount / embDim

        guard promptLength <= maxVoiceFrames else {
            throw PocketTTSError.processingFailed(
                "Voice file too large: \(promptLength) frames (max \(maxVoiceFrames))"
            )
        }

        let audioPrompt = data.withUnsafeBytes { rawBuffer in
            let floatBuffer = rawBuffer.bindMemory(to: Float.self)
            return Array(floatBuffer)
        }

        logger.info(
            "Loaded voice from \(url.lastPathComponent): \(promptLength) frames (\(data.count / 1024) KB)")
        return PocketTtsVoiceData(audioPrompt: audioPrompt, promptLength: promptLength)
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

    /// Extract conditioning floats from MLMultiArray [1, frames, embDim] via bulk memory copy.
    private static func extractConditioning(
        _ conditioning: MLMultiArray, frames: Int, embDim: Int
    ) -> [Float] {
        let count = frames * embDim
        if conditioning.dataType == .float16 {
            return (0..<count).map { i in
                let frame = i / embDim
                let dim = i % embDim
                return conditioning[[0, NSNumber(value: frame), NSNumber(value: dim)]].floatValue
            }
        }
        // Fast path: float32 bulk copy
        let srcPtr = conditioning.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: srcPtr, count: count))
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
            throw PocketTTSError.processingFailed("Failed to create target audio format")
        }

        let converter = AudioConverter(targetFormat: targetFormat)

        do {
            let samples = try converter.resampleAudioFile(url)

            guard !samples.isEmpty else {
                throw PocketTTSError.processingFailed("Audio file contains no samples")
            }

            return samples
        } catch {
            throw PocketTTSError.processingFailed("Failed to load audio file: \(error.localizedDescription)")
        }
    }
}
