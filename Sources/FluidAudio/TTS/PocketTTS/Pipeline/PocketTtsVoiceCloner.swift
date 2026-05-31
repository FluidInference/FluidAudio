import Accelerate
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

    /// Fixed encoder input length in samples (10s @ 24kHz). `mimi_encoderv2` has
    /// `hasShapeFlexibility: "0"` and accepts exactly this many samples.
    public static let encoderInputSamples: Int = 240_000

    /// Maximum voice prompt frames produced by the encoder for one forward pass
    /// (`encoderInputSamples / frameSize`). The encoder output shape is fixed at
    /// `[1, 125, 1024]`, so 125 is the hard ceiling.
    public static let maxVoiceFrames: Int = 125

    /// Minimum audio duration in seconds for voice cloning.
    public static let minDurationSeconds: Double = 1.0

    /// Maximum audio duration in seconds for voice cloning (matches
    /// `encoderInputSamples`). Audio longer than this is truncated.
    public static let maxDurationSeconds: Double = 10.0

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

        // mimi_encoderv2 has a fixed input shape [1, 1, 240000]. Pad shorter
        // audio with zeros; truncate longer audio. Track the real sample count
        // so we can drop encoded-zero-padding frames from the output.
        let realSampleCount = min(samples.count, encoderInputSamples)
        let encoderInput = makeEncoderInputBuffer(samples)

        logger.info(
            "Encoding \(realSampleCount) samples (\(String(format: "%.1f", durationSeconds))s) "
                + "padded/truncated to \(encoderInputSamples)"
        )

        // Create input tensor [1, 1, 240000]
        let audioArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: encoderInputSamples)], dataType: .float32)
        let dst = audioArray.dataPointer.bindMemory(to: Float.self, capacity: encoderInputSamples)
        encoderInput.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: encoderInputSamples)
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
        let usableFrames = usableFrameCount(
            realSampleCount: realSampleCount, availableFrames: numFrames)
        logger.info("Encoded to \(numFrames) frames, using \(usableFrames)")

        // Extract conditioning, honoring the array's strides (no zero-padding).
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

    /// Build a fixed-length `encoderInputSamples`-sized buffer: copy the first
    /// `encoderInputSamples` of `samples` (truncating overflow), zero-pad the
    /// remainder. `mimi_encoderv2`'s input shape is non-flexible at runtime.
    ///
    /// Exposed at internal access for unit tests; production callers go
    /// through `cloneVoice(from:using:)`.
    static func makeEncoderInputBuffer(_ samples: [Float]) -> [Float] {
        var buffer = [Float](repeating: 0, count: encoderInputSamples)
        let copyCount = min(samples.count, encoderInputSamples)
        if copyCount > 0 {
            buffer.replaceSubrange(0..<copyCount, with: samples[0..<copyCount])
        }
        return buffer
    }

    /// Number of encoder output frames that correspond to real (non-padded)
    /// audio. Drops trailing frames covering the zero-padded tail; rounds up
    /// so the last partial real frame still contributes voice content.
    /// Capped by both the encoder's actual frame output and `maxVoiceFrames`.
    ///
    /// Exposed at internal access for unit tests.
    static func usableFrameCount(realSampleCount: Int, availableFrames: Int) -> Int {
        let realFrames = (realSampleCount + frameSize - 1) / frameSize
        return min(availableFrames, realFrames, maxVoiceFrames)
    }

    /// Extract conditioning floats from MLMultiArray `[1, frames, embDim]`
    /// into packed row-major `[frames * embDim]`.
    ///
    /// CoreML can return the `conditioning` array strided / non-contiguous
    /// (padding between frames, or `dimStride != 1`), so we read using the
    /// array's reported strides rather than assuming packed storage. Reading
    /// a strided buffer as if it were contiguous scrambles the embedding
    /// order and produces clipped / clicky cloned audio (see FluidAudio
    /// #612).
    ///
    /// A genuinely contiguous array (`dimStride == 1 && frameStride == embDim`)
    /// keeps the fast bulk path: a single `UnsafeBufferPointer` copy for
    /// Float32, or vectorized `vDSP.convertElements` (fp16→fp32) for Float16,
    /// avoiding 128 k MLMultiArray subscript calls per clone. A strided array
    /// falls back to stride-aware pointer arithmetic; cloning runs once per
    /// voice (not in the generation loop), so the per-element copy is cheap.
    /// On x86 (no Swift `Float16`) the fp16 path routes through NSNumber
    /// subscripting, which is stride-correct by construction.
    ///
    /// Exposed at internal access for unit tests.
    static func extractConditioning(
        _ conditioning: MLMultiArray, frames: Int, embDim: Int
    ) -> [Float] {
        let count = frames * embDim
        let strides = conditioning.strides.map { $0.intValue }
        let frameStride = strides.count >= 3 ? strides[1] : embDim
        let dimStride = strides.count >= 3 ? strides[2] : 1
        let isContiguous = (dimStride == 1 && frameStride == embDim)
        // Highest element index reachable under the reported strides.
        let lastIndex = max(0, (frames - 1) * frameStride + (embDim - 1) * dimStride)

        if conditioning.dataType == .float16 {
            var result = [Float](repeating: 0, count: count)
            #if arch(arm64)
            let srcPtr = conditioning.dataPointer.bindMemory(
                to: Float16.self, capacity: lastIndex + 1)
            if isContiguous {
                let srcBuffer = UnsafeBufferPointer(start: srcPtr, count: count)
                result.withUnsafeMutableBufferPointer { dst in
                    vDSP.convertElements(of: srcBuffer, to: &dst)
                }
            } else {
                for frame in 0..<frames {
                    let base = frame * frameStride
                    for dim in 0..<embDim {
                        result[frame * embDim + dim] = Float(srcPtr[base + dim * dimStride])
                    }
                }
            }
            #else
            // x86: Swift Float16 unavailable. NSNumber subscripting is stride-safe.
            for frame in 0..<frames {
                for dim in 0..<embDim {
                    result[frame * embDim + dim] =
                        conditioning[[0, NSNumber(value: frame), NSNumber(value: dim)]]
                        .floatValue
                }
            }
            #endif
            return result
        }

        // Float32
        let srcPtr = conditioning.dataPointer.bindMemory(to: Float.self, capacity: lastIndex + 1)
        if isContiguous {
            return Array(UnsafeBufferPointer(start: srcPtr, count: count))
        }
        var result = [Float](repeating: 0, count: count)
        for frame in 0..<frames {
            let base = frame * frameStride
            for dim in 0..<embDim {
                result[frame * embDim + dim] = srcPtr[base + dim * dimStride]
            }
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
