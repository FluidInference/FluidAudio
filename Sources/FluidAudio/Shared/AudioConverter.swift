import AVFoundation
import Accelerate
import Foundation
import OSLog

/// Converts audio buffers to the format required by ASR (16kHz, mono, Float32).
///
/// Implementation notes:
/// - Uses `AVAudioConverter` for all sample-rate, sample-format, and channel-count conversions.
/// - Avoids any manual resampling; only raw sample extraction occurs after conversion.
/// - Creates a new converter for each operation (stateless).
final public class AudioConverter {
    private let logger = AppLogger(category: "AudioConverter")
    private let targetFormat: AVAudioFormat

    /// Public initializer so external modules (e.g. CLI) can construct the converter
    public init(targetFormat: AVAudioFormat? = nil) {
        if let format = targetFormat {
            self.targetFormat = format
        } else {
            /// Most audio models expect this format.
            /// Target format for ASR, Speaker diarization model: 16kHz, mono
            self.targetFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16000,
                channels: 1,
                interleaved: false
            )!
        }
    }

    /// Convert a standalone buffer to the target format.
    /// - Parameter buffer: Input audio buffer (any format)
    /// - Returns: Float array at 16kHz mono
    public func resampleBuffer(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        // Fast path: if already in target format, just extract samples
        if isTargetFormat(buffer.format) {
            return extractFloatArray(from: buffer)
        }

        return try convertBuffer(buffer, to: targetFormat)
    }

    /// Convert an audio file to 16kHz mono Float32 samples
    /// - Parameter url: URL of the audio file to read
    /// - Returns: Float array at 16kHz mono
    public func resampleAudioFile(_ url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        try audioFile.read(into: buffer)
        return try resampleBuffer(buffer)
    }

    /// Convert an audio file path to 16kHz mono Float32 samples
    /// - Parameter path: File path of the audio file to read
    /// - Returns: Float array at 16kHz mono
    public func resampleAudioFile(path: String) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        return try resampleAudioFile(url)
    }

    /// Convert a buffer to the target format.
    private func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
        let inputFormat = buffer.format

        // Check if we need to normalize problematic formats
        let needsNormalization = needsAudioNormalization(inputFormat)

        let bufferToConvert: AVAudioPCMBuffer
        if needsNormalization {
            logger.info("Normalizing audio: \(inputFormat) → converting to clean mono format")
            bufferToConvert = try normalizeAudio(buffer)
        } else {
            bufferToConvert = buffer
        }

        // For >2 channels, use manual linear resampling since AVAudioConverter has limitations
        if bufferToConvert.format.channelCount > 2 {
            return try linearResample(bufferToConvert, to: format)
        }

        guard let converter = AVAudioConverter(from: bufferToConvert.format, to: format) else {
            throw AudioConverterError.failedToCreateConverter
        }

        // Estimate first pass capacity and allocate
        let inputSampleRate = bufferToConvert.format.sampleRate
        let sampleRateRatio = format.sampleRate / inputSampleRate
        let estimatedOutputFrames = AVAudioFrameCount(
            (Double(bufferToConvert.frameLength) * sampleRateRatio).rounded(.up))

        func makeOutputBuffer(_ capacity: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
            guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
                throw AudioConverterError.failedToCreateBuffer
            }
            return out
        }

        var aggregated: [Float] = []
        aggregated.reserveCapacity(Int(estimatedOutputFrames))

        // Provide input once, then signal end-of-stream
        var provided = false
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            if !provided {
                provided = true
                status.pointee = .haveData
                return buffer
            } else {
                status.pointee = .endOfStream
                return nil
            }
        }

        var error: NSError?
        let inputSampleCount = Int(bufferToConvert.frameLength)

        // First pass: convert main data
        let firstOut = try makeOutputBuffer(estimatedOutputFrames)
        let firstStatus = converter.convert(to: firstOut, error: &error, withInputFrom: inputBlock)
        guard firstStatus != .error else { throw AudioConverterError.conversionFailed(error) }
        if firstOut.frameLength > 0 { aggregated.append(contentsOf: extractFloatArray(from: firstOut)) }

        // Drain remaining frames until EOS
        while true {
            let out = try makeOutputBuffer(4096)
            let status = converter.convert(to: out, error: &error, withInputFrom: inputBlock)
            guard status != .error else { throw AudioConverterError.conversionFailed(error) }
            if out.frameLength > 0 { aggregated.append(contentsOf: extractFloatArray(from: out)) }
            if status == .endOfStream { break }
        }

        let outputSampleCount = aggregated.count
        logger.debug(
            "Audio conversion: \(inputSampleCount) samples → \(outputSampleCount) samples, ratio: \(Double(outputSampleCount)/Double(inputSampleCount))"
        )

        return aggregated
    }

    /// Check if audio format needs normalization for diarization to work properly.
    /// Problematic formats include stereo audio, non-standard sample rates,
    /// and formats that confuse AVAudioConverter.
    private func needsAudioNormalization(_ format: AVAudioFormat) -> Bool {
        // Check if stereo (diarization models expect mono)
        if format.channelCount > 1 {
            logger.debug("Audio has \(format.channelCount) channels, normalizing to mono")
            return true
        }

        // Check if sample rate is significantly different from target
        let sampleRateRatio = format.sampleRate / targetFormat.sampleRate
        if sampleRateRatio > 3.0 || sampleRateRatio < 0.3 {
            logger.debug(
                "Sample rate \(format.sampleRate)Hz significantly differs from target \(targetFormat.sampleRate)Hz")
            return true
        }

        // Check for non-PCM formats that might cause issues
        if format.commonFormat != .pcmFormatFloat32 && format.commonFormat != .pcmFormatInt16 {
            logger.debug("Non-PCM format detected: \(format.commonFormat), normalizing")
            return true
        }

        return false
    }

    /// Normalize audio buffer to ensure compatibility with diarization models.
    /// Converts to mono, ensures clean PCM format, and removes problematic metadata.
    private func normalizeAudio(_ buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        let inputFormat = buffer.format

        // Create target format: mono, clean PCM
        let normalizedFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: inputFormat.sampleRate,
            channels: 1,
            interleaved: false
        )!

        guard let converter = AVAudioConverter(from: inputFormat, to: normalizedFormat) else {
            throw AudioConverterError.failedToCreateConverter
        }

        // Allocate output buffer
        let outputCapacity = AVAudioFrameCount(
            Double(buffer.frameLength) * (normalizedFormat.sampleRate / inputFormat.sampleRate))
        guard let outputBuffer = AVAudioPCMBuffer(pcmFormat: normalizedFormat, frameCapacity: outputCapacity) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        // Perform conversion
        var error: NSError?
        let status = converter.convert(to: outputBuffer, error: &error) { _, status in
            status.pointee = .haveData
            return buffer
        }

        guard status != .error else {
            throw AudioConverterError.conversionFailed(error)
        }

        logger.debug(
            "Normalized audio: \(inputFormat.channelCount)→\(normalizedFormat.channelCount) channels, \(inputFormat.sampleRate)→\(normalizedFormat.sampleRate)Hz"
        )
        return outputBuffer
    }

    /// Check if a format already matches the target output format.
    private func isTargetFormat(_ format: AVAudioFormat) -> Bool {
        return format.sampleRate == targetFormat.sampleRate
            && format.channelCount == targetFormat.channelCount
            && format.commonFormat == targetFormat.commonFormat
            && format.isInterleaved == targetFormat.isInterleaved
    }

    /// Resample high channel count audio (>2 channels) using linear interpolation
    /// AVAudioConverter has limitations with >2 channels, so we handle it via a linear resample. Accuracy may not be as good as AVAudioConverter.
    /// But this is needed for applications like Safari on speaker mode, or for particular hardware devices.
    private func linearResample(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
        let inputFormat = buffer.format
        guard let channelData = buffer.floatChannelData else {
            throw AudioConverterError.failedToCreateBuffer
        }

        let inputFrameCount = Int(buffer.frameLength)
        let channelCount = Int(inputFormat.channelCount)

        // Step 1: Mix down to mono
        var monoSamples = [Float](repeating: 0, count: inputFrameCount)
        let channelWeight = 1.0 / Float(channelCount)

        for frame in 0..<inputFrameCount {
            var sum: Float = 0
            for channel in 0..<channelCount {
                sum += channelData[channel][frame]
            }
            monoSamples[frame] = sum * channelWeight
        }

        // Step 2: Resample if needed
        let inputSampleRate = inputFormat.sampleRate
        let targetSampleRate = format.sampleRate

        if inputSampleRate == targetSampleRate {
            return monoSamples
        }

        // Linear interpolation resampling
        let resampleRatio = inputSampleRate / targetSampleRate
        let outputFrameCount = Int(Double(inputFrameCount) / resampleRatio)
        var outputSamples = [Float](repeating: 0, count: outputFrameCount)

        for i in 0..<outputFrameCount {
            let sourceIndex = Double(i) * resampleRatio
            let index = Int(sourceIndex)
            let fraction = Float(sourceIndex - Double(index))

            if index < inputFrameCount - 1 {
                // Linear interpolation between samples
                outputSamples[i] = monoSamples[index] * (1.0 - fraction) + monoSamples[index + 1] * fraction
            } else if index < inputFrameCount {
                outputSamples[i] = monoSamples[index]
            }
        }

        logger.debug(
            "Manual resampling: \(channelCount) channels → mono, \(inputSampleRate)Hz → \(targetSampleRate)Hz"
        )

        return outputSamples
    }

    /// Extract Float array from PCM buffer
    private func extractFloatArray(from buffer: AVAudioPCMBuffer) -> [Float] {
        // This function assumes mono, non-interleaved Float32 buffers.
        // All multi-channel or interleaved inputs should be converted via AVAudioConverter first.
        guard let channelData = buffer.floatChannelData else { return [] }

        let frameCount = Int(buffer.frameLength)
        if frameCount == 0 { return [] }

        // Enforce mono; converter guarantees this in normal flow.
        assert(buffer.format.channelCount == 1, "extractFloatArray expects mono buffers")

        // Fast copy using vDSP (equivalent to memcpy for contiguous Float32)
        let out = [Float](unsafeUninitializedCapacity: frameCount) { dest, initialized in
            vDSP_mmov(
                channelData[0],
                dest.baseAddress!,
                vDSP_Length(frameCount),
                1,
                1,
                1
            )
            initialized = frameCount
        }
        return out
    }

}

// MARK: - WAV Utilities (shared by TTS/ASR)
public enum AudioWAV {
    /// Convert float samples to 16-bit PCM mono WAV at the given sample rate.
    public static func data(from samples: [Float], sampleRate: Double) throws -> Data {
        // Normalize to [-1, 1]
        let maxVal = samples.map { abs($0) }.max() ?? 1.0
        let norm = maxVal > 0 ? samples.map { $0 / maxVal } : samples

        // Convert to 16-bit PCM
        var pcm = Data()
        pcm.reserveCapacity(norm.count * MemoryLayout<Int16>.size)
        for s in norm {
            let clipped = max(-1.0, min(1.0, s))
            let v = Int16(clipped * 32767)
            var le = v.littleEndian
            withUnsafeBytes(of: &le) { pcm.append(contentsOf: $0) }
        }

        // Build WAV header
        var wav = Data()
        // RIFF header
        wav.append(contentsOf: "RIFF".data(using: .ascii)!)
        var fileSize = UInt32(36 + pcm.count).littleEndian
        withUnsafeBytes(of: &fileSize) { wav.append(contentsOf: $0) }
        wav.append(contentsOf: "WAVE".data(using: .ascii)!)

        // fmt chunk
        wav.append(contentsOf: "fmt ".data(using: .ascii)!)
        var subchunk1Size = UInt32(16).littleEndian  // PCM
        withUnsafeBytes(of: &subchunk1Size) { wav.append(contentsOf: $0) }
        var audioFormat = UInt16(1).littleEndian  // PCM
        withUnsafeBytes(of: &audioFormat) { wav.append(contentsOf: $0) }
        var numChannels = UInt16(1).littleEndian
        withUnsafeBytes(of: &numChannels) { wav.append(contentsOf: $0) }
        var sr = UInt32(sampleRate).littleEndian
        withUnsafeBytes(of: &sr) { wav.append(contentsOf: $0) }
        var byteRate = UInt32(sampleRate * 2).littleEndian  // 16-bit mono
        withUnsafeBytes(of: &byteRate) { wav.append(contentsOf: $0) }
        var blockAlign = UInt16(2).littleEndian
        withUnsafeBytes(of: &blockAlign) { wav.append(contentsOf: $0) }
        var bitsPerSample = UInt16(16).littleEndian
        withUnsafeBytes(of: &bitsPerSample) { wav.append(contentsOf: $0) }

        // data chunk
        wav.append(contentsOf: "data".data(using: .ascii)!)
        var dataSize = UInt32(pcm.count).littleEndian
        withUnsafeBytes(of: &dataSize) { wav.append(contentsOf: $0) }
        wav.append(pcm)

        return wav
    }
}

/// Errors that can occur during audio conversion
public enum AudioConverterError: LocalizedError {
    case failedToCreateConverter
    case failedToCreateBuffer
    case conversionFailed(Error?)

    public var errorDescription: String? {
        switch self {
        case .failedToCreateConverter:
            return "Failed to create audio converter"
        case .failedToCreateBuffer:
            return "Failed to create conversion buffer"
        case .conversionFailed(let error):
            return "Audio conversion failed: \(error?.localizedDescription ?? "Unknown error")"
        }
    }
}
