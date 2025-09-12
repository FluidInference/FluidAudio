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
@available(macOS 13.0, iOS 16.0, *)
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

        // For >2 channels, use manual linear resampling since AVAudioConverter has limitations
        if inputFormat.channelCount > 2 {
            return try linearResample(buffer, to: format)
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: format) else {
            throw AudioConverterError.failedToCreateConverter
        }

        // Estimate first pass capacity and allocate
        let sampleRateRatio = format.sampleRate / inputFormat.sampleRate
        let estimatedOutputFrames = AVAudioFrameCount((Double(buffer.frameLength) * sampleRateRatio).rounded(.up))

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
        let inputSampleCount = Int(buffer.frameLength)

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

/// Errors that can occur during audio conversion
@available(macOS 13.0, iOS 16.0, *)
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
