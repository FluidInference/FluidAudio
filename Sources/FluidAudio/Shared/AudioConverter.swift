import AVFoundation
import Accelerate
import Foundation
import OSLog

/// Converts audio buffers to the format required by ASR (16kHz, mono, Float32).
///
/// Implementation notes:
/// - Uses `AVAudioConverter` for all sample-rate, sample-format, and channel-count conversions.
/// - Avoids any manual resampling; only raw sample extraction occurs after conversion.
/// - Supports streaming by retaining the converter between calls and deferring drain until finish.
@available(macOS 13.0, iOS 16.0, *)
final public class AudioConverter {
    private let logger = AppLogger(category: "AudioConverter")
    private let targetFormat: AVAudioFormat;

    private var converter: AVAudioConverter?

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

    /// Convert an AVAudioPCMBuffer to target Float array.
    /// - Parameters:
    ///   - buffer: Input audio buffer (any format)
    ///   - streaming: Use `true` for live/streaming chunks (no flush), `false` for standalone buffers (flush to EOS)
    /// - Returns: Float array at 16kHz mono
    public func resampleAudioBuffer(_ buffer: AVAudioPCMBuffer, streaming: Bool = false) throws -> [Float] {
        // Fast path: if already in target format, just extract samples
        if isTargetFormat(buffer.format) {
            return extractFloatArray(from: buffer)
        }

        // Route to a clear, single-purpose path
        if streaming {
            return try convertStreamingChunk(buffer, to: targetFormat)
        } else {
            return try convertBatchBuffer(buffer, to: targetFormat)
        }
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
        return try resampleAudioBuffer(buffer, streaming: false)
    }

    /// Convert an audio file path to 16kHz mono Float32 samples
    /// - Parameter path: File path of the audio file to read
    /// - Returns: Float array at 16kHz mono
    public func resampleAudioFile(path: String) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        return try resampleAudioFile(url)
    }

    /// Finish a streaming conversion sequence and flush any remaining samples.
    /// - Returns: Additional samples produced by draining the converter, if any
    public func finishStreamingConversion() throws -> [Float] {
        guard let converter = converter else { return [] }

        // For streaming we never re-prime; just drain remaining frames
        converter.primeMethod = .none

        var drained: [Float] = []
        var error: NSError?

        func makeOutputBuffer(_ capacity: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
            guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else {
                throw AudioConverterError.failedToCreateBuffer
            }
            return out
        }

        while true {
            let outBuf = try makeOutputBuffer(4096)
            let status = converter.convert(to: outBuf, error: &error) { _, statusPointer in
                statusPointer.pointee = .endOfStream
                return nil
            }
            guard status != .error else { throw AudioConverterError.conversionFailed(error) }
            if outBuf.frameLength > 0 {
                drained.append(contentsOf: extractFloatArray(from: outBuf))
            }
            if status == .endOfStream { break }
        }

        return drained
    }

    /// Convert a standalone buffer to the target format (flush to end-of-stream).
    private func convertBatchBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
        let inputFormat = buffer.format
        try ensureConverter(input: inputFormat, output: format)
        guard let converter = converter else { throw AudioConverterError.failedToCreateConverter }

        // Normal priming for full, standalone conversions
        converter.primeMethod = .normal

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

        return aggregated
    }

    /// Convert a streaming chunk to the target format (no flush/drain).
    private func convertStreamingChunk(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> [Float] {
        let inputFormat = buffer.format
        try ensureConverter(input: inputFormat, output: format)
        guard let converter = converter else { throw AudioConverterError.failedToCreateConverter }

        // For streaming, avoid priming and do a single pass without draining
        converter.primeMethod = .none

        // Estimate immediate output
        let sampleRateRatio = format.sampleRate / inputFormat.sampleRate
        let estimate = AVAudioFrameCount((Double(buffer.frameLength) * sampleRateRatio).rounded(.up))

        func makeOutputBuffer(_ capacity: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
            guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
                throw AudioConverterError.failedToCreateBuffer
            }
            return out
        }

        var provided = false
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            if !provided {
                provided = true
                status.pointee = .haveData
                return buffer
            } else {
                status.pointee = .noDataNow
                return nil
            }
        }

        var error: NSError?
        let out = try makeOutputBuffer(estimate)
        let status = converter.convert(to: out, error: &error, withInputFrom: inputBlock)
        guard status != .error else { throw AudioConverterError.conversionFailed(error) }
        return out.frameLength > 0 ? extractFloatArray(from: out) : []
    }

    /// Ensure the internal converter matches the given formats.
    private func ensureConverter(input: AVAudioFormat, output: AVAudioFormat) throws {
        if converter == nil || converter?.inputFormat != input || converter?.outputFormat != output {
            converter = AVAudioConverter(from: input, to: output)
            logger.debug(
                "Created audio converter: \(input.sampleRate)Hz \(input.channelCount)ch -> \(output.sampleRate)Hz \(output.channelCount)ch"
            )
        }
        guard converter != nil else { throw AudioConverterError.failedToCreateConverter }
    }

    /// Check if a format already matches the target output format.
    private func isTargetFormat(_ format: AVAudioFormat) -> Bool {
        return format.sampleRate == targetFormat.sampleRate
            && format.channelCount == targetFormat.channelCount
            && format.commonFormat == targetFormat.commonFormat
            && format.isInterleaved == targetFormat.isInterleaved
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

    /// Reset the converter (useful when switching audio formats)
    public func reset() {
        converter = nil
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
