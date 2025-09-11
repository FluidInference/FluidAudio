import AVFoundation
import Foundation
import OSLog

/// Converts audio buffers to the format required by ASR (16kHz, mono, Float32).
///
/// Implementation notes:
/// - Uses `AVAudioConverter` for all sample-rate, sample-format, and channel-count conversions.
/// - Avoids any manual resampling; only raw sample extraction occurs after conversion.
/// - Supports streaming by retaining the converter between calls and deferring drain until finish.
@available(macOS 13.0, iOS 16.0, *)
public actor AudioConverter {
    private let logger = AppLogger(category: "AudioConverter")
    private var converter: AVAudioConverter?

    /// Public initializer so external modules (e.g. CLI) can construct the converter
    public init() {}

    /// Target format for ASR: 16kHz, mono, Float32
    private let targetFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16000,
        channels: 1,
        interleaved: false
    )!

    /// Convert an AVAudioPCMBuffer to ASR-ready Float array
    /// - Parameter buffer: Input audio buffer (any format)
    /// - Returns: Float array at 16kHz mono
    /// - Parameter streaming: Set to true when converting streaming chunks to avoid end-of-stream flushing
    public func convertToAsrFormat(_ buffer: AVAudioPCMBuffer, streaming: Bool = false) throws -> [Float] {
        let inputFormat = buffer.format

        // If already in target format, just extract samples
        if inputFormat.sampleRate == targetFormat.sampleRate && inputFormat.channelCount == targetFormat.channelCount
            && inputFormat.commonFormat == targetFormat.commonFormat
        {
            return extractFloatArray(from: buffer)
        }

        // Convert to target format
        let samples = try convertBufferToSamples(buffer, to: targetFormat, streaming: streaming)
        return samples
    }

    /// Convert an audio file to 16kHz mono Float32 samples
    /// - Parameter url: URL of the audio file to read
    /// - Returns: Float array at 16kHz mono
    public func convertFileToAsrSamples(_ url: URL) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioConverterError.failedToCreateBuffer
        }

        try audioFile.read(into: buffer)
        return try convertToAsrFormat(buffer, streaming: false)
    }

    /// Convert an audio file path to 16kHz mono Float32 samples
    /// - Parameter path: File path of the audio file to read
    /// - Returns: Float array at 16kHz mono
    public func convertFileToAsrSamples(path: String) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        return try convertFileToAsrSamples(url)
    }

    /// Finish a streaming conversion sequence and flush any remaining samples
    /// - Returns: Additional samples produced by draining the converter, if any
    public func finishStreamingConversion() throws -> [Float] {
        guard let converter = converter else { return [] }

        // Do not re-prime; just drain remaining frames
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

    /// Convert buffer to target format using AVAudioConverter
    /// Returns extracted Float samples to make it easy to handle multi-pass conversion (incl. flush)
    private func convertBufferToSamples(
        _ buffer: AVAudioPCMBuffer,
        to format: AVAudioFormat,
        streaming: Bool
    ) throws -> [Float] {
        let inputFormat = buffer.format

        // Create or update converter if needed
        if converter == nil || converter?.inputFormat != inputFormat || converter?.outputFormat != format {
            converter = AVAudioConverter(from: inputFormat, to: format)
            // primeMethod is set per-call below based on streaming/batch

            logger.debug(
                "Created audio converter: \(inputFormat.sampleRate)Hz \(inputFormat.channelCount)ch -> \(format.sampleRate)Hz \(format.channelCount)ch"
            )
        }

        guard let converter = converter else {
            throw AudioConverterError.failedToCreateConverter
        }

        // Set prime method depending on mode
        converter.primeMethod = streaming ? .none : .normal

        // Prepare capacity estimation for first pass
        let sampleRateRatio = format.sampleRate / inputFormat.sampleRate
        let estimatedOutputFrames = AVAudioFrameCount((Double(buffer.frameLength) * sampleRateRatio).rounded(.up))

        // Helper to create an output buffer with a chosen capacity
        func makeOutputBuffer(capacity: AVAudioFrameCount) throws -> AVAudioPCMBuffer {
            guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else {
                throw AudioConverterError.failedToCreateBuffer
            }
            return out
        }

        var aggregatedSamples: [Float] = []
        aggregatedSamples.reserveCapacity(Int(estimatedOutputFrames))

        // Input management: provide the input buffer once, then either no-data (streaming) or end-of-stream (batch)
        var inputProvided = false
        let inputBlock: AVAudioConverterInputBlock = { _, statusPointer in
            if !inputProvided {
                statusPointer.pointee = .haveData
                inputProvided = true
                return buffer
            } else {
                statusPointer.pointee = streaming ? .noDataNow : .endOfStream
                return nil
            }
        }

        // First conversion pass: main data
        var error: NSError?
        do {
            let firstOut = try makeOutputBuffer(capacity: estimatedOutputFrames)
            let status = converter.convert(to: firstOut, error: &error, withInputFrom: inputBlock)
            guard status != .error else { throw AudioConverterError.conversionFailed(error) }
            if firstOut.frameLength > 0 {
                aggregatedSamples.append(contentsOf: extractFloatArray(from: firstOut))
            }

            // For streaming mode, we stop after the first pass to avoid flushing and preserve continuity
            if streaming {
                return aggregatedSamples
            }

            // Batch mode: drain any remaining frames by repeatedly converting with end-of-stream
            while true {
                let drainOut = try makeOutputBuffer(capacity: 4096)
                let drainStatus = converter.convert(to: drainOut, error: &error, withInputFrom: inputBlock)
                guard drainStatus != .error else { throw AudioConverterError.conversionFailed(error) }
                if drainOut.frameLength > 0 {
                    aggregatedSamples.append(contentsOf: extractFloatArray(from: drainOut))
                }
                if drainStatus == .endOfStream { break }
                // If converter reports inputRanDry or haveData unexpectedly, continue until endOfStream
            }
        } catch {
            throw error
        }

        return aggregatedSamples
    }

    /// Extract Float array from PCM buffer
    private func extractFloatArray(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else {
            return []
        }

        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)

        // For mono, just copy the data
        if channelCount == 1 {
            return Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        }

        // For multi-channel, average to mono
        var samples: [Float] = []
        samples.reserveCapacity(frameCount)

        for frame in 0..<frameCount {
            var sum: Float = 0
            for channel in 0..<channelCount {
                sum += channelData[channel][frame]
            }
            samples.append(sum / Float(channelCount))
        }

        return samples
    }

    /// Reset the converter (useful when switching audio formats)
    public func reset() {
        converter = nil
        logger.debug("Audio converter reset")
    }

    /// Cleanup all resources
    public func cleanup() {
        converter = nil
        logger.debug("Audio converter cleaned up")
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
