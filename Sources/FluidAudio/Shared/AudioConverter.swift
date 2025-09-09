import AVFoundation
import Foundation
import OSLog

/// Converts audio buffers to the format required by ASR (16kHz, mono, Float32)
@available(macOS 13.0, iOS 16.0, *)
public actor AudioConverter {
    private let logger = AppLogger(category: "AudioConverter")
    private var converter: AVAudioConverter?

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
    public func convertToAsrFormat(_ buffer: AVAudioPCMBuffer) throws -> [Float] {
        let inputFormat = buffer.format

        // If already in target format, just extract samples
        if inputFormat.sampleRate == targetFormat.sampleRate && inputFormat.channelCount == targetFormat.channelCount
            && inputFormat.commonFormat == targetFormat.commonFormat
        {
            return extractFloatArray(from: buffer)
        }

        // Convert to target format
        let convertedBuffer = try convertBuffer(buffer, to: targetFormat)
        return extractFloatArray(from: convertedBuffer)
    }

    /// Convert buffer to target format using AVAudioConverter
    private func convertBuffer(_ buffer: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        let inputFormat = buffer.format

        // Create or update converter if needed
        if converter == nil || converter?.inputFormat != inputFormat || converter?.outputFormat != format {
            converter = AVAudioConverter(from: inputFormat, to: format)
            converter?.primeMethod = .none  // Avoid timestamp drift

            logger.debug(
                "Created audio converter: \(inputFormat.sampleRate)Hz \(inputFormat.channelCount)ch -> \(format.sampleRate)Hz \(format.channelCount)ch"
            )
        }

        guard let converter = converter else {
            throw AudioConverterError.failedToCreateConverter
        }

        // Calculate output buffer size
        let sampleRateRatio = format.sampleRate / inputFormat.sampleRate
        let scaledFrameLength = Double(buffer.frameLength) * sampleRateRatio
        let frameCapacity = AVAudioFrameCount(scaledFrameLength.rounded(.up))

        guard
            let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: frameCapacity
            )
        else {
            throw AudioConverterError.failedToCreateBuffer
        }

        // Perform conversion
        var error: NSError?
        var inputConsumed = false

        let status = converter.convert(to: outputBuffer, error: &error) { _, statusPointer in
            if inputConsumed {
                statusPointer.pointee = .noDataNow
                return nil
            } else {
                statusPointer.pointee = .haveData
                inputConsumed = true
                return buffer
            }
        }

        guard status != .error else {
            throw AudioConverterError.conversionFailed(error)
        }

        return outputBuffer
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
