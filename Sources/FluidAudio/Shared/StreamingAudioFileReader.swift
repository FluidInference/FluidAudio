@preconcurrency import AVFoundation
import Foundation
import OSLog
import os

/// Provides streaming access to audio samples from a file with on-demand resampling.
/// Processes audio in fixed-size chunks to maintain constant memory usage regardless of file size.
public final class StreamingAudioFileReader: @unchecked Sendable {
    private let url: URL
    private let targetSampleRate: Int
    private let logger = AppLogger(category: "StreamingAudioFileReader")

    /// Total number of samples after resampling to target sample rate
    public let totalSampleCount: Int

    /// Duration of the audio in seconds
    public var duration: TimeInterval {
        Double(totalSampleCount) / Double(targetSampleRate)
    }

    /// Creates a streaming reader for the given audio file.
    /// - Parameters:
    ///   - url: URL to the audio file
    ///   - targetSampleRate: Target sample rate for output (default: 16000 Hz)
    /// - Throws: StreamingAudioError if the file cannot be opened or format is unsupported
    public init(url: URL, targetSampleRate: Int = 16000) throws {
        self.url = url
        self.targetSampleRate = targetSampleRate

        let audioFile = try AVAudioFile(forReading: url)
        let inputFormat = audioFile.processingFormat
        let sampleRateRatio = Double(targetSampleRate) / inputFormat.sampleRate
        self.totalSampleCount = Int((Double(audioFile.length) * sampleRateRatio).rounded(.up))
    }

    /// Creates an async stream that yields fixed-size buffers of resampled audio.
    /// - Parameter bufferSize: Size of each buffer in samples (default: 32,768, ~2 seconds at 16kHz)
    /// - Returns: AsyncThrowingStream yielding (samples: [Float], offset: Int) tuples
    public func streamBuffers(
        bufferSize: Int = 32_768
    ) -> AsyncThrowingStream<(samples: [Float], offset: Int), Error> {
        let alignedBufferSize = alignToEncoderFrames(bufferSize)

        return AsyncThrowingStream { continuation in
            Task {
                do {
                    try await self.performStreaming(
                        bufferSize: alignedBufferSize,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func performStreaming(
        bufferSize: Int,
        continuation: AsyncThrowingStream<(samples: [Float], offset: Int), Error>.Continuation
    ) async throws {
        let audioFile = try AVAudioFile(forReading: url)
        let inputFormat = audioFile.processingFormat
        let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 1,
            interleaved: false
        )!

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            throw StreamingAudioError.processingFailed(
                "Unsupported audio format \(inputFormat); cannot create converter"
            )
        }

        logger.debug(
            "Streaming \(self.url.lastPathComponent): \(inputFormat.sampleRate)Hz×\(inputFormat.channelCount)ch → \(targetFormat.sampleRate)Hz mono"
        )

        let inputCapacity: AVAudioFrameCount = 16_384
        guard
            let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: inputFormat,
                frameCapacity: inputCapacity
            )
        else {
            throw makeBufferError("Input", requestedFrames: Int(inputCapacity))
        }

        let estimatedOutputFrames = AVAudioFrameCount(
            (Double(inputCapacity) * targetFormat.sampleRate / inputFormat.sampleRate).rounded(.up)
        )
        guard
            let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: max(1024, estimatedOutputFrames)
            )
        else {
            throw makeBufferError("Output", requestedFrames: Int(estimatedOutputFrames))
        }

        var currentChunk: [Float] = []
        currentChunk.reserveCapacity(bufferSize)
        var currentOffset = 0
        var totalSamplesEmitted = 0

        let inputComplete = OSAllocatedUnfairLock(initialState: false)
        let readError = OSAllocatedUnfairLock<Error?>(initialState: nil)

        nonisolated(unsafe) let capturedInputBuffer = inputBuffer

        let inputBlock: AVAudioConverterInputBlock = { _, status in
            if inputComplete.withLock({ $0 }) {
                status.pointee = .endOfStream
                return nil
            }

            do {
                let remainingFrames = AVAudioFrameCount(audioFile.length - audioFile.framePosition)
                let framesToRead = min(inputCapacity, remainingFrames)
                if framesToRead > 0 {
                    try audioFile.read(into: capturedInputBuffer, frameCount: framesToRead)
                } else {
                    capturedInputBuffer.frameLength = 0
                }
            } catch {
                readError.withLock { $0 = error }
                capturedInputBuffer.frameLength = 0
            }

            guard capturedInputBuffer.frameLength > 0 else {
                inputComplete.withLock { $0 = true }
                status.pointee = .endOfStream
                return nil
            }

            status.pointee = .haveData
            return capturedInputBuffer
        }

        while true {
            outputBuffer.frameLength = 0
            var conversionError: NSError?
            let status = converter.convert(
                to: outputBuffer,
                error: &conversionError,
                withInputFrom: inputBlock
            )

            if let conversionError {
                throw StreamingAudioError.processingFailed(
                    "Audio conversion failed: \(conversionError.localizedDescription)"
                )
            }

            if let error = readError.withLock({ $0 }) {
                throw StreamingAudioError.processingFailed(
                    "Failed while reading audio: \(error.localizedDescription)"
                )
            }

            let producedFrames = Int(outputBuffer.frameLength)
            if producedFrames > 0 {
                guard let channelData = outputBuffer.floatChannelData?.pointee else {
                    throw StreamingAudioError.processingFailed("Missing channel data during conversion")
                }

                let samples = Array(UnsafeBufferPointer(start: channelData, count: producedFrames))
                currentChunk.append(contentsOf: samples)

                while currentChunk.count >= bufferSize {
                    let chunk = Array(currentChunk.prefix(bufferSize))
                    continuation.yield((samples: chunk, offset: currentOffset))
                    totalSamplesEmitted += chunk.count
                    currentOffset += chunk.count
                    currentChunk.removeFirst(bufferSize)
                }
            }

            if status == .endOfStream {
                break
            }
        }

        if !currentChunk.isEmpty {
            continuation.yield((samples: currentChunk, offset: currentOffset))
            totalSamplesEmitted += currentChunk.count
        }

        logger.debug("Streaming complete: emitted \(totalSamplesEmitted) samples")
    }

    private func alignToEncoderFrames(_ size: Int) -> Int {
        let samplesPerEncoderFrame = ASRConstants.samplesPerEncoderFrame
        return (size / samplesPerEncoderFrame) * samplesPerEncoderFrame
    }

    private func makeBufferError(_ name: String, requestedFrames: Int) -> StreamingAudioError {
        .processingFailed("Failed to allocate \(name.lowercased()) buffer (\(requestedFrames) frames)")
    }
}
