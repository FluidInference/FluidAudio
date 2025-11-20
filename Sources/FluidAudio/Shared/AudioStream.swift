//
//  File.swift
//  FluidAudio
//
//  Created by Benjamin Lee on 11/18/25.
//

import AVFoundation
import Foundation

public struct AudioStream: Sendable {
    // MARK: - public properties
    public typealias Callback = @Sendable (ArraySlice<Float>, TimeInterval) -> Void

    /// Audio sample rate
    public static let sampleRate: Double = 16_000

    /// Chunk duration in seconds
    public let chunkDuration: TimeInterval

    /// Overlap duration in seconds
    public let overlapDuration: TimeInterval

    /// Number of samples in a chunk
    public let chunkSize: Int

    /// Alignment mode for reading the chunks
    public let alignment: AudioStreamAlignment

    /// Whether the next chunk is ready to be read
    public var isChunkReady: Bool { buffer.count >= chunkSize }

    // MARK: - private properties
    private let hopSize: Int
    private let processGaps: Bool
    private var bufferStartTime: TimeInterval
    private var buffer: ContiguousArray<Float>
    private var callback: Callback?

    private var capacity: Int {
        if alignment == .backAligned {
            return chunkSize + hopSize
        }
        return buffer.capacity
    }

    // MARK: - init

    /// - Parameters:
    ///   - chunkDuration: Chunk duration in seconds
    ///   - overlapDuration: Chunk overlap duration in seconds
    ///   - time: Audio buffer start time
    ///   - alignment: Chunk alignment mode
    ///   - sampleRate: Audio sample rate in Hz
    ///   - processGaps: Whether gaps created during resynchronization should be processed (for front-aligned chunks only)
    public init(
        chunkDuration: TimeInterval = 10.0,
        overlapDuration: TimeInterval = 0.0,
        atTime time: TimeInterval = 0.0,
        alignment: AudioStreamAlignment = .backAligned,
        processGaps: Bool = false
    ) {
        self.chunkDuration = chunkDuration
        self.overlapDuration = overlapDuration
        self.bufferStartTime = time - (chunkDuration - overlapDuration)
        self.alignment = alignment

        self.chunkSize = Int(round(Self.sampleRate * chunkDuration))
        self.hopSize = Int(round(Self.sampleRate * (chunkDuration - overlapDuration)))
        self.processGaps = processGaps

        self.buffer = ContiguousArray(repeating: 0, count: chunkSize - hopSize)
        self.buffer.reserveCapacity(chunkSize + hopSize)
    }

    // MARK: - public methods

    /// Bind a callback to the chunk updates
    /// - Parameter callback: The callback to bind
    public mutating func bind(_ callback: @escaping Callback) {
        self.callback = callback
    }

    /// Remove update binding
    public mutating func unbind() {
        self.callback = nil
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - source: Audio samples to write
    ///   - time: Timestamp for resynchronization (optional)
    public mutating func write<C>(from source: C, atTime time: TimeInterval? = nil)
    where C: Collection, C.Element == Float {
        if let time {
            let startIndex = Int(round(bufferStartTime * Self.sampleRate))
            let endIndex = Int(round(time * Self.sampleRate))
            let expectedEndIndex = startIndex + source.count + buffer.count
            let indexOffset = endIndex - expectedEndIndex

            // re-synchronize
            if indexOffset > 0 {
                if processGaps == false || alignment == .backAligned {
                    trimToFit(count: indexOffset + source.count)
                }
                buffer.append(contentsOf: Array(repeating: 0, count: indexOffset))
            } else if indexOffset < 0 {
                buffer.removeLast(-indexOffset)
            }
        }

        if alignment == .backAligned {
            trimToFit(count: source.count)
        }

        buffer.append(contentsOf: source)

        guard let callback else { return }

        while isChunkReady {
            try? withChunk(callback)
        }
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - buffer: Audio buffer to write from
    ///   - time: Timestamp for resynchronization (optional)
    public mutating func write(from buffer: AVAudioPCMBuffer, atTime time: TimeInterval? = nil) throws {
        let samples = try AudioConverter().resampleBuffer(buffer)
        write(from: samples, atTime: time)
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - sampleBuffer: `CMSampleBuffer` to write from
    ///   - time: Timestamp for resynchronization (optional)
    public mutating func write(from sampleBuffer: CMSampleBuffer, atTime time: TimeInterval? = nil) throws {
        let samples = try AudioConverter().resampleSampleBuffer(sampleBuffer)
        write(from: samples, atTime: time)
    }

    /// Pop the next chunk and do something with it
    /// - Parameter body: Takes the chunk as an `ArraySlice<Float>` and the chunk start time.
    /// - Throws: `AudioStreamError.noChunksAvailable` if the next chunk is unavailable.
    public mutating func withChunk<R>(
        _ body: (ArraySlice<Float>, TimeInterval) throws -> R
    ) throws -> R {
        guard isChunkReady else {
            throw AudioStreamError.noChunksAvailable
        }

        defer {
            if alignment == .backAligned {
                let numRemoved = buffer.count - chunkSize + hopSize
                buffer.removeFirst(numRemoved)
                bufferStartTime += TimeInterval(numRemoved) / Self.sampleRate
            } else {
                buffer.removeFirst(hopSize)
                bufferStartTime += overlapDuration / Self.sampleRate
            }
        }

        let chunkOffset = TimeInterval(buffer.count - chunkSize) / Self.sampleRate
        let chunkStartTime = bufferStartTime + chunkOffset
        let sample = (alignment == .backAligned) ? buffer.suffix(chunkSize) : buffer.prefix(chunkSize)

        return try body(sample, chunkStartTime)
    }

    /// Pop the next chunk if it's ready
    /// - Returns: The next chunk and the chunk start time if its ready
    public mutating func readChunk() -> (chunk: [Float], chunkStartTime: TimeInterval)? {
        guard isChunkReady else {
            return nil
        }
        return try? withChunk { (Array($0), $1) }
    }

    // MARK: - private helpers

    /// Avoid unnecessary allocations due to overflow
    private mutating func trimToFit(count: Int) {
        let expectedCount = count + buffer.count
        if expectedCount > capacity {
            let numRemoved = min(expectedCount - capacity, buffer.count)
            buffer.removeFirst(expectedCount)
            bufferStartTime += TimeInterval(numRemoved) / Self.sampleRate
        }
    }
}

public enum AudioStreamError: Error, LocalizedError {
    case noChunksAvailable
}

public enum AudioStreamAlignment: Sendable {
    /// Ensures that the front of the chunks move forward consistently. Better if timing or precise overlap is very important.
    case frontAligned

    /// Use the most recent audio sample to form the chunk.  Better for staying up to date.
    case backAligned
}
