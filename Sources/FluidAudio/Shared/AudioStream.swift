import AVFoundation
import Foundation

/// A sliding window buffer for a real-time audio stream.
public final class AudioStream {
    public typealias Callback<C> = (C, TimeInterval) throws -> Void
    where C: RangeReplaceableCollection, C.Element == Float

    public typealias AsyncCallback<C> = @Sendable (C, TimeInterval) async -> Void
    where C: RangeReplaceableCollection, C.Element == Float

    // MARK: - public properties

    /// Audio sample rate
    public let sampleRate: Double

    /// Chunk duration in seconds
    public let chunkDuration: TimeInterval

    /// Duration between successive calls
    public let chunkSkip: TimeInterval

    /// Number of samples in a chunk
    public let chunkSize: Int

    /// Number of samples to skip between chunks
    public let skipSize: Int

    /// Alignment mode for reading the chunks
    public let chunkingStrategy: AudioStreamChunkingStrategy

    /// Whether the next chunk is ready to be read
    public var hasNewChunk: Bool { queue.sync { writeIndex >= temporaryChunkSize } }

    /// Duration of overlap between consecutive chunks
    public var chunkOverlap: TimeInterval { chunkDuration - chunkSkip }

    /// Number of samples that overlap between consecutive chunks
    public var overlapSize: Int { chunkSize - skipSize }

    // MARK: - private properties

    /// Index to which the next sample will be written
    private var writeIndex: Int

    /// Callback for when a new chunk is ready
    private var callback: Callback<[Float]>?

    /// Chunk size, but allows for the growing chunks that come with the `rampUpChunkSize` startup strategy
    private var temporaryChunkSize: Int

    /// Timestamp of the sample at index 0 in the audio buffer
    private var bufferStartTime: TimeInterval

    /// Sliding audio buffer
    private var buffer: ContiguousArray<Float>

    private let queue = DispatchQueue(label: "FluidAudio.AudioStream.queue", attributes: .concurrent)

    private let converter: AudioConverter

    // MARK: - init

    /// - Parameters:
    ///   - chunkDuration: Chunk duration in seconds
    ///   - chunkSkip: Duration between the start of each chunk (defaults to `chunkDuration`)
    ///   - time: Audio buffer start time
    ///   - chunkingStrategy: Strategy to determine how to align chunks at each write
    ///   - startupStrategy: Strategy to handle the first chunk(s) before the buffer is filled
    ///   - sampleRate: Audio sample rate
    ///   - bufferCapacitySeconds: The number of seconds of audio that the buffer can hold
    /// - Throws: `AudioStreamError.invalidChunkDuration` if `chunkDuration <= 0`
    /// - Throws: `AudioStreamError.invalidChunkSkip` if `chunkSkip <= 0` or `chunkSkip > chunkDuration`
    /// - Throws: `AudioStreamError.bufferTooSmall` if `bufferCapacitySeconds < chunkDuration`
    public init(
        chunkDuration: TimeInterval = 10.0,
        chunkSkip: TimeInterval? = nil,
        streamStartTime time: TimeInterval = 0.0,
        chunkingStrategy: AudioStreamChunkingStrategy = .useMostRecent,
        startupStrategy: AudioStreamStartupStrategy = .startSilent,
        sampleRate: Double = 16_000,
        bufferCapacitySeconds: TimeInterval? = nil
    ) throws {
        self.chunkDuration = chunkDuration
        self.chunkSkip = chunkSkip ?? chunkDuration

        self.chunkingStrategy = chunkingStrategy
        self.sampleRate = sampleRate

        self.chunkSize = Int(round(sampleRate * chunkDuration))
        self.skipSize = Int(round(sampleRate * self.chunkSkip))

        guard chunkSize > 0 else {
            throw AudioStreamError.invalidChunkDuration
        }

        guard skipSize > 0 && skipSize <= chunkSize else {
            throw AudioStreamError.invalidChunkSkip
        }

        let bufferDuration = bufferCapacitySeconds ?? (chunkDuration + self.chunkSkip)
        let capacity = Int(round(bufferDuration * sampleRate))
        guard capacity >= chunkSize else {
            throw AudioStreamError.bufferTooSmall
        }

        self.buffer = ContiguousArray(repeating: 0, count: capacity)

        self.converter = AudioConverter(
            targetFormat: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            )!
        )

        switch startupStrategy {
        case .startSilent:
            self.writeIndex = chunkSize - skipSize
            self.temporaryChunkSize = chunkSize
            self.bufferStartTime = time - (chunkDuration - self.chunkSkip)
        case .rampUpChunkSize:
            self.writeIndex = 0
            self.temporaryChunkSize = skipSize
            self.bufferStartTime = time
        case .waitForFullChunk:
            self.writeIndex = 0
            self.temporaryChunkSize = chunkSize
            self.bufferStartTime = time
        }
    }

    // MARK: - public methods

    /// Bind a callback to the chunk updates
    /// - Note: If the callback is slow, you may want to wrap it in a `Task`.
    /// - Parameter callback: The callback to bind
    public func bind(_ callback: @escaping Callback<[Float]>) {
        queue.sync(flags: .barrier) {
            self.callback = { (chunk: [Float], timestamp: TimeInterval) -> Void in
                try callback(chunk, timestamp)
            }
        }
    }

    /// Bind a callback to the chunk updates
    /// - Note: If the callback is slow, you may want to wrap it in a `Task`.
    /// - Parameter callback: The callback to bind
    public func bind<C>(_ callback: @escaping Callback<C>)
    where C: RangeReplaceableCollection, C.Element == Float {
        queue.sync(flags: .barrier) {
            self.callback = { (chunk: [Float], timestamp: TimeInterval) -> Void in
                try callback(C(chunk), timestamp)
            }
        }
    }

    /// Bind a fire and forget asynchronous callback to the chunk updates
    /// - Parameters:
    ///   - priority: Task priority
    ///   - callback: The callback to bind
    public func bind(
        priority: TaskPriority = .medium,
        _ callback: @escaping AsyncCallback<[Float]>
    ) {
        queue.sync(flags: .barrier) {
            self.callback = { (chunk: [Float], timestamp: TimeInterval) -> Void in
                let capturedChunk = chunk
                let capturedTimestamp = timestamp
                Task.detached(priority: priority) {
                    await callback(capturedChunk, capturedTimestamp)
                }
            }
        }
    }

    /// Bind a fire and forget asynchronous callback to the chunk updates
    /// - Parameters:
    ///   - priority: Task priority
    ///   - callback: The callback to bind
    public func bind<C>(
        priority: TaskPriority = .medium,
        _ callback: @escaping AsyncCallback<C>
    )
    where C: RangeReplaceableCollection, C.Element == Float, C: Sendable {
        queue.sync(flags: .barrier) {
            self.callback = { (chunk: [Float], timestamp: TimeInterval) -> Void in
                let capturedChunk = C(chunk)
                let capturedTimestamp = timestamp
                Task.detached(priority: priority) {
                    await callback(capturedChunk, capturedTimestamp)
                }
            }
        }
    }

    /// Remove update binding
    public func unbind() {
        queue.sync(flags: .barrier) {
            self.callback = nil
        }
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - source: Audio samples to write
    ///   - time: Timestamp for resynchronization (optional)
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public func write<C>(from source: C, atTime time: TimeInterval? = nil) throws
    where C: Collection, C.Element == Float {
        if let array = source as? [Float] {
            return try self.writeContiguousSamples(from: array, atTime: time)
        }
        if let slice = source as? ArraySlice<Float> {
            return try self.writeContiguousSamples(from: slice, atTime: time)
        }
        if let contiguous = source as? ContiguousArray<Float> {
            return try self.writeContiguousSamples(from: contiguous, atTime: time)
        }
        try self.writeContiguousSamples(from: Array(source), atTime: time)
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - buffer: Audio buffer to write from
    ///   - time: Timestamp for resynchronization (optional)
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public func write(from buffer: AVAudioPCMBuffer, atTime time: TimeInterval? = nil) throws {
        let samples = try converter.resampleBuffer(buffer)
        try write(from: samples, atTime: time)
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - sampleBuffer: `CMSampleBuffer` to write from
    ///   - time: Timestamp for resynchronization (optional)
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public func write(from sampleBuffer: CMSampleBuffer, atTime time: TimeInterval? = nil) throws {
        let samples = try converter.resampleSampleBuffer(sampleBuffer)
        try write(from: samples, atTime: time)
    }

    /// Pop the next chunk if available and do something with it
    /// - Parameter body: Takes the chunk as an `ArraySlice<Float>` and the chunk start time
    /// - Note: Chunks will never be available if the audio stream has a binding
    public func withChunkIfAvailable<R, C>(
        _ body: (C, TimeInterval) throws -> R
    ) rethrows -> R?
    where C: RangeReplaceableCollection, C.Element == Float {
        guard let (chunkArray, timestamp) = readChunkIfAvailable() else {
            return nil
        }
        return try body(C(chunkArray), timestamp)
    }

    /// Pop the next chunk if it's ready
    /// - Returns: The next chunk and the chunk start time if its ready
    /// - Note: Chunks will never be available if the audio stream has a binding
    public func readChunkIfAvailable() -> (chunk: [Float], chunkStartTime: TimeInterval)? {
        guard hasNewChunk else {
            return nil
        }

        return queue.sync(flags: .barrier) {
            var chunk: [Float] = []
            var chunkStartTime: TimeInterval = 0

            // Extract the chunk
            switch chunkingStrategy {
            case .useMostRecent:
                let chunkStartIndex = writeIndex - temporaryChunkSize
                chunkStartTime = bufferStartTime + TimeInterval(chunkStartIndex) / sampleRate
                chunk = Array(buffer[chunkStartIndex..<writeIndex])
            case .useFixedSkip:
                chunk = Array(buffer.prefix(temporaryChunkSize))
                chunkStartTime = bufferStartTime
            }

            if temporaryChunkSize == chunkSize {
                // Forget the front of the buffer
                switch chunkingStrategy {
                case .useMostRecent:
                    forgetOldest(writeIndex - overlapSize)
                case .useFixedSkip:
                    forgetOldest(skipSize)
                }
            } else {
                // Update temporary chunk size
                temporaryChunkSize = min(temporaryChunkSize + skipSize, chunkSize)
            }

            return (chunk, chunkStartTime)
        }
    }

    // MARK: - private helpers

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - source: Audio samples to write
    ///   - time: Timestamp for resynchronization (optional)
    private func writeContiguousSamples<C>(from source: C, atTime time: TimeInterval? = nil) throws
    where C: RandomAccessCollection, C.Element == Float {
        guard source.count > 0 else {
            return
        }

        queue.sync(flags: .barrier) {
            if let time {
                // Align samples with timestamps
                let startIndex = Int(round(bufferStartTime * sampleRate))
                let endIndex = startIndex + writeIndex + source.count
                let expectedEndIndex = Int(round(time * sampleRate))

                let deviation = expectedEndIndex - endIndex

                if deviation > 0 {
                    appendZeros(count: deviation, beforeAdding: source.count)
                } else if deviation < 0 {
                    rollbackNewest(-deviation)
                }
            }

            // Write new samples
            source.withContiguousStorageIfAvailable { ptr in
                append(from: ptr.baseAddress!, count: ptr.count)
            }
        }

        // It's technically possible to have multiple chunks ready at once
        while let callback = queue.sync(execute: { self.callback }),
            let (chunk, timestamp) = readChunkIfAvailable()
        {
            try callback(chunk, timestamp)
        }
    }

    /// Rollback the newest `count` samples.
    /// - Parameter count: Number of samples to rollback
    private func rollbackNewest(_ count: Int) {
        writeIndex -= count

        if writeIndex < 0 {
            bufferStartTime += TimeInterval(writeIndex) / sampleRate
            writeIndex = 0
        }
    }

    /// Forget the oldest `count` audio samples.
    /// - Parameter count: Number of samples to forget
    private func forgetOldest(_ count: Int) {
        // Bring all elements in the index range [count, writeIndex) to the front
        if count < writeIndex {
            buffer.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else {
                    return
                }

                let stride = MemoryLayout<Float>.stride

                memmove(
                    base,
                    base.advanced(by: count),
                    (writeIndex - count) * stride)
            }
        }
        writeIndex -= count
        bufferStartTime += TimeInterval(count) / sampleRate
    }

    /// Append new audio samples to the buffer
    /// - Parameters:
    ///   - src: Source pointer
    ///   - count: Number of samples to append
    private func append(from src: UnsafePointer<Float>, count: Int) {
        let shiftedWriteIndex: Int
        switch chunkingStrategy {
        case .useMostRecent:
            shiftedWriteIndex = temporaryChunkSize
        case .useFixedSkip:
            shiftedWriteIndex = buffer.count
        }

        let countAdded = prepareToAppendAndReturnNumAdded(
            count: count,
            cappingWriteIndexAt: buffer.count,
            shiftWriteIndexTo: shiftedWriteIndex
        )

        guard countAdded > 0 else {
            return
        }

        // Drop all samples that didn't fit
        let source = src.advanced(by: count - countAdded)

        // Append the source
        buffer.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else {
                return
            }
            let stride = MemoryLayout<Float>.stride
            memcpy(base.advanced(by: writeIndex), source, countAdded * stride)
            writeIndex += countAdded
        }
    }

    /// Append zeros to the buffer
    /// - Parameters:
    ///   - count: Number of zeros to append
    ///   - addedSampleCount: Number of samples being appended after the zeros
    private func appendZeros(count: Int, beforeAdding addedSampleCount: Int) {
        let shiftedWriteIndex: Int

        switch chunkingStrategy {
        case .useMostRecent:
            shiftedWriteIndex = temporaryChunkSize - addedSampleCount
        case .useFixedSkip:
            shiftedWriteIndex = buffer.count - addedSampleCount
        }

        let countAdded = prepareToAppendAndReturnNumAdded(
            count: count,
            cappingWriteIndexAt: buffer.count - addedSampleCount,
            shiftWriteIndexTo: shiftedWriteIndex
        )

        guard countAdded > 0 else {
            return
        }

        // Append the source
        buffer.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else {
                return
            }
            let stride = MemoryLayout<Float>.stride
            memset(base.advanced(by: writeIndex), 0, countAdded * stride)
            writeIndex += countAdded
        }
    }

    /// Prepare to append `count` samples to the buffer and trim the samples to fit
    /// - Parameters:
    ///   - count: Number of samples about to be appended
    ///   - maxWriteIndex: Maximum value `writeIndex` can reach before shifting back the buffer
    ///   - shiftedWriteIndex: Desired value of `writeIndex` after shifting back the buffer
    /// - Returns: The number of samples that will actually be appended
    @discardableResult
    private func prepareToAppendAndReturnNumAdded(
        count: Int,
        cappingWriteIndexAt maxWriteIndex: Int,
        shiftWriteIndexTo shiftedWriteIndex: Int
    ) -> Int {
        precondition(maxWriteIndex >= shiftedWriteIndex)

        var writeIndexAfterAppend = writeIndex + count

        // Shift back so the writeIndex will stay in bounds
        if writeIndexAfterAppend > maxWriteIndex {
            forgetOldest(writeIndexAfterAppend - shiftedWriteIndex)
            writeIndexAfterAppend = shiftedWriteIndex
        }

        // Exit if the entire source precedes the buffer
        guard writeIndexAfterAppend > 0 else {
            writeIndex = writeIndexAfterAppend
            return 0
        }

        // Remove any incoming samples that precede the buffer
        if writeIndex < 0 {
            let numToForget = -writeIndex
            writeIndex = 0
            return count - numToForget
        }

        return count
    }
}

public enum AudioStreamError: Error, LocalizedError {
    case bufferTooSmall
    case invalidChunkSkip
    case invalidChunkDuration
}

public enum AudioStreamChunkingStrategy: Sendable {
    /// Ensure that the number of samples between the start of each chunk is constant.
    case useFixedSkip

    /// Use the most recent audio samples to form the chunk.
    case useMostRecent
}

public enum AudioStreamStartupStrategy: Sendable {
    /// Start with a silent audio stream. Callbacks will begin after `chunkSkip` seconds.
    case startSilent

    /// Chunk size will increase by `chunkSkip` seconds each callback until reaching `chunkDuration`
    case rampUpChunkSize

    /// Wait for the first chunk to fill up before running
    case waitForFullChunk
}
