//
//  File.swift
//  FluidAudio
//
//  Created by Benjamin Lee on 11/18/25.
//

import AVFoundation
import Foundation

/// A sliding window buffer for a real-time audio stream.
public struct AudioStream: Sendable {
    // MARK: - public properties
    public typealias Callback = @Sendable (ArraySlice<Float>, TimeInterval) throws -> Void

    /// Audio sample rate
    public let sampleRate: Double

    /// Chunk duration in seconds
    public let chunkDuration: TimeInterval

    /// Duration between successive calls
    public let chunkSkip: TimeInterval

    /// Number of samples in a chunk
    public let chunkSize: Int

    /// Alignment mode for reading the chunks
    public let chunkingStrategy: AudioStreamChunkingStrategy

    /// Whether the next chunk is ready to be read
    public var isChunkReady: Bool { writeIndex >= temporaryChunkSize }

    // MARK: - private properties
    private let skipSize: Int
    private var bufferStartTime: TimeInterval
    private var buffer: ContiguousArray<Float>
    private var writeIndex: Int
    private var callback: Callback?
    private var temporaryChunkSize: Int

    // MARK: - init

    /// - Parameters:
    ///   - chunkDuration: Chunk duration in seconds
    ///   - chunkSkip: Duration between the start of each chunk (defaults to `chunkDuration`)
    ///   - time: Audio buffer start time
    ///   - alignment: Chunk alignment mode
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
        guard chunkDuration >= 1 / sampleRate else {
            throw AudioStreamError.invalidChunkDuration
        }
        
        self.chunkDuration = chunkDuration
        self.chunkSkip = chunkSkip ?? chunkDuration
        
        guard self.chunkSkip > 0 && self.chunkSkip <= chunkDuration else {
            throw AudioStreamError.invalidChunkSkip
        }
        
        self.chunkingStrategy = chunkingStrategy
        self.sampleRate = sampleRate
        
        self.chunkSize = Int(round(sampleRate * chunkDuration))
        self.skipSize = Int(round(sampleRate * self.chunkSkip))

        let capacity = Int(round((bufferCapacitySeconds ?? (chunkDuration + self.chunkSkip))
                                 * sampleRate))
        guard capacity >= chunkSize else {
            throw AudioStreamError.bufferTooSmall
        }
        self.buffer = ContiguousArray(repeating: 0, count: capacity)
        
        switch startupStrategy {
        case .startSilent:
            self.writeIndex = chunkSize - skipSize
            self.bufferStartTime = time - (chunkDuration - self.chunkSkip)
            self.temporaryChunkSize = chunkSize
        case .rampUpChunkSize:
            self.writeIndex = 0
            self.bufferStartTime = time
            self.temporaryChunkSize = skipSize
        case .waitForFullChunk:
            self.writeIndex = 0
            self.bufferStartTime = time
            self.temporaryChunkSize = chunkSize
        }
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
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public mutating func write(from source: [Float], atTime time: TimeInterval? = nil) throws {
        try writeGeneric(from: source, atTime: time)
    }
    
    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - source: Audio samples to write
    ///   - time: Timestamp for resynchronization (optional)
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public mutating func write(from source: ArraySlice<Float>, atTime time: TimeInterval? = nil) throws {
        try writeGeneric(from: source, atTime: time)
    }
    
    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - source: Audio samples to write
    ///   - time: Timestamp for resynchronization (optional)
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public mutating func write(from source: ContiguousArray<Float>, atTime time: TimeInterval? = nil) throws {
        try writeGeneric(from: source, atTime: time)
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - buffer: Audio buffer to write from
    ///   - time: Timestamp for resynchronization (optional)
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public mutating func write(from buffer: AVAudioPCMBuffer, atTime time: TimeInterval? = nil) throws {
        let samples = try AudioConverter().resampleBuffer(buffer)
        try write(from: samples, atTime: time)
    }

    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - sampleBuffer: `CMSampleBuffer` to write from
    ///   - time: Timestamp for resynchronization (optional)
    /// - Warning: Samples may be skipped if the time jumps forward significantly.
    public mutating func write(from sampleBuffer: CMSampleBuffer, atTime time: TimeInterval? = nil) throws {
        let samples = try AudioConverter().resampleSampleBuffer(sampleBuffer)
        try write(from: samples, atTime: time)
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
        
        // Do stuff with the chunk
        let result: R
        switch chunkingStrategy {
        case .useMostRecent:
            let chunkOffset = TimeInterval(writeIndex - temporaryChunkSize) / sampleRate
            let chunkStartTime = bufferStartTime + chunkOffset
            let sample = buffer[writeIndex-temporaryChunkSize..<writeIndex]
            result = try body(sample, chunkStartTime)
        case .useFixedHop:
            let sample = buffer.prefix(temporaryChunkSize)
            let chunkStartTime = bufferStartTime
            result = try body(sample, chunkStartTime)
        }
        
        // Update temporary chunk size if needed
        guard temporaryChunkSize == chunkSize else {
            temporaryChunkSize = min(temporaryChunkSize + skipSize, chunkSize)
            return result
        }
        
        // Forget the front of the buffer
        switch chunkingStrategy {
        case .useMostRecent:
            forgetFirst(writeIndex - (chunkSize - skipSize))
        case .useFixedHop:
            forgetFirst(skipSize)
        }
        
        return result
    }

    /// Pop the next chunk if it's ready
    /// - Returns: The next chunk and the chunk start time if its ready
    public mutating func readChunk() -> (chunk: [Float], chunkStartTime: TimeInterval)? {
        guard isChunkReady else {
            return nil
        }
        
        // In theory, this should never throw
        return try? withChunk { chunk, timestamp in
            return (Array(chunk), timestamp)
        }
    }
    
    // MARK: - private helpers
    
    /// Add new audio data to the buffer
    /// - Parameters:
    ///   - source: Audio samples to write
    ///   - time: Timestamp for resynchronization (optional)
    private mutating func writeGeneric<C>(from source: C, atTime time: TimeInterval? = nil) throws
    where C: Collection, C.Element == Float {
        let writeIndexReset = (chunkingStrategy == .useMostRecent ?
                               temporaryChunkSize : buffer.count)
        
        if let time {
            let startIndex = Int(round(bufferStartTime * sampleRate))
            let endIndex = startIndex + writeIndex + source.count
            let expectedEndIndex = Int(round(time * sampleRate))

            var missingSampleCount = expectedEndIndex - endIndex

            if missingSampleCount > 0 {
                prepareToAppend(from: nil,
                                count: &missingSampleCount,
                                maxWriteIndex: buffer.count - source.count,
                                shiftedWriteIndex: writeIndexReset - source.count)
                if missingSampleCount > 0 {
                    let stride = MemoryLayout<Float>.stride
                    memset(&buffer[writeIndex], 0, missingSampleCount * stride)
                    writeIndex += missingSampleCount
                }
            } else if missingSampleCount < 0 {
                rollbackLast(-missingSampleCount)
            }
        }

        let success: Void? = try source.withContiguousStorageIfAvailable { ptr in
            guard let base = ptr.baseAddress else {
                throw AudioStreamError.sourceBaseAddressUnknown
            }
            var count = ptr.count
            
            guard let sourceBase = prepareToAppend(from: base,
                                                   count: &count,
                                                   maxWriteIndex: buffer.count,
                                                   shiftedWriteIndex: writeIndexReset)
            else {
                return
            }
            
            try append(from: sourceBase, count: count)
        }
        
        guard let success else {
            throw AudioStreamError.discontiguousSourceBuffer
        }

        guard let callback else { return }

        while isChunkReady {
            try? withChunk(callback)
        }
    }
    
    private mutating func rollbackLast(_ count: Int) {
        writeIndex -= count
        
        if writeIndex < 0 {
            bufferStartTime += TimeInterval(writeIndex) / sampleRate
            writeIndex = 0
        }
    }
    
    private mutating func forgetFirst(_ count: Int) {
        // Bring all elements in the index range [count, writeIndex) to the front
        if count < writeIndex {
            buffer.withUnsafeMutableBufferPointer { ptr in
                guard let base = ptr.baseAddress else {
                    return
                }
                
                let stride = MemoryLayout<Float>.stride
                
                memmove(base,
                        base.advanced(by: count),
                        (writeIndex - count) * stride)
            }
        }
        writeIndex -= count
        bufferStartTime += TimeInterval(count) / sampleRate
    }
    
    private mutating func append(from src: UnsafePointer<Float>, count: Int) throws {
        try buffer.withUnsafeMutableBufferPointer { ptr in
            guard let base = ptr.baseAddress else {
                throw AudioStreamError.bufferBaseAddressUnknown
            }
            let stride = MemoryLayout<Float>.stride
            memcpy(base.advanced(by: writeIndex), src, count * stride)
            writeIndex += count
        }
    }
    
    @discardableResult
    private mutating func prepareToAppend(from src: UnsafePointer<Float>?, count: inout Int, maxWriteIndex: Int, shiftedWriteIndex: Int) -> UnsafePointer<Float>? {
        precondition(maxWriteIndex >= shiftedWriteIndex)
        
        var newWriteIndex = writeIndex + count
        
        guard newWriteIndex > 0 else {
            // none of the items will be written since they don't reach the start of the buffer
            writeIndex += count
            count = 0
            return nil
        }
        
        if newWriteIndex > maxWriteIndex {
            // shift back so that the write index is in bounds
            forgetFirst(newWriteIndex - shiftedWriteIndex)
            
            // check if the source now precedes the buffer
            guard shiftedWriteIndex > 0 else {
                writeIndex += count
                count = 0
                return nil
            }
        }
        
        // drop any part of the source the precedes the buffer
        if writeIndex < 0 {
            let shift = -writeIndex
            writeIndex = 0
            count -= shift
            return src?.advanced(by: shift)
        }
        
        return src
    }
}

public enum AudioStreamError: Error, LocalizedError {
    case noChunksAvailable
    case bufferTooSmall
    case invalidChunkSkip
    case invalidChunkDuration
    case bufferBaseAddressUnknown
    case sourceBaseAddressUnknown
    case discontiguousSourceBuffer
}

public enum AudioStreamChunkingStrategy: Sendable {
    /// Ensure that the number of samples between the start of each chunk is constant.
    case useFixedHop

    /// Use the most recent audio samples to form the chunk.
    case useMostRecent
}

public enum AudioStreamStartupStrategy: Sendable {
    /// Start with a silent audio stream. Callbacks will begin after `chunkDuration - chunkOverlap` seconds.
    case startSilent

    /// Chunk size will increase by `chunkDuration - chunkOverlap` seconds each callback until reaching `chunkDuration`
    case rampUpChunkSize
    
    /// Wait for the first chunk to fill up before running
    case waitForFullChunk
}
