import Foundation
import OSLog

/// Audio buffer error types
enum AudioBufferError: Error, LocalizedError {
    case bufferOverflow

    var errorDescription: String? {
        switch self {
        case .bufferOverflow:
            return "Audio buffer overflow - not enough space for new samples"
        }
    }
}

/// Thread-safe circular audio buffer using Swift concurrency
actor AudioBuffer {
    private let logger = AppLogger(category: "AudioBuffer")
    private var buffer: [Float]
    private let capacity: Int
    private var writePosition: Int = 0
    private var readPosition: Int = 0
    private var count: Int = 0

    /// Chunk information for tracking processed segments
    private struct ChunkInfo {
        let startSample: Int
        let endSample: Int
        let timestamp: Date
    }

    private var processedChunks: [ChunkInfo] = []

    init(capacity: Int) {
        self.capacity = capacity
        self.buffer = Array(repeating: 0, count: capacity)
    }

    /// Append audio samples to the buffer
    func append(_ samples: [Float]) throws {
        // Handle case where samples exceed total capacity
        if samples.count > capacity {
            throw AudioBufferError.bufferOverflow
        }

        // If adding samples would overflow, make room by discarding old samples
        let overflow = max(0, (count + samples.count) - capacity)
        if overflow > 0 {
            readPosition = (readPosition + overflow) % capacity
            count -= overflow
            logger.debug("Buffer overflow handled: discarded \(overflow) old samples")
        }

        // Write new samples
        let newSamplesStartPos = writePosition
        for sample in samples {
            buffer[writePosition] = sample
            writePosition = (writePosition + 1) % capacity
        }
        count += samples.count

        // After overflow, if the new samples should be prioritized, adjust read position
        // to start reading from the beginning of the newly added samples
        if overflow > 0 {
            readPosition = newSamplesStartPos
            count = samples.count
        }
    }

    /// Get a chunk of audio
    /// - Parameter size: Size of the chunk in samples
    /// - Returns: Audio chunk or nil if not enough samples available
    func getChunk(size: Int) -> [Float]? {
        logger.debug("getChunk: requestedSize=\(size), available=\(self.count)")

        // Check if we have enough samples
        guard count >= size else {
            logger.debug("Not enough samples: need \(size), have \(self.count)")
            return nil
        }

        // Extract chunk
        var chunk: [Float] = []
        chunk.reserveCapacity(size)

        var pos = readPosition
        for _ in 0..<size {
            chunk.append(buffer[pos])
            pos = (pos + 1) % capacity
        }

        let oldReadPosition = readPosition
        // Update read position and count
        readPosition = (readPosition + size) % capacity
        count -= size

        logger.debug(
            "Advanced buffer: oldReadPos=\(oldReadPosition), newReadPos=\(self.readPosition), consumed=\(size), remaining=\(self.count)"
        )

        // Track this chunk
        processedChunks.append(
            ChunkInfo(
                startSample: oldReadPosition,
                endSample: readPosition,
                timestamp: Date()
            ))

        // Keep only recent chunk info (last 10 chunks)
        if processedChunks.count > 10 {
            processedChunks.removeFirst()
        }

        return chunk
    }

    /// Get a partial chunk of audio (for last segment)
    /// - Parameters:
    ///   - requestedSize: Desired size of the chunk
    ///   - allowPartial: If true, returns available samples even if less than requested
    /// - Returns: Audio chunk (may be smaller than requested if allowPartial is true)
    func getChunkWithPartial(requestedSize: Int, allowPartial: Bool = false) -> [Float]? {
        logger.debug(
            "getChunkWithPartial: requestedSize=\(requestedSize), available=\(self.count), allowPartial=\(allowPartial)"
        )

        // If we have enough samples, use regular getChunk
        if count >= requestedSize {
            return getChunk(size: requestedSize)
        }

        // If partial chunks not allowed, return nil
        guard allowPartial && count > 0 else {
            logger.debug("Not enough samples and partial not allowed")
            return nil
        }

        // Return all available samples as a partial chunk
        let partialSize = count
        logger.debug("Returning partial chunk of \(partialSize) samples")

        var chunk: [Float] = []
        chunk.reserveCapacity(partialSize)

        var pos = readPosition
        for _ in 0..<partialSize {
            chunk.append(buffer[pos])
            pos = (pos + 1) % capacity
        }

        // Update read position and count
        readPosition = pos
        count = 0

        // Track this partial chunk
        processedChunks.append(
            ChunkInfo(
                startSample: readPosition,
                endSample: pos,
                timestamp: Date()
            ))

        return chunk
    }

    /// Get all available samples without removing them
    func peekAvailable() -> [Float] {
        var samples: [Float] = []
        samples.reserveCapacity(count)

        var pos = readPosition
        for _ in 0..<count {
            samples.append(buffer[pos])
            pos = (pos + 1) % capacity
        }

        return samples
    }

    /// Get the number of available samples
    func availableSamples() -> Int {
        return count
    }

    /// Clear the buffer
    func clear() {
        writePosition = 0
        readPosition = 0
        count = 0
        processedChunks.removeAll()
    }

    /// Get buffer utilization percentage
    func utilization() -> Float {
        return Float(count) / Float(capacity) * 100.0
    }
}
