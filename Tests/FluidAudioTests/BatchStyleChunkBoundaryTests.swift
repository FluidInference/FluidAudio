import Foundation
import XCTest

@testable import FluidAudio

/// Tests for chunk boundary timing logic in BatchStyleStreamingManager.
/// Verifies when chunks are triggered and how the audio buffer is managed.
final class BatchStyleChunkBoundaryTests: XCTestCase {

    // MARK: - Constants

    private let sampleRate = 16000

    // MARK: - Helper Types

    /// Simulates the chunk triggering logic from BatchStyleStreamingManager
    private struct ChunkBufferSimulator {
        let config: BatchStyleStreamingConfig
        var sampleBuffer: [Float] = []
        var totalSamplesReceived: Int = 0
        var lastProcessedSampleIndex: Int = 0
        var processedChunks: Int = 0

        mutating func appendSamples(_ samples: [Float]) -> [(start: Int, end: Int)] {
            sampleBuffer.append(contentsOf: samples)
            totalSamplesReceived += samples.count

            var triggeredChunks: [(start: Int, end: Int)] = []

            let chunkSamples = config.chunkSamples
            let overlapSamples = config.overlapSamples

            // Calculate available new samples
            let availableNewSamples = sampleBuffer.count - (lastProcessedSampleIndex > 0 ? overlapSamples : 0)

            // Check if we have enough for a chunk
            if availableNewSamples >= chunkSamples ||
                (sampleBuffer.count >= config.minChunkSamples && availableNewSamples >= config.minChunkSamples)
            {
                let chunkStart = max(0, sampleBuffer.count - availableNewSamples - overlapSamples)
                let chunkEnd = min(sampleBuffer.count, chunkStart + chunkSamples)

                if chunkEnd > chunkStart {
                    triggeredChunks.append((start: chunkStart, end: chunkEnd))
                    processedChunks += 1

                    // Simulate buffer trimming
                    lastProcessedSampleIndex = chunkEnd
                    let keepFromIndex = max(0, chunkEnd - overlapSamples)
                    if keepFromIndex > 0 {
                        sampleBuffer.removeFirst(keepFromIndex)
                        lastProcessedSampleIndex -= keepFromIndex
                    }
                }
            }

            return triggeredChunks
        }
    }

    // MARK: - First Chunk Triggering Tests
    //
    // NOTE: The actual BatchStyleStreamingManager triggers chunks when EITHER:
    // 1. We have >= chunkSamples worth of new audio, OR
    // 2. We have >= minChunkSamples worth of new audio (for low-latency updates)
    //
    // This means chunks trigger at minChunkSamples, not chunkSamples!

    func testFirstChunkTriggerAtMinChunkSize() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples  // 32000 (2s)

        // Add samples just under min chunk size - should NOT trigger
        var chunks = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples - 1))
        XCTAssertEqual(chunks.count, 0, "Should not trigger chunk with \(minChunkSamples - 1) samples")

        // Add one more sample to reach min chunk size - should trigger
        chunks = simulator.appendSamples([0.5])
        XCTAssertEqual(chunks.count, 1, "Should trigger chunk at exactly \(minChunkSamples) samples")
    }

    func testFirstChunkTriggerWithLowLatencyConfig() {
        var simulator = ChunkBufferSimulator(config: .lowLatency)
        let minChunkSamples = simulator.config.minChunkSamples  // 32000 (2s)

        // Add samples just under min chunk size
        var chunks = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples - 1))
        XCTAssertEqual(chunks.count, 0, "Should not trigger with \(minChunkSamples - 1) samples")

        // Trigger at exactly min chunk size
        chunks = simulator.appendSamples([0.5])
        XCTAssertEqual(chunks.count, 1, "Should trigger at \(minChunkSamples) samples")
    }

    // MARK: - Overlap Preservation Tests

    func testOverlapPreservedAfterChunk() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples
        let overlapSamples = simulator.config.overlapSamples

        // Trigger first chunk at min chunk size
        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))

        // Buffer should now contain only overlap samples
        XCTAssertEqual(
            simulator.sampleBuffer.count, overlapSamples,
            "Buffer should contain \(overlapSamples) overlap samples after first chunk"
        )
    }

    func testSecondChunkTriggersAtMinChunkNewSamples() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples  // 32000

        // Trigger first chunk at min chunk size
        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))
        XCTAssertEqual(simulator.processedChunks, 1)

        // After first chunk, need minChunkSamples of NEW audio for second chunk
        // (availableNewSamples = buffer - overlap, need >= minChunkSamples)
        var chunks = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples - 1))
        XCTAssertEqual(chunks.count, 0, "Should not trigger with minChunk - 1 new samples")

        chunks = simulator.appendSamples([0.5])
        XCTAssertEqual(chunks.count, 1, "Second chunk should trigger at minChunk new samples")
        XCTAssertEqual(simulator.processedChunks, 2)
    }

    // MARK: - Multiple Chunk Tests

    func testMultipleChunkProgression() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples  // 32000

        // First chunk at minChunkSamples
        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))
        XCTAssertEqual(simulator.processedChunks, 1, "First chunk")

        // Second chunk (need minChunkSamples NEW samples)
        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))
        XCTAssertEqual(simulator.processedChunks, 2, "Second chunk")

        // Third chunk (need minChunkSamples NEW samples)
        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))
        XCTAssertEqual(simulator.processedChunks, 3, "Third chunk")
    }

    func testTotalSamplesTracking() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples

        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))

        XCTAssertEqual(
            simulator.totalSamplesReceived, minChunkSamples,
            "Total samples should track all received samples"
        )

        _ = simulator.appendSamples(Array(repeating: 0.5, count: 50000))

        XCTAssertEqual(
            simulator.totalSamplesReceived, minChunkSamples + 50000,
            "Total samples should accumulate"
        )
    }

    // MARK: - Chunk Size Validation Tests

    func testChunkSizeMatchesAvailableAudio() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples

        // First chunk - size equals available audio (up to chunkSamples max)
        let chunks = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))

        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].end - chunks[0].start, minChunkSamples, "First chunk size should match available audio")
    }

    func testChunkStartsAtCorrectIndex() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples

        let chunks = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))

        XCTAssertEqual(chunks[0].start, 0, "First chunk should start at index 0")
    }

    // MARK: - Duration Calculation Tests

    func testDefaultConfigDurations() {
        let config = BatchStyleStreamingConfig.default

        let chunkDuration = Double(config.chunkSamples) / Double(sampleRate)
        let overlapDuration = Double(config.overlapSamples) / Double(sampleRate)
        let strideDuration = Double(config.chunkSamples - config.overlapSamples) / Double(sampleRate)

        XCTAssertEqual(chunkDuration, 14.0, accuracy: 0.001, "Chunk should be 14 seconds")
        XCTAssertEqual(overlapDuration, 2.0, accuracy: 0.001, "Overlap should be 2 seconds")
        XCTAssertEqual(strideDuration, 12.0, accuracy: 0.001, "Stride should be 12 seconds")
    }

    func testTimeToFirstUpdate() {
        // With default config, first update comes after 14 seconds of audio
        let config = BatchStyleStreamingConfig.default
        let firstUpdateTime = Double(config.chunkSamples) / Double(sampleRate)

        XCTAssertEqual(firstUpdateTime, 14.0, accuracy: 0.001, "First update at 14 seconds")
    }

    func testTimeBetweenUpdates() {
        // After first update, subsequent updates come every stride duration
        let config = BatchStyleStreamingConfig.default
        let strideTime = Double(config.chunkSamples - config.overlapSamples) / Double(sampleRate)

        XCTAssertEqual(strideTime, 12.0, accuracy: 0.001, "Updates every 12 seconds after first")
    }

    // MARK: - Edge Cases

    func testEmptyAppend() {
        var simulator = ChunkBufferSimulator(config: .default)

        let chunks = simulator.appendSamples([])

        XCTAssertEqual(chunks.count, 0, "Empty append should not trigger chunk")
        XCTAssertEqual(simulator.sampleBuffer.count, 0, "Buffer should still be empty")
    }

    func testSmallIncrementalAppends() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples

        // Add 100 samples at a time until we reach minChunkSamples
        let batchSize = 100
        var totalChunks = 0

        for _ in 0..<(minChunkSamples / batchSize) {
            let chunks = simulator.appendSamples(Array(repeating: 0.5, count: batchSize))
            totalChunks += chunks.count
        }

        XCTAssertEqual(totalChunks, 1, "Should trigger exactly one chunk at minChunkSamples")
    }

    func testLargeAppendTriggersOneChunk() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples

        // Add 2x minChunk size at once
        let chunks = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples * 2))

        // Current implementation processes one chunk at a time
        XCTAssertEqual(chunks.count, 1, "Should process one chunk at a time")
    }

    // MARK: - Minimum Chunk Tests

    func testMinChunkNotTriggeredPrematurely() {
        var simulator = ChunkBufferSimulator(config: .default)
        let minChunkSamples = simulator.config.minChunkSamples

        // Add less than min chunk
        let chunks = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples - 1))

        XCTAssertEqual(chunks.count, 0, "Should not trigger with less than min chunk samples")
    }

    // MARK: - Custom Config Tests

    func testCustomConfigChunkBoundaries() {
        let customConfig = BatchStyleStreamingConfig(
            chunkSeconds: 5.0,
            overlapSeconds: 1.0,
            minChunkSeconds: 1.0
        )
        var simulator = ChunkBufferSimulator(config: customConfig)

        let chunkSamples = customConfig.chunkSamples  // 80000 (5s)
        let stride = chunkSamples - customConfig.overlapSamples  // 64000 (4s)

        // First chunk at 5s
        _ = simulator.appendSamples(Array(repeating: 0.5, count: chunkSamples))
        XCTAssertEqual(simulator.processedChunks, 1)

        // Second chunk at 4s more (stride)
        _ = simulator.appendSamples(Array(repeating: 0.5, count: stride))
        XCTAssertEqual(simulator.processedChunks, 2)
    }

    // MARK: - Buffer Memory Management Tests

    func testBufferDoesNotGrowUnbounded() {
        var simulator = ChunkBufferSimulator(config: .default)
        let overlapSamples = simulator.config.overlapSamples
        let minChunkSamples = simulator.config.minChunkSamples

        // Process 10 chunks
        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))
        for _ in 1..<10 {
            _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))
        }

        // Buffer should only contain overlap samples after processing
        XCTAssertLessThanOrEqual(
            simulator.sampleBuffer.count,
            overlapSamples + minChunkSamples,  // overlap + any unprocessed new samples
            "Buffer should not grow unbounded"
        )
    }

    func testBufferSizeAfterMultipleChunks() {
        var simulator = ChunkBufferSimulator(config: .default)
        let overlapSamples = simulator.config.overlapSamples
        let minChunkSamples = simulator.config.minChunkSamples

        // Process first chunk at minChunkSamples
        _ = simulator.appendSamples(Array(repeating: 0.5, count: minChunkSamples))

        XCTAssertEqual(
            simulator.sampleBuffer.count, overlapSamples,
            "After first chunk, buffer should only have overlap"
        )
    }
}
