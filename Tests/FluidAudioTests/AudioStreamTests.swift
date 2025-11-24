import XCTest

@testable import FluidAudio

final class AudioStreamTests: XCTestCase {

    func testMostRecentReadChunkReturnsSamplesAndStartTime() throws {
        let chunkDuration: TimeInterval = 0.01
        var stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent
        )

        let chunkSize = stream.chunkSize
        let samples = (0..<chunkSize).map { Float($0) }
        try stream.write(from: samples)

        guard let result = stream.readChunkIfAvailable() else {
            XCTFail("Expected a chunk to be available")
            return
        }

        XCTAssertEqual(result.chunk, samples)
        XCTAssertEqual(result.chunkStartTime, 0, accuracy: 1e-6)
        XCTAssertFalse(stream.hasNewChunk)
    }

    func testBoundCallbackProducesSequentialMostRecentChunks() throws {
        let chunkDuration: TimeInterval = 0.01
        var stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent
        )

        let chunkSize = stream.chunkSize
        let chunkCount = 3
        let expectation = expectation(description: "Received bound chunks")
        expectation.expectedFulfillmentCount = chunkCount

        stream.bind { chunk, time in
            guard let firstSample = chunk.first else {
                XCTFail("Chunk should contain samples")
                return
            }

            let index = Int(firstSample) / chunkSize
            XCTAssertTrue(0 <= index && index < chunkCount)

            let expectedSamples = (0..<chunkSize).map { Float(index * chunkSize + $0) }
            XCTAssertEqual(Array(chunk), expectedSamples)
            XCTAssertEqual(time, Double(index) * chunkDuration, accuracy: 1e-6)
            expectation.fulfill()
        }

        for index in 0..<chunkCount {
            let base = index * chunkSize
            let payload = (0..<chunkSize).map { Float(base + $0) }
            try stream.write(from: payload)
        }

        wait(for: [expectation], timeout: 1)
    }

    func testFixedHopChunksPreserveOverlap() throws {
        let chunkDuration: TimeInterval = 0.02
        let chunkSkip: TimeInterval = 0.01
        let sampleRate = 16_000.0
        var stream = try AudioStream(
            chunkDuration: chunkDuration,
            chunkSkip: chunkSkip,
            streamStartTime: chunkDuration,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .startSilent,
            sampleRate: sampleRate
        )

        let chunkSize = stream.chunkSize
        let hopSize = Int(round(sampleRate * chunkSkip))
        let overlapSampleCount = chunkSize - hopSize

        // Warm up the buffer so the initial zero padding is removed.
        let warmupData = Array(repeating: Float(1), count: hopSize)
        try stream.write(from: warmupData)
        _ = stream.readChunkIfAvailable()

        let incrementalData = (0..<hopSize).map { Float($0) }
        try stream.write(from: incrementalData)
        guard let nextChunk = stream.readChunkIfAvailable()?.chunk else {
            XCTFail("Expected fixed-hop chunk after warm-up")
            return
        }

        XCTAssertEqual(nextChunk.count, chunkSize)
        XCTAssertEqual(Array(nextChunk.prefix(overlapSampleCount)), Array(repeating: 1, count: overlapSampleCount))
        XCTAssertEqual(Array(nextChunk.suffix(hopSize)), incrementalData)
    }

    func testGapFillShiftsChunkStartTimeForMostRecentStream() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 16_000.0
        var stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent,
            sampleRate: sampleRate
        )

        let chunkSize = stream.chunkSize
        let gapSamples = 20
        let gapDuration = Double(gapSamples) / sampleRate
        let timestamp = chunkDuration + gapDuration

        let samples = (0..<chunkSize).map { Float($0 + 1) }
        try stream.write(from: samples, atTime: timestamp)

        guard let (chunk, startTime) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk after writing data with a timestamp gap")
            return
        }

        // Most-recent strategy drops the oldest padding once the chunk is ready.
        XCTAssertEqual(chunk, samples)
        XCTAssertEqual(startTime, gapDuration, accuracy: 1e-6)
    }

    func testRampUpChunkSizeIncreasesUntilFullChunk() throws {
        let chunkDuration: TimeInterval = 0.03
        let chunkSkip: TimeInterval = 0.01
        let sampleRate = 16_000.0
        var stream = try AudioStream(
            chunkDuration: chunkDuration,
            chunkSkip: chunkSkip,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .rampUpChunkSize,
            sampleRate: sampleRate
        )

        let hopSize = Int(round(sampleRate * chunkSkip))
        let fullChunkSize = stream.chunkSize

        // First write fills to hopSize
        try stream.write(from: Array(repeating: Float(1), count: hopSize))
        guard let firstChunk = stream.readChunkIfAvailable()?.chunk else {
            XCTFail("Expected initial ramp-up chunk")
            return
        }
        XCTAssertEqual(firstChunk.count, hopSize)

        // Second write increases to 2 * hopSize
        try stream.write(from: Array(repeating: Float(2), count: hopSize))
        guard let secondChunk = stream.readChunkIfAvailable()?.chunk else {
            XCTFail("Expected second ramp-up chunk")
            return
        }
        XCTAssertEqual(secondChunk.count, hopSize * 2)

        // Third write reaches the full chunk size
        try stream.write(from: Array(repeating: Float(3), count: hopSize))
        guard let thirdChunk = stream.readChunkIfAvailable()?.chunk else {
            XCTFail("Expected full-size chunk after ramp-up")
            return
        }
        XCTAssertEqual(thirdChunk.count, fullChunkSize)
    }
}
