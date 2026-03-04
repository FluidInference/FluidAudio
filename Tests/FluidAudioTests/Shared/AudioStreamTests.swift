import AVFoundation
import XCTest

@testable import FluidAudio

final class AudioStreamTests: XCTestCase {

    func testInitializationValidationErrors() throws {
        XCTAssertThrowsError(try AudioStream(chunkDuration: 0)) { error in
            XCTAssertTrue(error is AudioStreamError)
        }

        XCTAssertThrowsError(try AudioStream(chunkDuration: 0.01, chunkSkip: 0)) { error in
            XCTAssertTrue(error is AudioStreamError)
        }

        XCTAssertThrowsError(try AudioStream(chunkDuration: 0.01, chunkSkip: 0.02)) { error in
            XCTAssertTrue(error is AudioStreamError)
        }

        XCTAssertThrowsError(
            try AudioStream(
                chunkDuration: 0.02,
                chunkSkip: 0.01,
                bufferCapacitySeconds: 0.01
            )
        ) { error in
            XCTAssertTrue(error is AudioStreamError)
        }
    }

    func testReadChunkUnavailableUntilFilled() throws {
        let chunkDuration: TimeInterval = 0.02
        let sampleRate = 16_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .waitForFullChunk,
            sampleRate: sampleRate
        )

        XCTAssertNil(stream.readChunkIfAvailable())

        let halfChunk = Array(repeating: Float(1), count: stream.chunkSize / 2)
        try stream.write(from: halfChunk)
        XCTAssertNil(stream.readChunkIfAvailable(), "Should not produce chunk until full")

        let remaining = Array(repeating: Float(2), count: stream.chunkSize - halfChunk.count)
        try stream.write(from: remaining)
        guard let (chunk, start) = stream.readChunkIfAvailable() else {
            XCTFail("Expected full chunk after buffer filled")
            return
        }
        XCTAssertEqual(chunk.count, stream.chunkSize)
        XCTAssertEqual(start, 0, accuracy: 1e-6)
        XCTAssertFalse(stream.hasNewChunk)
    }

    func testFixedSkipAdvancesStartTimeAndMaintainsOverlap() throws {
        let chunkDuration: TimeInterval = 0.02
        let chunkSkip: TimeInterval = 0.01
        let sampleRate = 16_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            chunkSkip: chunkSkip,
            streamStartTime: 0.0,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .waitForFullChunk,
            sampleRate: sampleRate
        )

        let hopSize = Int(round(sampleRate * chunkSkip))
        let first = Array(repeating: Float(1), count: stream.chunkSize)
        try stream.write(from: first)
        guard let (firstChunk, firstStart) = stream.readChunkIfAvailable() else {
            XCTFail("Expected first fixed-skip chunk")
            return
        }
        XCTAssertEqual(firstChunk, first)
        XCTAssertEqual(firstStart, 0, accuracy: 1e-6)

        let secondTail = [Float](repeating: Float(2), count: hopSize)
        try stream.write(from: secondTail)
        guard let (secondChunk, secondStart) = stream.readChunkIfAvailable() else {
            XCTFail("Expected second fixed-skip chunk")
            return
        }
        XCTAssertEqual(secondStart, chunkSkip, accuracy: 1e-6)
        let expectedPrefix = [Float](repeating: 1.0, count: stream.chunkSize - hopSize)
        XCTAssertEqual(Array(secondChunk.prefix(stream.chunkSize - hopSize)), expectedPrefix)
        XCTAssertEqual(Array(secondChunk.suffix(hopSize)), secondTail)
    }

    func testTimestampRollbackRewindsNewestSamples() throws {
        let chunkDuration: TimeInterval = 1.0
        let sampleRate = 4.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent,
            sampleRate: sampleRate
        )

        try stream.write(from: [0, 1, 2, 3], atTime: nil)
        try stream.write(from: [10, 11, 12, 13], atTime: 0.5)

        guard let (chunk, start) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk after rollback write")
            return
        }
        XCTAssertEqual(chunk, [10, 11, 12, 13])
        XCTAssertEqual(start, -0.5, accuracy: 1e-6)
    }

    func testLargeForwardJumpProducesSequentialChunksFixedSkip() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .startSilent,
            sampleRate: sampleRate,
            bufferCapacitySeconds: 0.1
        )

        let chunkSize = stream.chunkSize
        let payload = (0..<15).map { Float($0) }

        try stream.write(from: payload, atTime: 0.04)

        var chunks: [([Float], TimeInterval)] = []
        while let result = stream.readChunkIfAvailable() {
            chunks.append((result.chunk, result.chunkStartTime))
        }

        let expectedTimestamps: [TimeInterval] = [0.0, 0.01, 0.02, 0.03]
        let chunkTimestamps = chunks.map(\.1)

        XCTAssertEqual(chunks.count, 4)
        for (timestamp, expected) in zip(chunkTimestamps, expectedTimestamps) {
            XCTAssertEqual(timestamp, expected, accuracy: 1e-6)
        }

        let expectedChunks: [[Float]] = [
            Array(repeating: 0, count: chunkSize),
            Array(repeating: 0, count: chunkSize),
            Array(repeating: 0, count: chunkSize - 5) + Array(payload.prefix(5)),
            Array(payload.suffix(chunkSize)),
        ]
        XCTAssertEqual(chunks.map(\.0), expectedChunks)
    }

    func testLargeForwardJumpAlignsMostRecentChunkTimestamp() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent,
            sampleRate: sampleRate
        )

        let payload = (0..<15).map { Float($0) }
        try stream.write(from: payload, atTime: 0.04)

        guard let (chunk, start) = stream.readChunkIfAvailable() else {
            XCTFail("Expected most-recent chunk after large forward jump")
            return
        }

        XCTAssertEqual(chunk.count, stream.chunkSize)
        XCTAssertEqual(chunk, Array(payload.suffix(stream.chunkSize)))
        XCTAssertEqual(start, 0.03, accuracy: 1e-6)
    }

    func testNegativeTimestampDropsOldDataWithoutCrash() throws {
        let chunkDuration: TimeInterval = 0.02
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent,
            sampleRate: sampleRate
        )

        try stream.write(from: (0..<20).map(Float.init), atTime: nil)
        try stream.write(from: (100..<105).map(Float.init), atTime: -0.05)
        try stream.write(from: (200..<215).map(Float.init), atTime: nil)

        guard let (chunk, start) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk after negative timestamp rollback")
            return
        }

        XCTAssertEqual(start, -0.055, accuracy: 1e-6)
        XCTAssertEqual(chunk, (100..<105).map(Float.init) + (200..<215).map(Float.init))
    }

    func testBoundCallbackReceivesTimestampedMultiChunks() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .startSilent,
            sampleRate: sampleRate
        )

        var receivedTimes: [TimeInterval] = []
        let expectedTimes: [TimeInterval] = [0.01, 0.02]  // the first chunk should be skipped because it won't fit in the buffer
        stream.bind { _, time in
            receivedTimes.append(time)
        }

        try stream.write(from: Array(0..<20).map { Float($0) }, atTime: 0.03)

        for (receivedTime, expectedTime) in zip(receivedTimes, expectedTimes) {
            XCTAssertEqual(receivedTime, expectedTime, accuracy: 1e-6)
        }
        stream.unbind()
    }

    func testOscillatingTimestampsStillProduceOrderedChunks() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .startSilent,
            sampleRate: sampleRate,
            bufferCapacitySeconds: 1
        )

        var samples: [[Float]] = [
            (0..<10).map(Float.init),
            (10..<20).map(Float.init),
            (20..<50).map(Float.init),
        ]

        var timestamps: [TimeInterval] = [0.02, 0.00, 0.06]
        var chunks: [[Float]] = []

        var startTimes: [TimeInterval] = []
        while samples.isEmpty == false {
            let sample = samples.removeFirst()
            let time = timestamps.removeFirst()
            try stream.write(from: sample, atTime: time)
        }
        while let next = stream.readChunkIfAvailable() {
            startTimes.append(next.chunkStartTime)
            chunks.append(next.chunk)
        }

        XCTAssertEqual(startTimes.count, 7)
        for window in zip(startTimes, startTimes.dropFirst()) {
            XCTAssertLessThan(window.0, window.1 + 1e-9)
        }

        // 7th chunk should be 40-49
        // 6th chunk should be 30-39
        // 5th chunk should be 20-29
        // 4th chunk should be silent due to being wiped
        // 3rd chunk should be silent due to being wiped
        // 2nd chunk should be silent due to being wiped
        // 1st chunk should be 10-19

        let silentChunk = [Float](repeating: 0, count: 10)

        XCTAssertEqual(chunks[0], (10..<20).map(Float.init))
        XCTAssertEqual(chunks[1], silentChunk)
        XCTAssertEqual(chunks[2], silentChunk)
        XCTAssertEqual(chunks[3], silentChunk)
        XCTAssertEqual(chunks[4], (20..<30).map(Float.init))
        XCTAssertEqual(chunks[5], (30..<40).map(Float.init))
        XCTAssertEqual(chunks[6], (40..<50).map(Float.init))

        XCTAssertEqual(startTimes.first!, -0.01, accuracy: 1e-6)
        XCTAssertEqual(startTimes.last!, 0.05, accuracy: 1e-6)
    }

    func testStartTimeOffsetPropagatesToChunks() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let streamStart: TimeInterval = 5.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: streamStart,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent,
            sampleRate: sampleRate
        )

        try stream.write(from: Array(0..<stream.chunkSize).map { Float($0) })
        guard let (chunk, start) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk for non-zero streamStartTime")
            return
        }
        XCTAssertEqual(chunk.count, stream.chunkSize)
        XCTAssertEqual(start, streamStart, accuracy: 1e-6)
    }

    func testBoundPreventsManualReadsUntilUnbound() throws {
        let chunkDuration: TimeInterval = 0.01
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent
        )

        var callbacks: [(ArraySlice<Float>, TimeInterval)] = []
        stream.bind { chunk, time in
            callbacks.append((chunk, time))
        }

        let chunkSize = stream.chunkSize
        let firstPayload = (0..<chunkSize).map { Float($0) }
        try stream.write(from: firstPayload)

        XCTAssertNil(stream.readChunkIfAvailable(), "Bound stream should not expose chunks via read")
        XCTAssertEqual(callbacks.count, 1)
        XCTAssertEqual(Array(callbacks[0].0), firstPayload)

        stream.unbind()

        let secondPayload = (0..<chunkSize).map { Float($0 + chunkSize) }
        try stream.write(from: secondPayload)

        guard let (chunk, time) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk after unbinding")
            return
        }

        XCTAssertEqual(chunk, secondPayload)
        XCTAssertEqual(time, chunkDuration, accuracy: 1e-6)
    }

    func testRampUpWithFixedSkipAdvancesStartTimes() throws {
        let chunkDuration: TimeInterval = 0.03
        let chunkSkip: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            chunkSkip: chunkSkip,
            streamStartTime: 0.0,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .rampUpChunkSize,
            sampleRate: sampleRate
        )

        let hopSize = Int(round(sampleRate * chunkSkip))
        let payload = Array(repeating: Float(1), count: hopSize)

        try stream.write(from: payload)
        let first = stream.readChunkIfAvailable()
        XCTAssertEqual(first?.chunk.count, hopSize)
        XCTAssertEqual(first!.chunkStartTime, 0, accuracy: 1e-6)

        try stream.write(from: payload)
        let second = stream.readChunkIfAvailable()
        XCTAssertEqual(second?.chunk.count, hopSize * 2)
        XCTAssertEqual(second!.chunkStartTime, 0, accuracy: 1e-6)

        try stream.write(from: payload)
        let third = stream.readChunkIfAvailable()
        XCTAssertEqual(third?.chunk.count, stream.chunkSize)
        XCTAssertEqual(third!.chunkStartTime, 0, accuracy: 1e-6)

        try stream.write(from: payload)
        let fourth = stream.readChunkIfAvailable()
        XCTAssertEqual(fourth?.chunk.count, stream.chunkSize)
        XCTAssertEqual(fourth!.chunkStartTime, chunkSkip, accuracy: 1e-6)
    }

    func testBufferBackpressureDropsOldestAndAdvancesStart() throws {
        let chunkDuration: TimeInterval = 0.02
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .startSilent,
            sampleRate: sampleRate,
            bufferCapacitySeconds: 0.03
        )

        let payload = (0..<50).map { Float($0) }
        try stream.write(from: payload)

        guard let (chunk, start) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk after oversized write")
            return
        }

        XCTAssertEqual(chunk.count, stream.chunkSize)
        XCTAssertEqual(chunk, Array(payload.suffix(stream.chunkSize)))
        XCTAssertEqual(start, 0.03, accuracy: 1e-6)
    }

    func testAppendZerosProducesMultipleChunksWithPadding() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .waitForFullChunk,
            sampleRate: sampleRate
        )

        let chunkSize = stream.chunkSize
        let timestamp: TimeInterval = 0.02
        let payload = (1...5).map { Float($0) }

        try stream.write(from: payload, atTime: timestamp)

        guard let first = stream.readChunkIfAvailable() else {
            XCTFail("Expected first padded chunk")
            return
        }
        XCTAssertEqual(first.chunk.count, chunkSize)
        XCTAssertEqual(first.chunkStartTime, 0, accuracy: 1e-6)
        XCTAssertEqual(first.chunk, Array(repeating: Float(0), count: chunkSize))

        guard let second = stream.readChunkIfAvailable() else {
            XCTFail("Expected second padded chunk")
            return
        }
        XCTAssertEqual(second.chunkStartTime, chunkDuration, accuracy: 1e-6)
        let expectedSecond = Array(repeating: Float(0), count: chunkSize - payload.count) + payload
        XCTAssertEqual(second.chunk, expectedSecond)
    }

    func testAVAudioPCMBufferWriteRespectsResampling() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 16_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .waitForFullChunk,
            sampleRate: sampleRate
        )

        let samples = (0..<stream.chunkSize).map { Float($0) }
        let buffer = try makePCMBuffer(sampleRate: sampleRate, samples: samples)

        try stream.write(from: buffer)
        guard let (chunk, time) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk from AVAudioPCMBuffer")
            return
        }

        XCTAssertEqual(chunk, samples)
        XCTAssertEqual(time, 0, accuracy: 1e-6)
    }

    func testCMSampleBufferWriteRespectsResampling() throws {
        let chunkDuration: TimeInterval = 0.01
        let sampleRate = 16_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .waitForFullChunk,
            sampleRate: sampleRate
        )

        let samples = (0..<stream.chunkSize).map { Float($0 + 1) }
        let sampleBuffer = try makeCMSampleBuffer(sampleRate: sampleRate, samples: samples)

        try stream.write(from: sampleBuffer)
        guard let (chunk, time) = stream.readChunkIfAvailable() else {
            XCTFail("Expected chunk from CMSampleBuffer")
            return
        }

        XCTAssertEqual(chunk, samples)
        XCTAssertEqual(time, 0, accuracy: 1e-6)
    }

    func testOverlappingChunksDrainInOrder() throws {
        let chunkDuration: TimeInterval = 0.02
        let chunkSkip: TimeInterval = 0.01
        let sampleRate = 1_000.0
        let stream = try AudioStream(
            chunkDuration: chunkDuration,
            chunkSkip: chunkSkip,
            streamStartTime: 0.0,
            chunkingStrategy: .useMostRecent,
            startupStrategy: .waitForFullChunk,
            sampleRate: sampleRate
        )

        var received: [(ArraySlice<Float>, TimeInterval)] = []
        stream.bind { chunk, time in
            received.append((chunk, time))
        }

        let firstPayload = (0..<stream.chunkSize).map { Float($0) }
        try stream.write(from: firstPayload)
        XCTAssertEqual(received.count, 1)
        XCTAssertEqual(Array(received[0].0), firstPayload)
        XCTAssertEqual(received[0].1, 0, accuracy: 1e-6)

        received.removeAll()

        let hop = stream.chunkSize - stream.overlapSize
        let secondPayload = (0..<hop).map { Float(100 + $0) }
        try stream.write(from: secondPayload)

        XCTAssertEqual(received.count, 1)
        let expectedChunk = Array(firstPayload.suffix(stream.overlapSize)) + secondPayload
        XCTAssertEqual(Array(received[0].0), expectedChunk)
        XCTAssertEqual(received[0].1, chunkSkip, accuracy: 1e-6)

        stream.unbind()
    }

    func testMostRecentReadChunkReturnsSamplesAndStartTime() throws {
        let chunkDuration: TimeInterval = 0.01
        let stream = try AudioStream(
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
        let stream = try AudioStream(
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
        let stream = try AudioStream(
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
        let stream = try AudioStream(
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
        let stream = try AudioStream(
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

    // MARK: - Helpers

    private enum AudioStreamTestError: Error {
        case failedToCreateFormat
        case failedToCreateBuffer
        case failedToCreateSampleBuffer
        case failedToCreateBlockBuffer(OSStatus)
    }

    private func makePCMBuffer(
        sampleRate: Double,
        samples: [Float]
    ) throws -> AVAudioPCMBuffer {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw AudioStreamTestError.failedToCreateFormat
        }

        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(samples.count)
            )
        else {
            throw AudioStreamTestError.failedToCreateBuffer
        }

        buffer.frameLength = AVAudioFrameCount(samples.count)
        if let channel = buffer.floatChannelData?.pointee {
            channel.update(from: samples, count: samples.count)
        }
        return buffer
    }

    private func makeCMSampleBuffer(
        sampleRate: Double,
        samples: [Float]
    ) throws -> CMSampleBuffer {
        var asbd = AudioStreamBasicDescription(
            mSampleRate: sampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: UInt32(MemoryLayout<Float>.size),
            mFramesPerPacket: 1,
            mBytesPerFrame: UInt32(MemoryLayout<Float>.size),
            mChannelsPerFrame: 1,
            mBitsPerChannel: UInt32(MemoryLayout<Float>.size * 8),
            mReserved: 0
        )

        var formatDescription: CMAudioFormatDescription?
        let formatStatus = CMAudioFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            asbd: &asbd,
            layoutSize: 0,
            layout: nil,
            magicCookieSize: 0,
            magicCookie: nil,
            extensions: nil,
            formatDescriptionOut: &formatDescription
        )
        guard formatStatus == noErr, let formatDescription else {
            throw AudioStreamTestError.failedToCreateFormat
        }

        let dataLength = samples.count * MemoryLayout<Float>.size
        var blockBuffer: CMBlockBuffer?
        let blockStatus = CMBlockBufferCreateWithMemoryBlock(
            allocator: kCFAllocatorDefault,
            memoryBlock: nil,
            blockLength: dataLength,
            blockAllocator: kCFAllocatorDefault,
            customBlockSource: nil,
            offsetToData: 0,
            dataLength: dataLength,
            flags: 0,
            blockBufferOut: &blockBuffer
        )
        guard blockStatus == kCMBlockBufferNoErr, let blockBuffer else {
            throw AudioStreamTestError.failedToCreateBlockBuffer(blockStatus)
        }

        let replaceStatus = samples.withUnsafeBytes { bytes in
            CMBlockBufferReplaceDataBytes(
                with: bytes.baseAddress!,
                blockBuffer: blockBuffer,
                offsetIntoDestination: 0,
                dataLength: dataLength
            )
        }
        guard replaceStatus == kCMBlockBufferNoErr else {
            throw AudioStreamTestError.failedToCreateSampleBuffer
        }

        var timing = CMSampleTimingInfo(
            duration: CMTime(
                value: 1,
                timescale: Int32(sampleRate)
            ),
            presentationTimeStamp: .zero,
            decodeTimeStamp: .invalid
        )
        var sampleSize = MemoryLayout<Float>.size

        var sampleBuffer: CMSampleBuffer?
        let status = CMSampleBufferCreate(
            allocator: kCFAllocatorDefault,
            dataBuffer: blockBuffer,
            dataReady: true,
            makeDataReadyCallback: nil,
            refcon: nil,
            formatDescription: formatDescription,
            sampleCount: samples.count,
            sampleTimingEntryCount: 1,
            sampleTimingArray: &timing,
            sampleSizeEntryCount: 1,
            sampleSizeArray: &sampleSize,
            sampleBufferOut: &sampleBuffer
        )

        guard status == noErr, let sampleBuffer else {
            throw AudioStreamTestError.failedToCreateSampleBuffer
        }

        return sampleBuffer
    }
}
