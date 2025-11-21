import XCTest

@testable import FluidAudio

final class AudioStreamTests: XCTestCase {

    func testBackAlignedReadChunkReturnsSamplesAndStartTime() throws {
        let chunkDuration: TimeInterval = 0.01
        var stream = AudioStream(
            chunkDuration: chunkDuration,
            atTime: chunkDuration,
            alignment: .backAligned
        )

        let chunkSize = stream.chunkSize
        let samples = (0..<chunkSize).map { Float($0) }
        stream.write(from: samples)

        guard let result = stream.readChunk() else {
            XCTFail("Expected a chunk to be available")
            return
        }

        XCTAssertEqual(result.chunk, samples)
        XCTAssertEqual(result.chunkStartTime, 0, accuracy: 1e-6)
        XCTAssertFalse(stream.isChunkReady)
    }

    func testBoundCallbackProducesSequentialBackAlignedChunks() {
        let chunkDuration: TimeInterval = 0.01
        var stream = AudioStream(
            chunkDuration: chunkDuration,
            strideDuration: chunkDuration,
            atTime: chunkDuration,
            alignment: .backAligned
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
            stream.write(from: payload)
        }

        wait(for: [expectation], timeout: 1)
    }

    func testFrontAlignedChunksPreserveOverlapFromPreviousChunk() {
        let chunkDuration: TimeInterval = 0.02
        let strideDuration: TimeInterval = 0.01
        var stream = AudioStream(
            chunkDuration: chunkDuration,
            strideDuration: strideDuration,
            atTime: chunkDuration,
            alignment: .frontAligned
        )

        let chunkSize = stream.chunkSize
        let hopSize = Int(round(AudioStream.sampleRate * strideDuration))
        let overlapSampleCount = chunkSize - hopSize

        // Warm up the buffer so the initial zero padding is removed.
        let warmupData = Array(repeating: Float(1), count: hopSize)
        stream.write(from: warmupData)
        _ = stream.readChunk()

        let incrementalData = (0..<hopSize).map { Float($0) }
        stream.write(from: incrementalData)
        guard let nextChunk = stream.readChunk()?.chunk else {
            XCTFail("Expected front-aligned chunk after warm-up")
            return
        }

        XCTAssertEqual(nextChunk.count, chunkSize)
        XCTAssertEqual(Array(nextChunk.prefix(overlapSampleCount)), Array(repeating: 1, count: overlapSampleCount))
        XCTAssertEqual(Array(nextChunk.suffix(hopSize)), incrementalData)
    }

    func testProcessGapsInsertsSilenceForFrontAlignedStream() {
        let chunkDuration: TimeInterval = 0.01
        var stream = AudioStream(
            chunkDuration: chunkDuration,
            strideDuration: chunkDuration,
            atTime: chunkDuration,
            alignment: .frontAligned,
            processGaps: true
        )

        let chunkSize = stream.chunkSize
        let gapSamples = 20
        let gapDuration = Double(gapSamples) / AudioStream.sampleRate
        let timestamp = chunkDuration + gapDuration

        let samples = (0..<chunkSize).map { Float($0 + 1) }
        stream.write(from: samples, atTime: timestamp)

        guard let chunk = stream.readChunk()?.chunk else {
            XCTFail("Expected chunk after writing data with a timestamp gap")
            return
        }

        XCTAssertEqual(Array(chunk.prefix(gapSamples)), Array(repeating: 0, count: gapSamples))
        XCTAssertEqual(Array(chunk.dropFirst(gapSamples)), Array(samples.prefix(chunkSize - gapSamples)))
    }
}
