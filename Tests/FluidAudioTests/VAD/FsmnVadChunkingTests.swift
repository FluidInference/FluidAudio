import XCTest

@testable import FluidAudio

/// Regression tests for long-audio chunk tiling (PR #653 review): the valid frontend
/// convolutions make each chunk yield 40 ms less than it spans, so back-to-back chunking
/// compressed the timeline by 40 ms per boundary (~4.7 s over an hour). The schedule must
/// keep every concatenated frame on the absolute 10 ms grid with drift independent of
/// chunk count. Frame counts mirror the published preprocessor geometry via
/// `lfrFrameCount`; no model inference is involved.
final class FsmnVadChunkingTests: XCTestCase {

    /// Scorer stand-in that tags each output frame with its absolute frame index, derived
    /// only from the chunk's sample offset — mismatches surface as value != position.
    private func absoluteFrameIndices(_ range: Range<Int>) -> [Float] {
        let base = range.lowerBound / FsmnVadManager.hopSamples
        return (0..<FsmnVadManager.lfrFrameCount(samples: range.count)).map { Float(base + $0) }
    }

    func testFullChunkFrameCountMatchesReview() {
        // 488,320-sample chunk: 3050 fbank frames -> 3048 output frames (review arithmetic).
        XCTAssertEqual(FsmnVadManager.lfrFrameCount(samples: 488_320), 3048)
        XCTAssertEqual(FsmnVadManager.lfrFrameCount(samples: 399), 0)
    }

    func testChunkOutputsTileAbsoluteGrid() throws {
        // ~100 minutes: enough for >100 chunk boundaries.
        let totalSamples = 100 * 60 * 16_000
        var chunkStarts: [Int] = []
        let sil = FsmnVadManager.concatenateChunks(sampleCount: totalSamples) { range in
            chunkStarts.append(range.lowerBound)
            XCTAssertEqual(range.lowerBound % FsmnVadManager.hopSamples, 0)
            return absoluteFrameIndices(range)
        }
        XCTAssertGreaterThan(chunkStarts.count, 100)
        // Every kept frame sits at its absolute index: no gaps, overlaps, or drift.
        for (index, value) in sil.enumerated() {
            if Int(value) != index {
                XCTFail("frame \(index) carries absolute index \(Int(value))")
                break
            }
        }
        // Full coverage up to the frontend's edge loss at the very end (< 50 ms, not per chunk).
        let wholeFileFrames = FsmnVadManager.lfrFrameCount(samples: totalSamples)
        XCTAssertGreaterThanOrEqual(sil.count, wholeFileFrames - FsmnVadManager.lfrPadFrames)
        XCTAssertLessThanOrEqual(sil.count, wholeFileFrames)
    }

    func testHourLongAudioHasNoCumulativeDrift() {
        // Pre-fix, an hour lost ~40 ms x 117 boundaries ≈ 468 frames.
        let totalSamples = 60 * 60 * 16_000
        let sil = FsmnVadManager.concatenateChunks(sampleCount: totalSamples) { range in
            absoluteFrameIndices(range)
        }
        XCTAssertGreaterThanOrEqual(sil.count, 360_000 - 5)
        XCTAssertLessThanOrEqual(sil.count, 360_000)
    }

    func testShortAudioSingleChunkUnchanged() {
        let totalSamples = 20 * 16_000
        var calls = 0
        let sil = FsmnVadManager.concatenateChunks(sampleCount: totalSamples) { range in
            calls += 1
            XCTAssertEqual(range, 0..<totalSamples)
            return absoluteFrameIndices(range)
        }
        XCTAssertEqual(calls, 1)
        XCTAssertEqual(sil.count, FsmnVadManager.lfrFrameCount(samples: totalSamples))
    }

    func testTooShortAudioYieldsNoFrames() {
        let sil = FsmnVadManager.concatenateChunks(sampleCount: 300) { _ in [] }
        XCTAssertTrue(sil.isEmpty)
    }

    func testOddLengthTailTerminates() {
        // A tail that lands just past a boundary must not loop or emit misaligned frames.
        let totalSamples = 488_320 + 7_213
        let sil = FsmnVadManager.concatenateChunks(sampleCount: totalSamples) { range in
            absoluteFrameIndices(range)
        }
        for (index, value) in sil.enumerated() {
            if Int(value) != index {
                XCTFail("frame \(index) carries absolute index \(Int(value))")
                break
            }
        }
        let wholeFileFrames = FsmnVadManager.lfrFrameCount(samples: totalSamples)
        XCTAssertGreaterThanOrEqual(sil.count, wholeFileFrames - FsmnVadManager.lfrPadFrames)
    }
}
