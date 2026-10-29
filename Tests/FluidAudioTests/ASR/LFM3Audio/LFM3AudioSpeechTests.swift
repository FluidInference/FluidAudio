#if arch(arm64)
import XCTest

@testable import FluidAudio

final class LFM3AudioSpeechTests: XCTestCase {

    /// The top-k threshold from quickselect equals the k-th value of a descending sort (ties included).
    func testKthLargestMatchesSort() throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip("LFM3-Audio needs macOS 15 / iOS 18") }
        var random = LFM3AudioRandom(seed: 7)
        for trial in 0..<50 {
            let count = 2017
            var values = (0..<count).map { _ in Float(Int(random.next() % 4001) - 2000) / 100 }
            if trial % 5 == 0 { values = values.map { ($0 * 2).rounded() / 2 } }  // many ties
            let sorted = values.sorted(by: >)
            for k in [1, 2, 64, 128, 1000, count] {
                var scratch = values
                let value = scratch.withUnsafeMutableBufferPointer { buffer in
                    LFM3AudioDepthDecoder.kthLargest(buffer.baseAddress!, count: count, k: k)
                }
                XCTAssertEqual(value, sorted[k - 1], "trial \(trial) k \(k)")
            }
        }
    }

    func testKthLargestClampsArguments() throws {
        guard #available(macOS 15, iOS 18, *) else { throw XCTSkip("LFM3-Audio needs macOS 15 / iOS 18") }
        var values: [Float] = [3, 1, 2]
        values.withUnsafeMutableBufferPointer { b in
            XCTAssertEqual(LFM3AudioDepthDecoder.kthLargest(b.baseAddress!, count: 3, k: 0), 3)
            XCTAssertEqual(LFM3AudioDepthDecoder.kthLargest(b.baseAddress!, count: 3, k: 9), 1)
            XCTAssertEqual(LFM3AudioDepthDecoder.kthLargest(b.baseAddress!, count: 0, k: 1), -.infinity)
        }
    }

    // MARK: - Overlap-add

    /// Periodic Hann(1280), the shape of the codec's synthesis window.
    private func hann() -> [Float] {
        (0..<1280).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 1280)) }
    }

    private func frames(rows: Int, seed: UInt64) -> [Float] {
        var random = LFM3AudioRandom(seed: seed)
        return (0..<(rows * 1280)).map { _ in Float(Int(random.next() % 20001) - 10000) / 10000 }
    }

    /// Streaming in codec-sized pieces (F = 1 / 4 / 25 -> 6 / 24 / 150 rows) == one call over all rows.
    func testOverlapAddStreamingMatchesOneCall() throws {
        let rows = [6, 24, 150, 6, 24]
        let all = frames(rows: rows.reduce(0, +), seed: 3)
        let once = LFM3AudioOverlapAdd(window: hann(), maxRows: all.count / 1280)
        let expected = try all.withUnsafeBufferPointer {
            try once.add($0.baseAddress!, rows: all.count / 1280, finalize: true)
        }

        let stream = LFM3AudioOverlapAdd(window: hann(), maxRows: 150)
        var got: [Float] = []
        var offset = 0
        for (i, r) in rows.enumerated() {
            let piece = Array(all[(offset * 1280)..<((offset + r) * 1280)])
            got += try piece.withUnsafeBufferPointer {
                try stream.add($0.baseAddress!, rows: r, finalize: i == rows.count - 1)
            }
            offset += r
        }
        XCTAssertEqual(got.count, expected.count)
        XCTAssertEqual(expected.count, (offset - 1) * 320 + 1280 - 2 * 480)
        for i in 0..<got.count { XCTAssertEqual(got[i], expected[i], accuracy: 1e-5 * max(1, abs(expected[i]))) }
    }

    func testOverlapAddFinalizeFlushesTheTailOnce() throws {
        let ola = LFM3AudioOverlapAdd(window: hann(), maxRows: 6)
        let piece = frames(rows: 6, seed: 5)
        let first = try piece.withUnsafeBufferPointer { try ola.add($0.baseAddress!, rows: 6, finalize: false) }
        XCTAssertEqual(first.count, 5 * 320 + 1280 - 480 - 960)
        let flush = try piece.withUnsafeBufferPointer { try ola.add($0.baseAddress!, rows: 0, finalize: true) }
        XCTAssertEqual(flush.count, 960 - 480)
        XCTAssertFalse(flush.contains { $0.isNaN })
        // finalized: no tail left, so no 0 / 0 samples
        let again = try piece.withUnsafeBufferPointer { try ola.add($0.baseAddress!, rows: 0, finalize: true) }
        XCTAssertTrue(again.isEmpty)
        XCTAssertThrowsError(
            try piece.withUnsafeBufferPointer { try ola.add($0.baseAddress!, rows: 7, finalize: true) })
    }

    func testDepthFrameScheduleRevealsFourCodebooksPerPass() {
        var random = LFM3AudioRandom(seed: 1)
        var scratch: [Float] = []
        let frame = LFM3AudioDepthFrame.draw(&random, scratch: &scratch)
        XCTAssertEqual(Set(frame.order), Set(0..<12))
        XCTAssertEqual(frame.noise.count, 12 * 2017)
        let schedule = frame.schedule
        for pass in 0..<3 {
            let row = schedule[(pass * 12)..<((pass + 1) * 12)]
            XCTAssertEqual(row.filter { $0 == 1 }.count, 4)
            for i in (4 * pass)..<(4 * pass + 4) { XCTAssertEqual(row[pass * 12 + frame.order[i]], 1) }
        }
        // The scratch buffer is reused: a second draw keeps its size and gives new noise.
        let second = LFM3AudioDepthFrame.draw(&random, scratch: &scratch)
        XCTAssertEqual(scratch.count, 12 * 2017)
        XCTAssertNotEqual(second.noise, frame.noise)
    }
}
#endif
