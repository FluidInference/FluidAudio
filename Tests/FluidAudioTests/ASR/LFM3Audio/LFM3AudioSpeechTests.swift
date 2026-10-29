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
