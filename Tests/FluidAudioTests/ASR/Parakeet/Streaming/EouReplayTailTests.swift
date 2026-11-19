import XCTest

@testable import FluidAudio

/// The audio `rearm(keepTranscript:)` replays is whatever the buffer shifted out since the debounce anchor, so these
/// tests pin what a replay would be fed and how far the conversation clock rewinds.
final class EouReplayTailTests: XCTestCase {

    func testTakeReturnsEveryShiftInOrderAndEmptiesTheTail() {
        var tail = EouReplayTail(maxSamples: 100)
        tail.record([1, 2, 3][...])
        tail.record([4, 5][...])

        let taken = tail.take()

        XCTAssertEqual(taken.samples, [1, 2, 3, 4, 5])
        XCTAssertEqual(taken.chunkCount, 2)
        XCTAssertTrue(tail.samples.isEmpty)
        XCTAssertEqual(tail.chunkCount, 0)
    }

    /// A replay that starts at a chunk boundary must rewind by the chunks it drops, or timestamps drift.
    func testCapDropsWholeOldestShiftsAndKeepsChunkCountInStep() {
        var tail = EouReplayTail(maxSamples: 5)
        tail.record([1, 2, 3][...])
        tail.record([4, 5, 6][...])
        tail.record([7, 8][...])

        XCTAssertEqual(tail.samples, [4, 5, 6, 7, 8])
        XCTAssertEqual(tail.chunkCount, 2)
    }

    func testClearForgetsTheAnchorsAudio() {
        var tail = EouReplayTail(maxSamples: 100)
        tail.record([1, 2, 3][...])

        tail.clear()

        XCTAssertEqual(tail, EouReplayTail(maxSamples: 100))
    }
}
