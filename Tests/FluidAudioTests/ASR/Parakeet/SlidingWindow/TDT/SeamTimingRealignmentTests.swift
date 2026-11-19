import XCTest

@testable import FluidAudio

/// The overlap merge keeps one copy of every token both windows decoded.
/// Its timing should come from the window that heard the token away from
/// its own edge — see "Seam Timing" in Documentation/ASR/LongTranscription.md.
final class SeamTimingRealignmentTests: XCTestCase {

    private typealias Token = (token: Int, timestamp: Int, confidence: Float, duration: Int)
    private typealias Windows = ChunkProcessor.WindowFrames
    private typealias Seam = ChunkProcessor.SeamWindows

    /// The mel-context layout: the left window's audio ends at frame 186
    /// (14.88 s); the right window decodes from frame 161 (12.88 s), with
    /// no warm-up prefix. Left copies of the overlap tokens run early, as
    /// measured.
    private let seam = Seam(
        left: Windows(decodeStart: 0, end: 186),
        right: Windows(decodeStart: 161, end: 347))

    private var left: [Token] {
        [
            (token: 10, timestamp: 150, confidence: 0.9, duration: 2),
            (token: 11, timestamp: 163, confidence: 0.9, duration: 2),  // right copy inside the head guard
            (token: 12, timestamp: 170, confidence: 0.9, duration: 2),
            (token: 13, timestamp: 176, confidence: 0.9, duration: 1),
            (token: 14, timestamp: 179, confidence: 0.9, duration: 1),
        ]
    }

    private var right: [Token] {
        [
            (token: 11, timestamp: 164, confidence: 0.8, duration: 3),
            (token: 12, timestamp: 171, confidence: 0.8, duration: 4),
            (token: 13, timestamp: 179, confidence: 0.8, duration: 3),
            (token: 14, timestamp: 183, confidence: 0.8, duration: 4),
            (token: 15, timestamp: 190, confidence: 0.8, duration: 2),
        ]
    }

    private func merge(_ left: [Token], _ right: [Token], seam: Seam?) -> [Token] {
        ChunkProcessor(audioSamples: []).mergeTokenWindowsForTesting(left: left, right: right, seam: seam)
    }

    private func merge(seam: Seam?) -> [Token] {
        merge(left, right, seam: seam)
    }

    // MARK: Contiguous matches

    func testMatchedTokensTakeTheRightWindowsTiming() {
        let merged = merge(seam: seam)
        XCTAssertEqual(merged.map(\.token), [10, 11, 12, 13, 14, 15])
        XCTAssertEqual(merged.map(\.timestamp), [150, 163, 171, 179, 183, 190])
        XCTAssertEqual(merged.map(\.duration), [2, 2, 4, 3, 4, 2])
    }

    func testTokensInsideTheRightWindowsHeadKeepTheLeftTiming() {
        // Token 11's right copy is 3 frames into its window, and its left
        // copy is 23 frames from the left window's end: the left copy is the
        // one farther from an edge.
        let merged = merge(seam: seam)
        XCTAssertEqual(merged[1].timestamp, 163)
        XCTAssertEqual(merged[1].duration, 2)
    }

    func testIdentityStaysWithTheLeftWindow() {
        let merged = merge(seam: seam)
        for token in merged.dropLast() {
            XCTAssertEqual(token.confidence, 0.9, accuracy: 0.0001)
        }
    }

    func testTextIsUnchangedByTheRule() {
        XCTAssertEqual(merge(seam: seam).map(\.token), merge(seam: nil).map(\.token))
    }

    func testWithoutASeamTheMergeIsUnchanged() {
        let merged = merge(seam: nil)
        XCTAssertEqual(merged.map(\.timestamp), [150, 163, 170, 176, 179, 190])
        XCTAssertEqual(merged.map(\.duration), [2, 2, 2, 1, 1, 2])
    }

    func testMergedTimestampsStayMonotonic() {
        let stamps = merge(seam: seam).map(\.timestamp)
        XCTAssertEqual(stamps, stamps.sorted())
    }

    func testEnabledByDefault() {
        XCTAssertTrue(ASRConfig.default.seamTimingRealignment)
        XCTAssertTrue(UnifiedConfig().seamTimingRealignment)
    }

    // MARK: The guard is measured from the decode start

    func testWarmUpPrefixCountsTowardsTheHeadGuard() {
        // The right window decoded a warm-up prefix from frame 150 before
        // emitting from 161 (the v3 no-mel path and the end-aligned final
        // window). Token 11 at frame 164 is then 14 frames into the decode,
        // well past the guard, and takes the right timing it would have been
        // denied had the guard been measured from the first emitted frame.
        let warmedUp = Seam(
            left: Windows(decodeStart: 0, end: 186),
            right: Windows(decodeStart: 150, end: 347))
        let merged = merge(seam: warmedUp)
        XCTAssertEqual(merged.map(\.timestamp), [150, 164, 171, 179, 183, 190])
        XCTAssertEqual(merged[1].duration, 3)
    }

    // MARK: Minimum overlap

    func testMinimumOverlapSeamTakesTheCopyFartherFromItsEdge() {
        // Silence-aligned starts can leave only the 6-frame minimum overlap
        // (frames 180–186), so every right copy is inside the head guard.
        // Each pair then goes to whichever copy is farther from its own
        // window's edge.
        let left: [Token] = [
            (token: 20, timestamp: 170, confidence: 0.9, duration: 2),
            (token: 21, timestamp: 180, confidence: 0.9, duration: 2),  // 6 from the tail; right copy 2 into the head
            (token: 22, timestamp: 182, confidence: 0.9, duration: 1),  // 4 from the tail; right copy 3 into the head
            (token: 23, timestamp: 184, confidence: 0.9, duration: 1),  // 2 from the tail; right copy 5 into the head
        ]
        let right: [Token] = [
            (token: 21, timestamp: 182, confidence: 0.8, duration: 3),
            (token: 22, timestamp: 183, confidence: 0.8, duration: 2),
            (token: 23, timestamp: 185, confidence: 0.8, duration: 2),
            (token: 24, timestamp: 188, confidence: 0.8, duration: 2),
        ]
        let narrow = Seam(
            left: Windows(decodeStart: 0, end: 186),
            right: Windows(decodeStart: 180, end: 366))
        let merged = merge(left, right, seam: narrow)
        XCTAssertEqual(merged.map(\.token), [20, 21, 22, 23, 24])
        XCTAssertEqual(merged.map(\.timestamp), [170, 180, 182, 185, 188])
        XCTAssertEqual(merged.map(\.duration), [2, 2, 1, 2, 2])
    }

    // MARK: Unmatched tokens between matches

    func testGapTokensMoveWithTheMatchBeforeThem() {
        // The left window heard an extra token (99) between two matches,
        // and the contiguous matcher's longest run is 11–12 on one side of
        // it and 13–14 on the other — fewer than the pair minimum, so the
        // merge goes through the LCS branch. Token 99 keeps its place and
        // moves by the same delta as the match before it (+1), instead of
        // keeping a left-clock timestamp that the monotonic clamp would
        // have flattened onto its neighbour.
        let left: [Token] = [
            (token: 10, timestamp: 150, confidence: 0.9, duration: 2),
            (token: 11, timestamp: 168, confidence: 0.9, duration: 2),
            (token: 12, timestamp: 170, confidence: 0.9, duration: 2),
            (token: 99, timestamp: 172, confidence: 0.9, duration: 1),
            (token: 13, timestamp: 176, confidence: 0.9, duration: 1),
            (token: 14, timestamp: 179, confidence: 0.9, duration: 1),
        ]
        let right: [Token] = [
            (token: 11, timestamp: 169, confidence: 0.8, duration: 3),
            (token: 12, timestamp: 171, confidence: 0.8, duration: 4),
            (token: 13, timestamp: 179, confidence: 0.8, duration: 3),
            (token: 14, timestamp: 183, confidence: 0.8, duration: 4),
            (token: 15, timestamp: 190, confidence: 0.8, duration: 2),
        ]
        let merged = merge(left, right, seam: seam)
        XCTAssertEqual(merged.map(\.token), [10, 11, 12, 99, 13, 14, 15])
        XCTAssertEqual(merged.map(\.timestamp), [150, 169, 171, 173, 179, 183, 190])
        let stamps = merged.map(\.timestamp)
        XCTAssertEqual(stamps, stamps.sorted())
        // Without the seam the gap token is untouched.
        XCTAssertEqual(merge(left, right, seam: nil).map(\.timestamp), [150, 168, 170, 172, 176, 179, 190])
    }

    func testGapTokensAreHeldAtTheNextMatchWhenTheShiftWouldOvertakeIt() {
        // The match before the gap moves by +7 and the one after it by +4,
        // so the gap token's shift (173 → 180) would overtake its right-hand
        // neighbour at 178; it is held there instead.
        let left: [Token] = [
            (token: 10, timestamp: 150, confidence: 0.9, duration: 2),
            (token: 11, timestamp: 163, confidence: 0.9, duration: 2),
            (token: 12, timestamp: 170, confidence: 0.9, duration: 2),
            (token: 99, timestamp: 173, confidence: 0.9, duration: 1),
            (token: 13, timestamp: 174, confidence: 0.9, duration: 1),
            (token: 14, timestamp: 179, confidence: 0.9, duration: 1),
        ]
        let right: [Token] = [
            (token: 11, timestamp: 164, confidence: 0.8, duration: 3),
            (token: 12, timestamp: 177, confidence: 0.8, duration: 4),
            (token: 13, timestamp: 178, confidence: 0.8, duration: 3),
            (token: 14, timestamp: 183, confidence: 0.8, duration: 4),
            (token: 15, timestamp: 190, confidence: 0.8, duration: 2),
        ]
        let merged = merge(left, right, seam: seam)
        XCTAssertEqual(merged.map(\.token), [10, 11, 12, 99, 13, 14, 15])
        XCTAssertEqual(merged.map(\.timestamp), [150, 163, 177, 178, 178, 183, 190])
        let stamps = merged.map(\.timestamp)
        XCTAssertEqual(stamps, stamps.sorted())
    }
}
