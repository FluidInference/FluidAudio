import XCTest

@testable import FluidAudio

/// The overlap merge keeps one copy of every token both windows decoded.
/// Its timing should come from the window that heard the token away from
/// its own edge — see "Seam Timing" in Documentation/ASR/LongTranscription.md.
final class SeamTimingRealignmentTests: XCTestCase {

    private typealias Token = (token: Int, timestamp: Int, confidence: Float, duration: Int)

    /// Right window starts at frame 161 (12.88 s); the left window ends at
    /// frame 186. Left copies of the overlap tokens run early, as measured.
    private let rightStart = 161

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

    private func merge(rightWindowStartFrame: Int?) -> [Token] {
        ChunkProcessor(audioSamples: []).mergeTokenWindowsForTesting(
            left: left, right: right, rightWindowStartFrame: rightWindowStartFrame)
    }

    func testMatchedTokensTakeTheRightWindowsTiming() {
        let merged = merge(rightWindowStartFrame: rightStart)
        XCTAssertEqual(merged.map(\.token), [10, 11, 12, 13, 14, 15])
        XCTAssertEqual(merged.map(\.timestamp), [150, 163, 171, 179, 183, 190])
        XCTAssertEqual(merged.map(\.duration), [2, 2, 4, 3, 4, 2])
    }

    func testTokensInsideTheRightWindowsHeadKeepTheLeftTiming() {
        // Token 11's right copy is 3 frames into its window: not yet stable.
        let merged = merge(rightWindowStartFrame: rightStart)
        XCTAssertEqual(merged[1].timestamp, 163)
        XCTAssertEqual(merged[1].duration, 2)
    }

    func testIdentityStaysWithTheLeftWindow() {
        let merged = merge(rightWindowStartFrame: rightStart)
        for token in merged.dropLast() {
            XCTAssertEqual(token.confidence, 0.9, accuracy: 0.0001)
        }
    }

    func testTextIsUnchangedByTheRule() {
        XCTAssertEqual(
            merge(rightWindowStartFrame: rightStart).map(\.token),
            merge(rightWindowStartFrame: nil).map(\.token))
    }

    func testWithoutAWindowStartTheMergeIsUnchanged() {
        let merged = merge(rightWindowStartFrame: nil)
        XCTAssertEqual(merged.map(\.timestamp), [150, 163, 170, 176, 179, 190])
        XCTAssertEqual(merged.map(\.duration), [2, 2, 2, 1, 1, 2])
    }

    func testMergedTimestampsStayMonotonic() {
        let stamps = merge(rightWindowStartFrame: rightStart).map(\.timestamp)
        XCTAssertEqual(stamps, stamps.sorted())
    }

    func testEnabledByDefault() {
        XCTAssertTrue(ASRConfig.default.seamTimingRealignment)
    }
}
