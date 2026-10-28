import Foundation
import XCTest

@testable import FluidAudio

/// Tests for `CanaryManager.packSegments`, which groups VAD speech segments into
/// contiguous ≤ 15 s spans for seam-free long-form decoding.
final class CanarySegmentPackingTests: XCTestCase {

    private let maxN = CanaryConfig.maxSamples
    private let second = CanaryConfig.sampleRate

    func testEmptyInputGivesNoSpans() {
        XCTAssertEqual(CanaryManager.packSegments([], maxSamples: maxN), [])
    }

    func testSegmentsThatFitArePackedIntoOneSpan() {
        // 0–4 s, 5–9 s, 10–14 s: the span 0–14 s fits one window.
        let segments = [0..<(4 * second), (5 * second)..<(9 * second), (10 * second)..<(14 * second)]
        XCTAssertEqual(CanaryManager.packSegments(segments, maxSamples: maxN), [0..<(14 * second)])
    }

    func testSpanBreaksBeforeExceedingWindow() {
        // Adding 12–16 s to a span starting at 0 would make 16 s; it starts a new span.
        let segments = [0..<(6 * second), (7 * second)..<(11 * second), (12 * second)..<(16 * second)]
        XCTAssertEqual(
            CanaryManager.packSegments(segments, maxSamples: maxN),
            [0..<(11 * second), (12 * second)..<(16 * second)])
    }

    func testSpanKeepsSilenceBetweenPackedSegments() {
        // The span runs from the first segment's start to the last's end, pauses included.
        let segments = [(1 * second)..<(2 * second), (8 * second)..<(9 * second)]
        XCTAssertEqual(CanaryManager.packSegments(segments, maxSamples: maxN), [(1 * second)..<(9 * second)])
    }

    func testOverlongSegmentBecomesItsOwnSpan() {
        // A 20 s segment cannot fit; it is emitted alone so the overlap chunker handles it.
        let segments = [0..<(3 * second), (4 * second)..<(24 * second), (25 * second)..<(27 * second)]
        XCTAssertEqual(
            CanaryManager.packSegments(segments, maxSamples: maxN),
            [0..<(3 * second), (4 * second)..<(24 * second), (25 * second)..<(27 * second)])
    }

    func testExactWindowLengthFits() {
        let segments = [0..<(7 * second), (8 * second)..<maxN]
        XCTAssertEqual(CanaryManager.packSegments(segments, maxSamples: maxN), [0..<maxN])
    }

    func testEmptySegmentsAreSkipped() {
        let segments = [(2 * second)..<(2 * second), (3 * second)..<(5 * second)]
        XCTAssertEqual(CanaryManager.packSegments(segments, maxSamples: maxN), [(3 * second)..<(5 * second)])
    }
}
