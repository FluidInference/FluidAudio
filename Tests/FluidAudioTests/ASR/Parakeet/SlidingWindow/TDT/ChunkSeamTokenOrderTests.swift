import XCTest

@testable import FluidAudio

/// Regression tests for issue #825: the post-merge global timestamp sort
/// reordered tokens the merger had placed in correct linear (text) order,
/// interleaving subwords across a chunk seam and inverting word order. The
/// fix replaces the sort with `enforceMonotonicTimestamps`, which clamps
/// timestamps upward without reordering.
final class ChunkSeamTokenOrderTests: XCTestCase {

    private typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float, duration: Int)

    private func tok(_ id: Int, _ ts: Int) -> TokenWindow {
        (token: id, timestamp: ts, confidence: 0.9, duration: 1)
    }

    /// "im Frühjahr": the right window's "Früh" carries a frame index one
    /// lower than the left window's "im" at the seam. A timestamp sort would
    /// interleave them into "Früh im jahr" ("imüh Frjahr"); the clamp keeps
    /// the merged order.
    func testCrossWindowSubwordOrderPreserved() {
        let im = 11
        let fruh = 12
        let jahr = 13
        let merged = [tok(im, 160), tok(fruh, 159), tok(jahr, 161)]

        let result = ChunkProcessor.enforceMonotonicTimestamps(merged)

        XCTAssertEqual(result.map { $0.token }, [im, fruh, jahr], "token order must be preserved")
        XCTAssertEqual(result.map { $0.timestamp }, [160, 160, 161], "timestamps clamped monotonic")
        // A timestamp sort would have produced the corrupted order.
        XCTAssertEqual(merged.sorted { $0.timestamp < $1.timestamp }.map { $0.token }, [fruh, im, jahr])
    }

    /// "Für die Regale": the merger produced [Für, die, Regale] but "die"
    /// carries an earlier frame than "Für". A sort inverts to "die Für";
    /// the clamp keeps "Für die Regale".
    func testWordOrderInversionPrevented() {
        let fur = 21
        let die = 22
        let regale = 23
        let merged = [tok(fur, 160), tok(die, 158), tok(regale, 162)]

        let result = ChunkProcessor.enforceMonotonicTimestamps(merged)

        XCTAssertEqual(result.map { $0.token }, [fur, die, regale])
        XCTAssertEqual(result.map { $0.timestamp }, [160, 160, 162])
    }

    func testAlreadyMonotonicIsUnchanged() {
        let merged = [tok(1, 10), tok(2, 12), tok(3, 12), tok(4, 20)]
        let result = ChunkProcessor.enforceMonotonicTimestamps(merged)
        XCTAssertEqual(result.map { $0.token }, [1, 2, 3, 4])
        XCTAssertEqual(result.map { $0.timestamp }, [10, 12, 12, 20])
    }

    func testEqualTimestampsKeepEmissionOrder() {
        // Same-frame subwords ("P","un","kt" of "Punkt") must not be shuffled.
        let merged = [tok(100, 40), tok(101, 40), tok(102, 40)]
        let result = ChunkProcessor.enforceMonotonicTimestamps(merged)
        XCTAssertEqual(result.map { $0.token }, [100, 101, 102])
        XCTAssertEqual(result.map { $0.timestamp }, [40, 40, 40])
    }

    func testSingleAndEmptyAreNoOps() {
        XCTAssertEqual(ChunkProcessor.enforceMonotonicTimestamps([]).count, 0)
        let one = [tok(7, 5)]
        XCTAssertEqual(ChunkProcessor.enforceMonotonicTimestamps(one).map { $0.timestamp }, [5])
    }

    func testConfidenceAndDurationArePreserved() {
        let merged = [
            (token: 1, timestamp: 30, confidence: Float(0.8), duration: 3),
            (token: 2, timestamp: 28, confidence: Float(0.6), duration: 2),
        ]
        let result = ChunkProcessor.enforceMonotonicTimestamps(merged)
        XCTAssertEqual(result[1].timestamp, 30)
        XCTAssertEqual(result[1].confidence, 0.6)
        XCTAssertEqual(result[1].duration, 2)
    }
}
