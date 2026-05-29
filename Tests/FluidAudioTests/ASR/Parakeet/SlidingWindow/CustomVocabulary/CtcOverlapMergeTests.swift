import XCTest

@testable import FluidAudio

/// Tests for `CtcKeywordSpotter.mergeOverlapFrame`, the logmeanexp-based
/// chunk-boundary averager that replaced the previous arithmetic mean of
/// log-probabilities.
final class CtcOverlapMergeTests: XCTestCase {

    // MARK: - Helpers

    /// `logmeanexp(a, b)` reference implementation in `Double` precision.
    private func referenceLogMeanExp(_ a: Float, _ b: Float) -> Float {
        let pa = exp(Double(a))
        let pb = exp(Double(b))
        return Float(log((pa + pb) / 2.0))
    }

    // MARK: - Probability-space mean

    func testEqualInputsReturnSameValue() {
        // logmeanexp(x, x) = x for all x.
        let inputs: [Float] = [-0.1, -1.0, -3.0, -10.0]
        for x in inputs {
            let merged = CtcKeywordSpotter.mergeOverlapFrame(
                existing: [x], incoming: [x]
            )
            XCTAssertEqual(merged[0], x, accuracy: 1e-5, "logmeanexp(\(x), \(x)) should equal \(x)")
        }
    }

    func testMatchesReferenceImplementation() {
        // For modest log-prob values, the float result should match the
        // double-precision reference within a tight tolerance.
        let pairs: [(Float, Float)] = [
            (-0.1, -0.5),
            (-1.0, -3.0),
            (-2.5, -7.0),
            (-0.05, -0.05),
        ]
        for (a, b) in pairs {
            let merged = CtcKeywordSpotter.mergeOverlapFrame(
                existing: [a], incoming: [b]
            )
            let expected = referenceLogMeanExp(a, b)
            XCTAssertEqual(
                merged[0], expected, accuracy: 1e-4,
                "logmeanexp(\(a), \(b)) should equal \(expected)")
        }
    }

    /// Critical correctness property: the previous implementation took the
    /// arithmetic mean of log-probs, i.e. `(a + b) / 2`. That equals
    /// `log(sqrt(p_a * p_b))` — the geometric mean. A correct probability-
    /// space mean must be **strictly larger** whenever `a ≠ b` (AM ≥ GM).
    func testProbabilitySpaceMeanExceedsLogSpaceMean() {
        let a: Float = -0.1  // p ≈ 0.905
        let b: Float = -3.0  // p ≈ 0.050
        let logSpaceMean = (a + b) / 2.0  // old behavior
        let merged = CtcKeywordSpotter.mergeOverlapFrame(
            existing: [a], incoming: [b]
        )
        XCTAssertGreaterThan(
            merged[0], logSpaceMean,
            "Probability-space mean must exceed log-space arithmetic mean for unequal inputs"
        )
        // Sanity: result must lie between the two inputs.
        XCTAssertLessThanOrEqual(merged[0], max(a, b))
        XCTAssertGreaterThanOrEqual(merged[0], min(a, b))
    }

    // MARK: - Numerical stability

    func testStableForLargelyNegativeValues() {
        // Inputs around -50 would overflow naive exp(); the max-shift
        // formulation must keep this finite.
        let merged = CtcKeywordSpotter.mergeOverlapFrame(
            existing: [-50.0], incoming: [-49.0]
        )
        XCTAssertTrue(merged[0].isFinite)
        // logmeanexp(-50, -49) = -49 + log((e^-1 + 1) / 2) ≈ -49.379
        XCTAssertEqual(merged[0], -49.379, accuracy: 0.01)
    }

    func testNegativeInfinityInBothInputsPropagates() {
        let merged = CtcKeywordSpotter.mergeOverlapFrame(
            existing: [-Float.infinity], incoming: [-Float.infinity]
        )
        XCTAssertEqual(merged[0], -Float.infinity)
    }

    func testNegativeInfinityInOneInputDefersToOther() {
        // logmeanexp(-inf, x) = x - log(2)
        let log2: Float = 0.69314718
        let x: Float = -2.0
        let merged = CtcKeywordSpotter.mergeOverlapFrame(
            existing: [-Float.infinity], incoming: [x]
        )
        XCTAssertEqual(merged[0], x - log2, accuracy: 1e-5)
    }

    // MARK: - Vector behavior

    func testEntireVectorMerged() {
        let existing: [Float] = [-0.1, -2.0, -5.0, -10.0]
        let incoming: [Float] = [-2.0, -0.1, -10.0, -5.0]
        let merged = CtcKeywordSpotter.mergeOverlapFrame(
            existing: existing, incoming: incoming
        )
        XCTAssertEqual(merged.count, existing.count)
        // Symmetry: logmeanexp is commutative.
        for j in 0..<merged.count {
            let expected = referenceLogMeanExp(existing[j], incoming[j])
            XCTAssertEqual(merged[j], expected, accuracy: 1e-4)
        }
    }

    func testEmptyVectorReturnsEmpty() {
        let merged = CtcKeywordSpotter.mergeOverlapFrame(
            existing: [], incoming: []
        )
        XCTAssertTrue(merged.isEmpty)
    }
}
