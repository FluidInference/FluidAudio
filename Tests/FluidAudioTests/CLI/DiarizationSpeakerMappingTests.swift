#if os(macOS)
import XCTest

@testable import FluidAudio
@testable import FluidAudioCLI

/// Tests for the diarization scorer's speaker mapping (issue #922).
///
/// `computeSpeakerMapping` builds its speaker index orders from dictionary
/// keys, and the assignment solvers keep the first-encountered winner among
/// tied overlaps. Before the key arrays were sorted, per-instance random
/// dictionary order made the winning mapping — and with it JER — drift
/// between runs on tied confusion-matrix entries.
final class DiarizationSpeakerMappingTests: XCTestCase {

    private func segment(_ speaker: String, _ start: Float, _ end: Float) -> TimedSpeakerSegment {
        TimedSpeakerSegment(
            speakerId: speaker, embedding: [], startTimeSeconds: start, endTimeSeconds: end,
            qualityScore: 1.0)
    }

    /// Two predicted speakers tie exactly on overlap with one ground-truth
    /// speaker (4.75 s each after the 0.25 s collar). Only one can win the
    /// assignment; the winner must not depend on dictionary key order.
    func testTiedOverlapsMapDeterministically() {
        let groundTruth = [segment("spk_a", 0, 10)]
        let predicted = [
            segment("left", 0, 5),
            segment("right", 5, 10),
        ]

        let first = DiarizationMetricsCalculator.offlineMetrics(
            predicted: predicted, groundTruth: groundTruth)
        XCTAssertEqual(first.speakerMapping.count, 1)
        XCTAssertEqual(Set(first.speakerMapping.values), ["spk_a"])

        // Every call regroups segments into fresh dictionaries, so each
        // iteration samples a new per-instance key order.
        for _ in 0..<50 {
            let metrics = DiarizationMetricsCalculator.offlineMetrics(
                predicted: predicted, groundTruth: groundTruth)
            XCTAssertEqual(metrics.speakerMapping, first.speakerMapping)
            XCTAssertEqual(metrics.jer, first.jer)
            XCTAssertEqual(metrics.der, first.der)
        }
    }

    /// Unambiguous overlaps must still produce the correct mapping after the
    /// key arrays are sorted.
    func testUnambiguousMappingIsCorrect() {
        let groundTruth = [
            segment("alice", 0, 10),
            segment("bob", 11, 20),
        ]
        let predicted = [
            segment("s1", 0, 9),
            segment("s2", 11.5, 20),
        ]

        let metrics = DiarizationMetricsCalculator.offlineMetrics(
            predicted: predicted, groundTruth: groundTruth)
        XCTAssertEqual(metrics.speakerMapping, ["s1": "alice", "s2": "bob"])
    }
}
#endif
