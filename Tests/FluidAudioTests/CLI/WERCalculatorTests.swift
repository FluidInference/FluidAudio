import XCTest

@testable import FluidAudioCLI

/// Insertion/deletion labelling of the shared WER scorer.
///
/// The backtrace used to label a reference word missing from the hypothesis
/// as an "insertion" and an extra hypothesis word as a "deletion" (swapped).
/// WER was unaffected (it sums all three) but any consumer that reads the
/// breakdown — the recall metric in `enhance-benchmark`, the S/D/I lines in
/// `canary-transcribe` and `tts-asr-verify` — was wrong; an empty hypothesis
/// scored 100% recall.
final class WERCalculatorTests: XCTestCase {

    func testEmptyHypothesisIsAllDeletions() {
        let m = WERCalculator.calculateWERMetrics(hypothesis: "", reference: "one two three")
        XCTAssertEqual(m.totalWords, 3)
        XCTAssertEqual(m.deletions, 3)
        XCTAssertEqual(m.insertions, 0)
        XCTAssertEqual(m.substitutions, 0)
        XCTAssertEqual(m.wer, 1.0, accuracy: 1e-9)
    }

    func testEmptyReferenceIsAllInsertions() {
        let m = WERCalculator.calculateWERMetrics(hypothesis: "one two", reference: "")
        XCTAssertEqual(m.totalWords, 0)
        XCTAssertEqual(m.insertions, 2)
        XCTAssertEqual(m.deletions, 0)
    }

    func testMixedEdits() {
        // ref: a b c d ; hyp: a x c d e  -> 1 substitution (b->x), 1 insertion (e)
        let m = WERCalculator.calculateWERMetrics(hypothesis: "a x c d e", reference: "a b c d")
        XCTAssertEqual(m.substitutions, 1)
        XCTAssertEqual(m.insertions, 1)
        XCTAssertEqual(m.deletions, 0)
        XCTAssertEqual(m.wer, 0.5, accuracy: 1e-9)

        // ref: a b c d ; hyp: a c -> 2 deletions
        let d = WERCalculator.calculateWERMetrics(hypothesis: "a c", reference: "a b c d")
        XCTAssertEqual(d.deletions, 2)
        XCTAssertEqual(d.insertions, 0)
        XCTAssertEqual(d.substitutions, 0)
    }

    func testRecallFromBreakdown() {
        // Recall as enhance-benchmark computes it: (N - D - S) / N.
        let m = WERCalculator.calculateWERMetrics(hypothesis: "a c z", reference: "a b c d")
        let recall = Double(m.totalWords - m.deletions - m.substitutions) / Double(m.totalWords)
        XCTAssertEqual(recall, 0.5, accuracy: 1e-9)
        let empty = WERCalculator.calculateWERMetrics(hypothesis: "", reference: "a b c d")
        XCTAssertEqual(Double(empty.totalWords - empty.deletions - empty.substitutions), 0)
    }
}
