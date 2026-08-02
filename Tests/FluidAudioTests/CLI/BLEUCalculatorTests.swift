#if os(macOS)
import XCTest

@testable import FluidAudioCLI

final class BLEUCalculatorTests: XCTestCase {

    func testTokenizeSplitsPunctuation() {
        XCTAssertEqual(
            BLEUCalculator.tokenize("Hallo, Welt! Es geht."),
            ["Hallo", ",", "Welt", "!", "Es", "geht", "."])
        XCTAssertEqual(BLEUCalculator.tokenize("  a  b "), ["a", "b"])
        XCTAssertEqual(BLEUCalculator.tokenize(""), [])
    }

    func testTokenizeKeepsDigitInternalPunctuation() {
        // mteval-13a keeps . and , attached when both neighbors are digits.
        XCTAssertEqual(
            BLEUCalculator.tokenize("3,5 Millionen um 2.5 Uhr."),
            ["3,5", "Millionen", "um", "2.5", "Uhr", "."])
        XCTAssertEqual(BLEUCalculator.tokenize("1,234.56"), ["1,234.56"])
        // Digit on one side only still splits.
        XCTAssertEqual(BLEUCalculator.tokenize("5, dann"), ["5", ",", "dann"])
    }

    func testPerfectMatchIs100() {
        let s = ["Das Wetter ist heute ungewöhnlich warm, oder?"]
        XCTAssertEqual(BLEUCalculator.corpusBLEU(hypotheses: s, references: s), 100.0, accuracy: 1e-9)
    }

    func testDisjointIsZero() {
        let bleu = BLEUCalculator.corpusBLEU(
            hypotheses: ["a b c d e"], references: ["v w x y z"])
        XCTAssertEqual(bleu, 0.0)
    }

    func testCaseSensitive() {
        let bleu = BLEUCalculator.corpusBLEU(
            hypotheses: ["das wetter ist heute sehr warm"],
            references: ["Das Wetter ist heute sehr warm"])
        XCTAssertLessThan(bleu, 100.0)
    }

    func testBrevityPenaltyAppliesToShortHypothesis() {
        // Hypothesis is a strict prefix: all n-gram precisions are 1.0, so the
        // score is exactly the brevity penalty exp(1 - r/h).
        let bleu = BLEUCalculator.corpusBLEU(
            hypotheses: ["a b c d e"], references: ["a b c d e f g h i j"])
        XCTAssertEqual(bleu, exp(1.0 - 10.0 / 5.0) * 100.0, accuracy: 1e-6)
    }

    func testCorpusPoolsAcrossSegments() {
        // Second segment supplies the 4-grams the first lacks; corpus-level
        // pooling must not zero out like per-sentence scoring would.
        let bleu = BLEUCalculator.corpusBLEU(
            hypotheses: ["x y", "a b c d e f"],
            references: ["x q", "a b c d e f"])
        XCTAssertGreaterThan(bleu, 0.0)
        XCTAssertLessThan(bleu, 100.0)
    }
}
#endif
