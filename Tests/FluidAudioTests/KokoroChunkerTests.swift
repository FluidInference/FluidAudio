import XCTest

@testable import FluidAudio

@available(macOS 13.0, *)
final class KokoroChunkerTests: XCTestCase {

    private let allowed: Set<String> = ["A", "B", "C", "F", "I", "a", "n", "d", " ", "."]

    func testCaseSensitiveLexiconPreferredForAbbreviations() {
        let text = "F.B.I and C.I.A"

        let lowerLexicon: [String: [String]] = [
            "f": ["F"],
            "b": ["B"],
            "i": ["I"],
            "and": ["a", "n", "d"],
            "c": ["C"],
            "a": ["A"],
        ]

        let caseSensitiveLexicon: [String: [String]] = [
            "F": ["F"],
            "B": ["B"],
            "I": ["I"],
            "F.B.I": ["F", "B", "I"],
            "C": ["C"],
            "A": ["A"],
            "C.I.A": ["C", "I", "A"],
        ]

        let chunks = KokoroChunker.chunk(
            text: text,
            wordToPhonemes: lowerLexicon,
            caseSensitiveLexicon: caseSensitiveLexicon,
            targetTokens: 120,
            hasLanguageToken: false,
            allowedPhonemes: allowed
        )

        XCTAssertEqual(chunks.count, 1)
        guard let chunk = chunks.first else {
            XCTFail("Missing chunk output")
            return
        }
        XCTAssertTrue(chunk.text.contains("F. B. I"))
        XCTAssertTrue(chunk.text.contains("C. I. A"))
        XCTAssertTrue(chunk.phonemes.contains("A"), "Expected final letter phoneme to be preserved")
    }

    func testRunOnTextRespectsTokenBudget() {
        let text = "alpha beta gamma delta epsilon zeta eta theta"

        let lexicon: [String: [String]] = [
            "alpha": ["a"],
            "beta": ["b"],
            "gamma": ["g"],
            "delta": ["d"],
            "epsilon": ["e"],
            "zeta": ["z"],
            "eta": ["h"],
            "theta": ["t"],
        ]

        let allowed: Set<String> = ["a", "b", "g", "d", "e", "z", "h", "t", " "]

        let chunks = KokoroChunker.chunk(
            text: text,
            wordToPhonemes: lexicon,
            caseSensitiveLexicon: [:],
            targetTokens: 20,
            hasLanguageToken: false,
            allowedPhonemes: allowed
        )

        XCTAssertGreaterThan(chunks.count, 1, "Expected run-on text to be split under tight token budget")
        chunks.forEach { chunk in
            XCTAssertFalse(chunk.words.isEmpty)
            XCTAssertLessThanOrEqual(chunk.phonemes.count, 6)
        }
    }
}
