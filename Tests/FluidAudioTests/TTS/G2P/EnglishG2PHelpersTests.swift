import XCTest

@testable import FluidAudio

/// Pins the pure-logic helpers in `EnglishG2PHelpers` that are now shared
/// between Kokoro and StyleTTS2. No model assets are touched here.
final class EnglishG2PHelpersTests: XCTestCase {

    // MARK: - stemInflected

    /// Tiny synthetic lexicon: just enough stems to drive the stemmer.
    private let lex: [String: [String]] = [
        "jump": ["ʤ", "ˈ", "ʌ", "m", "p"],
        "phrase": ["f", "ɹ", "ˈ", "e", "ɪ", "z"],
        "make": ["m", "ˈ", "e", "ɪ", "k"],
        "run": ["ɹ", "ˈ", "ʌ", "n"],
        "bus": ["b", "ˈ", "ʌ", "s"],
        "rate": ["ɹ", "ˈ", "e", "ɪ", "t"],
    ]

    func testStemPluralS() {
        // "buses" — stem ends in /s/ (sibilant), inserts ᵻ before z.
        let result = EnglishG2PHelpers.stemInflected("buses", lexicon: lex)
        XCTAssertEqual(result?.suffix(2).map { String($0) }, ["ᵻ", "z"])
    }

    func testStemPastEd() {
        // "jumped" — stem ends in /p/ (voiceless stop), suffix = t.
        let result = EnglishG2PHelpers.stemInflected("jumped", lexicon: lex)
        XCTAssertEqual(result?.last, "t")
    }

    func testStemSilentEEd() {
        // "phrased" — strips just the trailing 'd' to reach "phrase".
        // Stem ends in /z/ (voiced fricative), suffix = d.
        let result = EnglishG2PHelpers.stemInflected("phrased", lexicon: lex)
        XCTAssertEqual(result?.last, "d")
    }

    func testStemIngDoubledConsonant() {
        // "running" — drops "ning", reaches "run". Stem ends in /n/, suffix = ɪ ŋ.
        let result = EnglishG2PHelpers.stemInflected("running", lexicon: lex)
        XCTAssertEqual(result?.suffix(2).map { String($0) }, ["ɪ", "ŋ"])
    }

    func testStemIngSilentE() {
        // "making" — drops "ing", reaches "make". Stem ends in /k/, suffix = ɪ ŋ.
        let result = EnglishG2PHelpers.stemInflected("making", lexicon: lex)
        XCTAssertEqual(result?.suffix(2).map { String($0) }, ["ɪ", "ŋ"])
    }

    func testStemReturnsNilForUnknownStem() {
        XCTAssertNil(EnglishG2PHelpers.stemInflected("xyzzied", lexicon: lex))
        XCTAssertNil(EnglishG2PHelpers.stemInflected("nope", lexicon: lex))
    }

    func testStemRespectsAllowedFilter() {
        // Filter strips all lex tokens → stem produces nothing useful.
        let result = EnglishG2PHelpers.stemInflected(
            "jumped", lexicon: lex, allowed: Set<String>(["xx"]))
        XCTAssertNil(result)
    }

    // MARK: - spelledOutTokens

    func testSpelledOutTokensBasic() {
        XCTAssertEqual(
            EnglishG2PHelpers.spelledOutTokens(for: "25"),
            ["twenty", "five"])
    }

    func testSpelledOutTokensYear() {
        let result = EnglishG2PHelpers.spelledOutTokens(for: "2025")
        XCTAssertEqual(result, ["two", "thousand", "twenty", "five"])
    }

    func testSpelledOutTokensRejectsNonNumeric() {
        XCTAssertNil(EnglishG2PHelpers.spelledOutTokens(for: "abc"))
        XCTAssertNil(EnglishG2PHelpers.spelledOutTokens(for: "1.5"))
        XCTAssertNil(EnglishG2PHelpers.spelledOutTokens(for: ""))
    }

    // MARK: - letterPronunciations

    func testLetterPronunciationsCoversAllLowercaseAscii() {
        for letter in "abcdefghijklmnopqrstuvwxyz" {
            XCTAssertNotNil(
                EnglishG2PHelpers.letterPronunciations[String(letter)],
                "Missing entry for '\(letter)'")
        }
    }
}
