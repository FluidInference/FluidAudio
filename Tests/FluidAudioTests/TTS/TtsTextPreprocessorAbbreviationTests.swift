import XCTest

@testable import FluidAudio

final class TtsTextPreprocessorAbbreviationTests: XCTestCase {

    // Abbreviations ending in a period used to slip past the replacement because
    // the regex relied on a trailing `\b` boundary, which never matches when the
    // next character is whitespace (`.` + ` ` is non-word→non-word). The fix
    // switches those entries to a lookahead that accepts whitespace, EOS, or any
    // non-word/non-dot character.

    func testDrAbbreviationMidSentenceIsExpanded() {
        let input = "Currently Dr. Morita holds a certification."
        let output = TtsTextPreprocessor.preprocess(input)
        XCTAssertFalse(output.contains("Dr."), "Dr. should be expanded, got: \(output)")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor' expansion, got: \(output)")
    }

    func testMrAndMrsAbbreviations() {
        let output = TtsTextPreprocessor.preprocess("Mr. Smith met Mrs. Jones.")
        XCTAssertFalse(output.contains("Mr."))
        XCTAssertFalse(output.contains("Mrs."))
        XCTAssertTrue(output.contains("Mister"))
        XCTAssertTrue(output.contains("Missus"))
    }

    func testProfAbbreviation() {
        let output = TtsTextPreprocessor.preprocess("Prof. Chen lectures today.")
        XCTAssertFalse(output.contains("Prof."))
        XCTAssertTrue(output.contains("Professor"))
    }

    func testStAbbreviation() {
        let output = TtsTextPreprocessor.preprocess("We visited St. Louis.")
        XCTAssertFalse(output.contains("St."))
        XCTAssertTrue(output.contains("Saint"))
    }

    func testVsAbbreviationWithPeriod() {
        let output = TtsTextPreprocessor.preprocess("Cats vs. dogs.")
        XCTAssertTrue(output.contains("versus"), "Expected 'versus', got: \(output)")
    }

    func testEtcAbbreviationAtEndOfSentence() {
        // End-of-string must still match even though there is no trailing space.
        let output = TtsTextPreprocessor.preprocess("apples, oranges, etc.")
        XCTAssertTrue(output.contains("etcetera"), "Expected 'etcetera', got: \(output)")
    }

    func testEgAbbreviation() {
        let output = TtsTextPreprocessor.preprocess("fruits, e.g. apples.")
        XCTAssertTrue(output.contains("for example"), "Expected 'for example', got: \(output)")
        XCTAssertFalse(output.contains("e.g."))
    }

    func testIeAbbreviation() {
        let output = TtsTextPreprocessor.preprocess("the reds, i.e. ripe tomatoes.")
        XCTAssertTrue(output.contains("that is"), "Expected 'that is', got: \(output)")
        XCTAssertFalse(output.contains("i.e."))
    }

    func testAbbreviationFollowedByCommaIsExpanded() {
        // `.` followed by `,` — also non-word→non-word; must still match.
        let output = TtsTextPreprocessor.preprocess("See Dr., then leave.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor', got: \(output)")
    }

    func testAbbreviationAtEndOfInput() {
        let output = TtsTextPreprocessor.preprocess("Ask Dr.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor', got: \(output)")
    }

    func testNoDoubleReplacementWhenEmbeddedInLongerWord() {
        // "Drs" should not match "Dr." (no period). The leading `\b` still guards this,
        // but this test guards against a regression if the trailing lookahead were
        // overly permissive.
        let output = TtsTextPreprocessor.preprocess("Drs gathered at the conference.")
        XCTAssertFalse(output.contains("Doctor"), "Bare 'Drs' should not expand, got: \(output)")
    }

    func testMultipleAbbreviationsInOneSentence() {
        let output = TtsTextPreprocessor.preprocess("Dr. Smith vs. Prof. Jones at St. Mary's.")
        XCTAssertTrue(output.contains("Doctor"))
        XCTAssertTrue(output.contains("versus"))
        XCTAssertTrue(output.contains("Professor"))
        XCTAssertTrue(output.contains("Saint"))
    }

    // MARK: - Edge cases

    /// `Dr.Smith` (no space) should NOT expand — the following `S` is a word char,
    /// so the lookahead `[^\w.]` rejects it. Guards against over-matching.
    func testAbbreviationWithoutTrailingSpaceIsNotExpanded() {
        let output = TtsTextPreprocessor.preprocess("Dr.Smith arrived late.")
        XCTAssertFalse(output.contains("Doctor"), "Dr.Smith (no space) must not expand, got: \(output)")
    }

    /// Regex uses `.caseInsensitive`, so `dr.` still matches.
    func testLowercaseAbbreviationMatches() {
        let output = TtsTextPreprocessor.preprocess("see dr. jones tomorrow.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor' from lowercase 'dr.', got: \(output)")
    }

    /// Closing punctuation (`,` `;` `:` `)` `?` `!` `"`) after the period must still match.
    func testAbbreviationFollowedBySemicolonIsExpanded() {
        let output = TtsTextPreprocessor.preprocess("Ask Dr.; then decide.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor', got: \(output)")
    }

    func testAbbreviationFollowedByCloseParenIsExpanded() {
        let output = TtsTextPreprocessor.preprocess("(Dr.) Morita is on call.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor', got: \(output)")
    }

    func testAbbreviationFollowedByQuestionMarkIsExpanded() {
        let output = TtsTextPreprocessor.preprocess("Is that Dr.? I think so.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor', got: \(output)")
    }

    func testAbbreviationFollowedByQuoteIsExpanded() {
        let output = TtsTextPreprocessor.preprocess("They said \"Dr.\" loudly.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor', got: \(output)")
    }

    /// Start-of-input position still matches via leading `\b`.
    func testAbbreviationAtStartOfInput() {
        let output = TtsTextPreprocessor.preprocess("Dr. Smith arrived.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor', got: \(output)")
    }

    /// `\s` in the lookahead includes newline/tab.
    func testAbbreviationFollowedByNewlineIsExpanded() {
        let output = TtsTextPreprocessor.preprocess("Paged Dr.\nSmith took over.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor' across newline, got: \(output)")
    }

    func testAbbreviationFollowedByTabIsExpanded() {
        let output = TtsTextPreprocessor.preprocess("Paged Dr.\tSmith took over.")
        XCTAssertTrue(output.contains("Doctor"), "Expected 'Doctor' across tab, got: \(output)")
    }

    // MARK: - Pre-existing bug: overlapping "vs" and "vs." entries

    /// `commonAbbreviations` contains both `"vs"` and `"vs."`. Swift dict iteration
    /// order is undefined. If `\bvs\b` fires first on `"vs."`, it replaces the `vs`
    /// inside the abbreviation, leaving a dangling `.` and preventing the proper
    /// `"vs."` entry from matching.
    ///
    /// This test asserts the CORRECT expansion (no stray period before " dogs").
    /// Expected to fail intermittently (or always, depending on Swift version's
    /// dict order) until `processCommonAbbreviations` iterates longer keys first.
    func testVsDotIsExpandedWithoutStrayPeriod() {
        let output = TtsTextPreprocessor.preprocess("Cats vs. dogs.")
        XCTAssertFalse(
            output.contains("versus."),
            "Stray period after 'versus' indicates 'vs' matched before 'vs.'; got: \(output)"
        )
        XCTAssertTrue(output.contains("versus dogs"), "Expected 'versus dogs', got: \(output)")
    }

    /// Same shape for `etc` vs `etc.` and `e.g.` vs `i.e.` (both of which have only
    /// the dotted form, so are unaffected — included for coverage).
    func testEtcDotIsExpandedWithoutStrayPeriod() {
        let output = TtsTextPreprocessor.preprocess("apples, oranges, etc. today.")
        XCTAssertFalse(
            output.contains("etcetera."),
            "Stray period after 'etcetera' indicates 'etc' matched before 'etc.'; got: \(output)"
        )
    }
}
