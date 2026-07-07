import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the shared conservative English raw-text normalization pass
/// (issue #711): strict standalone numbers/ordinals/decimals/12-hour times
/// are spelled out, while ambiguous or structured forms are left untouched.
final class EnglishTextNormalizerTests: XCTestCase {

    private func normalize(_ text: String) -> String {
        EnglishTextNormalizer.normalize(text)
    }

    // MARK: - Supported standalone forms

    func testCardinalInteger() {
        XCTAssertEqual(normalize("I am 26 years old."), "I am twenty six years old.")
        XCTAssertEqual(normalize("100"), "one hundred")
    }

    func testOrdinal() {
        XCTAssertEqual(normalize("Today is June 13th."), "Today is June thirteenth.")
        XCTAssertEqual(normalize("the 21st"), "the twenty first")
    }

    func testDecimal() {
        XCTAssertEqual(normalize("The score is 3.14."), "The score is three point one four.")
        XCTAssertEqual(normalize("0.5"), "zero point five")
    }

    func testLeadingZeroDigitString() {
        XCTAssertEqual(normalize("Agent 007"), "Agent zero zero seven")
    }

    func testTwelveHourMeridiemTime() {
        XCTAssertEqual(
            normalize("The current time is 1:49 PM."),
            "The current time is one forty nine p m.")
        XCTAssertEqual(normalize("1:49 p.m."), "one forty nine p m")
        XCTAssertEqual(normalize("meet at 9:00 AM"), "meet at nine o'clock a m")
        XCTAssertEqual(normalize("3:05 pm"), "three oh five p m")
    }

    func testDecadeForms() {
        // 4-digit decades read year-style then pluralize the last word.
        XCTAssertEqual(normalize("in the 1770s it began"), "in the seventeen seventies it began")
        XCTAssertEqual(normalize("the 1990s"), "the nineteen nineties")
        XCTAssertEqual(normalize("the 2020s"), "the twenty twenties")
        // Round centuries pluralize `hundred`/`thousand`.
        XCTAssertEqual(normalize("the 1900s"), "the nineteen hundreds")
        XCTAssertEqual(normalize("the 2000s"), "the two thousands")
        // 2-digit decades, with and without a leading apostrophe.
        XCTAssertEqual(normalize("music from the '90s"), "music from the nineties")
        XCTAssertEqual(normalize("the 80s sound"), "the eighties sound")
    }

    func testDecadeTrailingPunctuationPreserved() {
        XCTAssertEqual(normalize("born in the 1980s."), "born in the nineteen eighties.")
    }

    func testNonZeroDecadeUnchanged() {
        // Only decades ending in 0 are rewritten.
        XCTAssertEqual(normalize("1995s"), "1995s")
    }

    func testAllZeroDecadeUnchanged() {
        // `'00s`/`00s` has no century to anchor it and must not read as "zeros".
        XCTAssertEqual(normalize("music from the '00s"), "music from the '00s")
        XCTAssertEqual(normalize("the 00s"), "the 00s")
    }

    func testBareYear() {
        XCTAssertEqual(normalize("it began in 1770"), "it began in seventeen seventy")
        XCTAssertEqual(normalize("the year 2026"), "the year twenty twenty six")
        XCTAssertEqual(normalize("back in 1905"), "back in nineteen oh five")
        XCTAssertEqual(normalize("since 2005"), "since two thousand five")
        XCTAssertEqual(normalize("in 2000"), "in two thousand")
    }

    func testFourDigitOutsideYearRangeUsesCardinal() {
        // Below 1000 / above 2099 fall through to the cardinal rule.
        XCTAssertEqual(normalize("2100"), "two thousand one hundred")
        XCTAssertEqual(normalize("9999"), "nine thousand nine hundred ninety nine")
    }

    func testMultipleFormsInOneSentence() {
        XCTAssertEqual(
            normalize("At 1:49 PM on the 13th I scored 3.14 in 26 tries."),
            "At one forty nine p m on the thirteenth I scored three point one four in twenty six tries.")
    }

    // MARK: - Ambiguous / structured forms left unchanged

    func testVersionStringUnchanged() {
        XCTAssertEqual(normalize("Install 1.2.3 now"), "Install 1.2.3 now")
    }

    func testGroupedNumberUnchanged() {
        XCTAssertEqual(normalize("It costs 1,234 dollars"), "It costs 1,234 dollars")
    }

    func testEmbeddedDigitsUnchanged() {
        XCTAssertEqual(normalize("word26 and 26word"), "word26 and 26word")
    }

    func testLooseColonNumberUnchanged() {
        // No meridiem → not a time we rewrite.
        XCTAssertEqual(normalize("ratio 1:49 here"), "ratio 1:49 here")
    }

    func testInvalidTimeUnchanged() {
        // Minute out of range → left as-is.
        XCTAssertEqual(normalize("1:99 PM"), "1:99 PM")
    }

    func testTwentyFourHourTimeUnchanged() {
        XCTAssertEqual(normalize("13:49"), "13:49")
        XCTAssertEqual(normalize("13:49 PM"), "13:49 PM")
    }

    func testInvalidOrdinalSuffixUnchanged() {
        // `1th`/`2th` aren't grammatical ordinals → not rewritten.
        XCTAssertEqual(normalize("1th"), "1th")
        XCTAssertEqual(normalize("2th"), "2th")
    }

    // MARK: - Boundary details

    func testTrailingSentencePunctuationPreserved() {
        XCTAssertEqual(normalize("I scored 26."), "I scored twenty six.")
        XCTAssertEqual(normalize("pi is 3.14, roughly"), "pi is three point one four, roughly")
    }

    func testDecimalNotConfusedWithVersionPrefix() {
        // `3.14` inside `3.14.2` (version-like) must stay untouched.
        XCTAssertEqual(normalize("v3.14.2"), "v3.14.2")
    }

    func testNoDigitsIsUnchanged() {
        XCTAssertEqual(normalize("Hello world"), "Hello world")
    }

    func testEmptyStringIsUnchanged() {
        XCTAssertEqual(normalize(""), "")
    }
}
