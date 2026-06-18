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
