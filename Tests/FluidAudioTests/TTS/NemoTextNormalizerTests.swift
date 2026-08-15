import XCTest

@testable import FluidAudio

/// Exercises the bundled compiled-FST text-normalization engine
/// (`text-processing-rs`) through the `NemoTextNormalizer` wrapper.
///
/// Every expected value is byte-exact with NVIDIA NeMo's
/// `Normalizer(lang=…, deterministic=True)`. These run against the linked
/// `NemoTextProcessing` xcframework, so a green run also proves the FFI + the
/// bundled grammars are wired correctly.
final class NemoTextNormalizerTests: XCTestCase {

    private func assertNormalizes(
        _ input: String,
        _ language: NemoTextNormalizer.Language,
        to expected: String,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(
            NemoTextNormalizer.normalize(input, language: language),
            expected,
            file: file,
            line: line
        )
    }

    // MARK: - English

    func testEnglishMoney() {
        assertNormalizes("$5", .english, to: "five dollars")
        assertNormalizes("$2", .english, to: "two dollars")
    }

    func testEnglishInSentence() {
        assertNormalizes("I have 5 cats", .english, to: "I have five cats")
    }

    func testEnglishCardinal() {
        assertNormalizes("1", .english, to: "one")
    }

    // MARK: - Mandarin

    func testMandarinYear() {
        assertNormalizes("2024年", .mandarin, to: "二零二四年")
    }

    func testMandarinMoney() {
        assertNormalizes("$123", .mandarin, to: "一百二十三美元")
    }

    func testMandarinDecimal() {
        assertNormalizes("12.5", .mandarin, to: "十二点五")
    }

    // MARK: - Other languages

    func testJapaneseCardinal() {
        assertNormalizes("1", .japanese, to: "一")
    }

    func testFrenchCardinal() {
        assertNormalizes("83", .french, to: "quatre-vingt-trois")
    }

    func testSpanishCardinal() {
        assertNormalizes("2", .spanish, to: "dos")
    }

    func testGermanCardinal() {
        assertNormalizes("1", .german, to: "eins")
    }

    func testHindiWithNumber() {
        assertNormalizes("4 चौके", .hindi, to: "चार चौके")
    }

    // MARK: - Passthrough / safety

    /// Plain text with no semiotic tokens must pass through unchanged — the
    /// normalizer is safe to call on every input as a frontend pre-pass.
    func testPlainTextIsNoOp() {
        assertNormalizes("plain text", .english, to: "plain text")
        assertNormalizes("hello world", .english, to: "hello world")
    }

    func testEmptyStringIsNoOp() {
        assertNormalizes("", .english, to: "")
    }

    /// Every supported language code round-trips through the FFI without
    /// returning the raw input (i.e. the language is actually bundled).
    func testAllLanguagesAreBundled() {
        let probes: [(NemoTextNormalizer.Language, String)] = [
            (.english, "1"), (.mandarin, "2024年"), (.japanese, "1"),
            (.french, "83"), (.spanish, "2"), (.german, "1"), (.hindi, "4 चौके"),
        ]
        for (language, input) in probes {
            let out = NemoTextNormalizer.normalize(input, language: language)
            XCTAssertNotEqual(
                out, input,
                "\(language.rawValue) grammar appears unbundled (input returned unchanged)")
        }
    }
}
