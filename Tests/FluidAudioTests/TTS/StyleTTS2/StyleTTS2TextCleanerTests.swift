import XCTest

@testable import FluidAudio

final class StyleTTS2TextCleanerTests: XCTestCase {

    // MARK: - Symbol table layout

    func testPadSymbolIsZero() {
        // The leading symbol is the pad token ($) and must map to ID 0.
        XCTAssertEqual(StyleTTS2TextCleaner.symbols.first, "$")
        XCTAssertEqual(StyleTTS2TextCleaner.dictionary["$"], 0)
    }

    func testVocabularySizeMatchesSpec() {
        // 1 (pad) + punctuation + Latin letters + IPA letters.
        // Compute from the canonical sources so a future symbol table edit
        // doesn't silently drift the reported vocab size out of sync.
        let expected =
            1
            + StyleTTS2TextCleaner.punctuation.count
            + StyleTTS2TextCleaner.letters.count
            + StyleTTS2TextCleaner.ipaLetters.count
        XCTAssertEqual(StyleTTS2TextCleaner.vocabularySize, expected)
    }

    func testPunctuationContainsSpace() {
        // The space character is part of the punctuation table — it MUST
        // round-trip to a non-nil ID so word-separated phoneme strings
        // encode without dropped tokens.
        XCTAssertNotNil(StyleTTS2TextCleaner.dictionary[" "])
    }

    // MARK: - encode()

    func testEncodeEmptyEmitsLeadingPad() {
        let ids = StyleTTS2TextCleaner.encode("")
        XCTAssertEqual(ids, [0])
    }

    func testEncodeEmptyNoPad() {
        let ids = StyleTTS2TextCleaner.encode("", prependPad: false)
        XCTAssertEqual(ids, [])
    }

    func testEncodeRoundTripsKnownLetters() {
        // Each character maps to its index in the symbol table — encoding a
        // string should reproduce that index sequence (after the pad).
        let ids = StyleTTS2TextCleaner.encode("abc", prependPad: false)
        XCTAssertEqual(ids.count, 3)
        XCTAssertEqual(ids[0], StyleTTS2TextCleaner.dictionary["a"])
        XCTAssertEqual(ids[1], StyleTTS2TextCleaner.dictionary["b"])
        XCTAssertEqual(ids[2], StyleTTS2TextCleaner.dictionary["c"])
    }

    func testEncodeDropsUnknownCharacters() {
        // The Cyrillic 'ж' is not in the table; it must be silently dropped
        // (matches the upstream `print(text); continue` branch).
        let ids = StyleTTS2TextCleaner.encode("aжb", prependPad: false)
        XCTAssertEqual(ids.count, 2)
        XCTAssertEqual(ids[0], StyleTTS2TextCleaner.dictionary["a"])
        XCTAssertEqual(ids[1], StyleTTS2TextCleaner.dictionary["b"])
    }

    func testEncodePrependsPadByDefault() {
        let ids = StyleTTS2TextCleaner.encode("a")
        XCTAssertEqual(ids.first, 0)
        XCTAssertEqual(ids.count, 2)
    }
}
