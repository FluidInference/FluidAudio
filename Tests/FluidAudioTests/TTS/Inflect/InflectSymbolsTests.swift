import XCTest

@testable import FluidAudio

/// Parity tests for the Inflect symbol table + token encoder. Golden vectors
/// are computed from the upstream `cleaned_text_to_sequence` + `intersperse`
/// against the shipped `symbols.py` (see mobius `inflect-v2/coreml`).
final class InflectSymbolsTests: XCTestCase {

    func testSymbolTableMatchesUpstreamLayout() {
        // keithito: [_pad] + punctuation + letters + IPA == 178 entries.
        XCTAssertEqual(InflectSymbols.symbols.count, 178)
        XCTAssertEqual(InflectSymbols.symbols[0], Unicode.Scalar("_"))
        XCTAssertEqual(InflectSymbols.blankID, 0)
    }

    func testKnownScalarIDs() {
        // Space is id 16 (pad + 15 punctuation chars precede it).
        XCTAssertEqual(InflectSymbols.sequence(for: " "), [16])
        // Golden: "hello" → [50, 47, 54, 54, 57].
        XCTAssertEqual(InflectSymbols.sequence(for: "hello"), [50, 47, 54, 54, 57])
    }

    func testStressedIPASequenceMatchesGolden() {
        // "həlˈoʊ" → [50, 83, 54, 156, 57, 135] (ˈ is a standalone symbol).
        XCTAssertEqual(
            InflectSymbols.sequence(for: "həlˈoʊ"),
            [50, 83, 54, 156, 57, 135])
    }

    func testIntersperseWrapsAndSeparatesWithBlanks() {
        // commons.intersperse([50,47,54,54,57], 0)
        XCTAssertEqual(
            InflectSymbols.encode("hello"),
            [0, 50, 0, 47, 0, 54, 0, 54, 0, 57, 0])
    }

    func testMultiWordGoldenVector() {
        // "ðə kwˈɪk bɹˈaʊn fˈɑks" full pipeline golden.
        let expected: [Int32] = [
            0, 81, 0, 83, 0, 16, 0, 53, 0, 65, 0, 156, 0, 102, 0, 53, 0, 16, 0, 44, 0, 123, 0,
            156, 0, 43, 0, 135, 0, 56, 0, 16, 0, 48, 0, 156, 0, 69, 0, 53, 0, 61, 0,
        ]
        XCTAssertEqual(InflectSymbols.encode("ðə kwˈɪk bɹˈaʊn fˈɑks"), expected)
    }

    func testUnknownScalarsAreDropped() {
        // A char outside the 178-symbol set is skipped (matches the Python
        // `if s in _symbol_to_id` filter). "@" is not in the table.
        XCTAssertEqual(InflectSymbols.sequence(for: "h@i"), InflectSymbols.sequence(for: "hi"))
    }

    func testSyllabicConsonantIsTwoScalars() {
        // n̩ = U+006E U+0329 — two symbols, not one grapheme.
        let ids = InflectSymbols.sequence(for: "n\u{0329}")
        XCTAssertEqual(ids.count, 2)
    }

    func testEmptyInputEncodesToLoneBlank() {
        XCTAssertEqual(InflectSymbols.encode(""), [0])
    }
}
