import XCTest

#if canImport(FluidAudioTTS) && canImport(ESpeakNG)
@testable import FluidAudioTTS
@testable import FluidAudio

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

        let chunks: [TextChunk]
        do {
            chunks = try KokoroChunker.chunk(
                text: text,
                wordToPhonemes: lowerLexicon,
                caseSensitiveLexicon: caseSensitiveLexicon,
                customLexicon: nil,
                targetTokens: 120,
                hasLanguageToken: false,
                allowedPhonemes: allowed,
                phoneticOverrides: []
            )
        } catch {
            XCTFail("Chunker threw unexpected error: \(error)")
            return
        }

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

        let chunks: [TextChunk]
        do {
            chunks = try KokoroChunker.chunk(
                text: text,
                wordToPhonemes: lexicon,
                caseSensitiveLexicon: [:],
                customLexicon: nil,
                targetTokens: 20,
                hasLanguageToken: false,
                allowedPhonemes: allowed,
                phoneticOverrides: []
            )
        } catch {
            XCTFail("Chunker threw unexpected error: \(error)")
            return
        }

        XCTAssertGreaterThan(chunks.count, 1, "Expected run-on text to be split under tight token budget")
        chunks.forEach { chunk in
            XCTAssertFalse(chunk.words.isEmpty)
            XCTAssertLessThanOrEqual(chunk.phonemes.count, 6)
        }
    }

    func testEmojiDoesNotDesynchronizeOverrideIndices() {
        let text = "Hello 😊 [Kokoro](/k o k o ɹ o/)"

        let preprocessing = TtsTextPreprocessor.preprocessDetailed(text)
        XCTAssertEqual(preprocessing.phoneticOverrides.count, 1, "Expected a single phonetic override")
        XCTAssertEqual(
            preprocessing.phoneticOverrides.first?.wordIndex,
            2,
            "Emoji should advance the word index so the override lands on Kokoro"
        )

        let lexicon: [String: [String]] = [
            "hello": ["h", "e", "l", "o"]
        ]
        let allowed: Set<String> = ["h", "e", "l", "o", "k", "ɹ", " "]

        let chunks: [TextChunk]
        do {
            chunks = try KokoroChunker.chunk(
                text: preprocessing.text,
                wordToPhonemes: lexicon,
                caseSensitiveLexicon: [:],
                customLexicon: nil,
                targetTokens: 32,
                hasLanguageToken: false,
                allowedPhonemes: allowed,
                phoneticOverrides: preprocessing.phoneticOverrides
            )
        } catch {
            XCTFail("Chunker threw unexpected error: \(error)")
            return
        }

        guard let chunk = chunks.first else {
            XCTFail("Expected at least one chunk")
            return
        }

        XCTAssertTrue(chunk.words.contains("Kokoro"))
        XCTAssertEqual(
            Array(chunk.phonemes.suffix(6)),
            ["k", "o", "k", "o", "ɹ", "o"],
            "Override phonemes should be applied after the emoji"
        )
    }
}
#endif
