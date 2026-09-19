import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsChunkingTests: XCTestCase {

    /// One piece per word and punctuation mark, plus single letters so a
    /// per-character segmentation is always available and always longer.
    private func makeTokenizer() throws -> SentencePieceTokenizer {
        let wordPieces = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
            .map { "\u{2581}" + $0 }
        let letters = "abcdefghijklmnopqrstuvwxyz".map { String($0) }
        return try SentencePieceTestModel.tokenizer(
            pieces: ["<unk>", "\u{2581}", ".", ","] + wordPieces + letters)
    }

    func testParagraphBreakDoesNotChangeChunking() throws {
        let tokenizer = try makeTokenizer()
        let spaced = "one two three. four five six, seven eight nine."
        let broken = "one two three.\n\nfour five six, seven eight nine."

        let expected = PocketTtsSynthesizer.chunkTextWithMetadata(
            spaced, tokenizer: tokenizer, maxTokens: 8)
        let actual = PocketTtsSynthesizer.chunkTextWithMetadata(
            broken, tokenizer: tokenizer, maxTokens: 8)

        XCTAssertEqual(actual, expected)
    }

    func testSentenceAfterParagraphBreakIsNotCutAtAWordBoundary() throws {
        // The second sentence is 8 tokens and fits the cap whole. Counted with
        // its leading newlines it measured one token per character, overflowed,
        // and was cut between words.
        let tokenizer = try makeTokenizer()
        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            "one two three.\n\nfour five six, seven eight nine.",
            tokenizer: tokenizer, maxTokens: 8)

        XCTAssertEqual(
            chunks,
            [
                PocketTtsSynthesizer.TextChunk(text: "one two three.", isMidSentence: false),
                PocketTtsSynthesizer.TextChunk(
                    text: "four five six, seven eight nine.", isMidSentence: false),
            ])
    }

    func testChunksCarryNoNewlines() throws {
        let tokenizer = try makeTokenizer()
        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            "one two three.\nfour five six.\r\n\r\nseven eight nine.",
            tokenizer: tokenizer, maxTokens: 8)
        for chunk in chunks {
            XCTAssertNil(
                chunk.text.rangeOfCharacter(from: .newlines),
                "chunk still carries a newline: \(chunk.text.debugDescription)")
        }
    }

    func testCollapseWhitespaceFoldsNewlinesAndRuns() {
        XCTAssertEqual(
            PocketTtsSynthesizer.collapseWhitespace("a\n\nb \t c\r\nd"),
            "a b c d")
    }
}
