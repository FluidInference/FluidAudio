import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the shared phoneme-string chunker (issue #712): long input is
/// split at whitespace / pause punctuation into chunks under the cap, with
/// words kept intact and punctuation attached to the preceding chunk.
final class PhonemeChunkerTests: XCTestCase {

    private func chunk(_ text: String, _ maxLength: Int) -> [String] {
        PhonemeChunker.chunk(text, maxLength: maxLength)
    }

    // MARK: - Fast path

    func testWithinCapReturnsSingleTrimmedChunk() {
        XCTAssertEqual(chunk("hello world", 100), ["hello world"])
        XCTAssertEqual(chunk("  hello world  ", 100), ["hello world"])
    }

    func testBlankInputReturnsEmpty() {
        XCTAssertEqual(chunk("", 100), [])
        XCTAssertEqual(chunk("   ", 100), [])
    }

    func testExactlyAtCapIsNotSplit() {
        let text = String(repeating: "a", count: 10)
        XCTAssertEqual(chunk(text, 10), [text])
    }

    // MARK: - Splitting

    func testSplitsAtWhitespaceWithoutBreakingWords() {
        // 4 words of 5 chars + spaces = "aaaaa bbbbb ccccc ddddd" (23 chars).
        let text = "aaaaa bbbbb ccccc ddddd"
        let chunks = chunk(text, 12)
        for piece in chunks {
            XCTAssertLessThanOrEqual(piece.count, 12)
        }
        // No word is split across chunks.
        XCTAssertEqual(chunks.joined(separator: " "), text)
    }

    func testEveryChunkWithinCap() {
        let words = (0..<60).map { "w\($0)" }.joined(separator: " ")
        let chunks = chunk(words, 20)
        XCTAssertGreaterThan(chunks.count, 1)
        for piece in chunks {
            XCTAssertLessThanOrEqual(piece.count, 20)
            XCTAssertFalse(piece.hasPrefix(" "))
            XCTAssertFalse(piece.hasSuffix(" "))
        }
    }

    func testPrefersLatestBoundaryToFillChunks() {
        // "one two three four" — cap 9 should pack "one two" (7), not stop at "one".
        let chunks = chunk("one two three four", 9)
        XCTAssertEqual(chunks.first, "one two")
    }

    func testPunctuationStaysWithPrecedingChunk() {
        // Break should land after the comma's following space, keeping the
        // comma attached to the first clause.
        let chunks = chunk("hello there, friend over yonder", 14)
        XCTAssertEqual(chunks.first, "hello there,")
    }

    func testHardSplitsWordLongerThanCap() {
        // A single 25-char run with no boundary is split at the cap.
        let text = String(repeating: "x", count: 25)
        let chunks = chunk(text, 10)
        XCTAssertEqual(chunks, ["xxxxxxxxxx", "xxxxxxxxxx", "xxxxx"])
    }

    func testReassemblyPreservesAllNonWhitespaceContent() {
        let text = "the quick brown fox jumps over the lazy dog repeatedly today"
        let chunks = chunk(text, 13)
        let original = text.split(separator: " ").joined()
        let roundTrip = chunks.joined().replacingOccurrences(of: " ", with: "")
        XCTAssertEqual(roundTrip, original)
    }
}
