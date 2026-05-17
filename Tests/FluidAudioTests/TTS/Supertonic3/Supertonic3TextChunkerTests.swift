import XCTest

@testable import FluidAudio

final class Supertonic3TextChunkerTests: XCTestCase {

    // MARK: - Trivial inputs

    func testEmptyInputReturnsNoChunks() {
        XCTAssertEqual(Supertonic3TextChunker.chunk(text: "", maxLen: 110), [])
        XCTAssertEqual(Supertonic3TextChunker.chunk(text: "   \n   ", maxLen: 110), [])
    }

    func testShortInputReturnsSingleChunkUnchanged() {
        let chunks = Supertonic3TextChunker.chunk(
            text: "Hello there.", maxLen: 110)
        XCTAssertEqual(chunks, ["Hello there."])
    }

    func testInputAtMaxLenBoundaryFitsInOneChunk() {
        let text = String(repeating: "a", count: 110)
        let chunks = Supertonic3TextChunker.chunk(text: text, maxLen: 110)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks.first?.count, 110)
    }

    // MARK: - Sentence packing

    func testSentencesAreCombinedUpToMaxLen() {
        let text = "One. Two. Three. Four."
        let chunks = Supertonic3TextChunker.chunk(text: text, maxLen: 110)
        XCTAssertEqual(chunks.count, 1)
    }

    func testLongSentenceTriggersBoundarySplit() {
        // Two sentences of ~60 chars each — together exceed maxLen=80, so the
        // packer should emit two chunks, one per sentence.
        let sentenceA = String(repeating: "a", count: 60) + "."
        let sentenceB = String(repeating: "b", count: 60) + "."
        let chunks = Supertonic3TextChunker.chunk(
            text: "\(sentenceA) \(sentenceB)", maxLen: 80)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 80 })
    }

    // MARK: - Abbreviation awareness

    func testAbbreviationDoesNotSplitMidSentence() {
        // "Dr." should not be treated as a sentence terminator. The packer
        // should keep "Dr. Smith arrived early." as one sentence.
        let chunks = Supertonic3TextChunker.chunk(
            text: "Dr. Smith arrived early. Then he left.", maxLen: 110)
        XCTAssertEqual(chunks.count, 1)
        XCTAssertTrue(chunks[0].contains("Dr. Smith"))
        XCTAssertTrue(chunks[0].contains("Then he left."))
    }

    // MARK: - Comma fallback

    func testLongSentenceFallsBackToCommaBoundaries() {
        let parts = (0..<6).map { _ in String(repeating: "x", count: 18) }
        let sentence = parts.joined(separator: ", ") + "."
        let chunks = Supertonic3TextChunker.chunk(text: sentence, maxLen: 50)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 50 })
    }

    // MARK: - Word fallback

    func testVeryLongCommaFreeRunFallsBackToWordBoundaries() {
        let words = Array(repeating: "word", count: 40)
        let sentence = words.joined(separator: " ") + "."
        let chunks = Supertonic3TextChunker.chunk(text: sentence, maxLen: 30)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 30 })
    }

    // MARK: - Paragraph split (blank-line boundary)

    func testParagraphsAreSplitOnBlankLines() {
        let text = "First paragraph.\n\nSecond paragraph."
        let chunks = Supertonic3TextChunker.chunk(text: text, maxLen: 110)
        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0], "First paragraph.")
        XCTAssertEqual(chunks[1], "Second paragraph.")
    }
}
