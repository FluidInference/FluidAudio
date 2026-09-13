import XCTest

@testable import FluidAudio

final class MossTtsNanoTextChunkerTests: XCTestCase {

    /// Whitespace-delimited word count stands in for the tokenizer.
    private func wordCount(_ s: String) -> Int {
        s.split(whereSeparator: { $0.isWhitespace }).count
    }

    func testPrepareCapitalizesAndTerminates() throws {
        XCTAssertEqual(try MossTtsNanoTextChunker.prepare("hello there"), "Hello there.")
        XCTAssertEqual(try MossTtsNanoTextChunker.prepare("Done already!"), "Done already!")
        XCTAssertEqual(try MossTtsNanoTextChunker.prepare("  line one\nline two  "), "Line one line two.")
    }

    func testPrepareCJKGetsIdeographicFullStop() throws {
        XCTAssertEqual(try MossTtsNanoTextChunker.prepare("欢迎关注"), "欢迎关注。")
        XCTAssertEqual(try MossTtsNanoTextChunker.prepare("欢迎关注！"), "欢迎关注！")
    }

    func testPrepareRejectsEmpty() {
        XCTAssertThrowsError(try MossTtsNanoTextChunker.prepare("   \n"))
    }

    func testSplitKeepsClosingPunctuationWithSentence() {
        let parts = MossTtsNanoTextChunker.split(
            "He said \"go.\" Then left! Really?", at: MossTtsNanoTextChunker.sentenceEnd)
        XCTAssertEqual(parts, ["He said \"go.\"", "Then left!", "Really?"])
    }

    func testChunkPacksSentencesUpToBudget() throws {
        let text = "One two three. Four five six. Seven eight nine ten eleven. Twelve."
        let chunks = try MossTtsNanoTextChunker.chunk(text, maxTokens: 6, count: wordCount)
        XCTAssertEqual(chunks, ["One two three. Four five six.", "Seven eight nine ten eleven.", "Twelve."])
    }

    func testChunkFallsBackToClausesAndBudgetCuts() throws {
        let text = "alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda mu nu xi omicron pi."
        let chunks = try MossTtsNanoTextChunker.chunk(text, maxTokens: 4, count: wordCount)
        XCTAssertFalse(chunks.isEmpty)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(wordCount(chunk), 4, "chunk over budget: \(chunk)")
        }
        XCTAssertEqual(
            chunks.joined(separator: " ").replacingOccurrences(of: ",", with: ""),
            "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi.")
    }

    func testChunkDisabledReturnsPreparedText() throws {
        XCTAssertEqual(try MossTtsNanoTextChunker.chunk("a. b. c", maxTokens: 0, count: wordCount), ["A. b. c."])
    }

    func testJoinRespectsCJKSpacing() {
        XCTAssertEqual(MossTtsNanoTextChunker.join("Hello.", "World."), "Hello. World.")
        XCTAssertEqual(MossTtsNanoTextChunker.join("你好。", "世界。"), "你好。世界。")
    }

    func testPauseLengthDependsOnChunkWordCount() {
        XCTAssertEqual(
            MossTtsNanoTextChunker.pauseSeconds(after: "Short one here."), MossTtsNanoConstants.interChunkPauseShort)
        XCTAssertEqual(
            MossTtsNanoTextChunker.pauseSeconds(after: "This chunk has quite a few words in it."),
            MossTtsNanoConstants.interChunkPauseLong)
    }
}
