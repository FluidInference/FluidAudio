import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsCacheBudgetTests: XCTestCase {

    func testStandardVoiceLimitUsesRemainingCacheCapacity() throws {
        let limit = try PocketTtsSynthesizer.effectiveMaxTokensPerChunk(
            requested: PocketTtsConstants.maxTokensPerChunk,
            voiceCachePosition: 126)
        XCTAssertEqual(limit, 385)
    }

    func testLongClonedVoiceReducesRawTextCeiling() throws {
        let limit = try PocketTtsSynthesizer.effectiveMaxTokensPerChunk(
            requested: PocketTtsConstants.maxTokensPerChunk,
            voiceCachePosition: 251)
        XCTAssertEqual(limit, 260)
    }

    func testInvalidTextCeilingIsRejected() {
        XCTAssertThrowsError(
            try PocketTtsSynthesizer.effectiveMaxTokensPerChunk(
                requested: 0, voiceCachePosition: 126))
    }

    func testCacheCapacityAcceptsLastSlotAndRejectsOverflow() throws {
        XCTAssertNoThrow(
            try PocketTtsSynthesizer.validateKVCacheCapacity(
                currentPosition: 511, additionalPositions: 1))
        XCTAssertThrowsError(
            try PocketTtsSynthesizer.validateKVCacheCapacity(
                currentPosition: 512, additionalPositions: 1))
    }

    func testSentenceAboveFiftyTokensRemainsWhole() throws {
        // Match issue #933's relevant shape: an ordinary-length sentence
        // that tokenizes to 69 pieces, above the old 50-token guideline.
        let tokenizer = try makeRepeatedWordTokenizer()
        let text = Array(repeating: "aa", count: 34).joined(separator: " ") + "."

        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            text,
            tokenizer: tokenizer,
            maxTokens: PocketTtsConstants.maxTokensPerChunk,
            preferredMaxTokens: PocketTtsConstants.preferredTokensPerChunk,
            voiceCachePosition: 126)

        XCTAssertEqual(tokenizer.encode(text).count, 69)
        XCTAssertEqual(chunks, [.init(text: text, isMidSentence: false)])
    }

    func testIssue933SentenceFitsStandardVoiceBudget() {
        let text =
            "The humidity feels comfortable despite the approaching storm, and the air carries "
            + "that distinctive electric charge that precedes rain — a mix of ozone and wet soil "
            + "that makes everything feel fresh and alive before the first drops hit the pavement."

        // The issue reports 69 SentencePiece tokens and a 125-frame voice
        // prompt (126 cache positions with BOS). The sentence should not be
        // split merely because its token count exceeds the preferred 50.
        let requiredPositions =
            126 + 69 + PocketTtsSynthesizer.estimateRequiredCacheFrames(text: text)
        XCTAssertLessThanOrEqual(requiredPositions, PocketTtsConstants.kvCacheMaxLen)
    }

    func testSameTokenCountSplitsWhenVoiceLeavesTooLittleCache() throws {
        let tokenizer = try makeRepeatedWordTokenizer()
        let text = Array(repeating: "aa", count: 34).joined(separator: " ") + "."

        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            text,
            tokenizer: tokenizer,
            maxTokens: PocketTtsConstants.maxTokensPerChunk,
            preferredMaxTokens: PocketTtsConstants.preferredTokensPerChunk,
            voiceCachePosition: 300)

        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertTrue(
                PocketTtsSynthesizer.fitsTextChunk(
                    chunk.text,
                    tokenizer: tokenizer,
                    maxTokens: PocketTtsConstants.maxTokensPerChunk,
                    voiceCachePosition: 300,
                    isMidSentence: chunk.isMidSentence))
        }
    }

    func testGenerationFrameCountStopsAtCacheBoundary() {
        XCTAssertEqual(
            PocketTtsSynthesizer.boundedGenerationFrameCount(
                text: Array(repeating: "word", count: 100).joined(separator: " "),
                cachePosition: 226),
            286)
        XCTAssertEqual(
            PocketTtsSynthesizer.boundedGenerationFrameCount(
                text: "word", cachePosition: PocketTtsConstants.kvCacheMaxLen),
            0)
    }

    func testSeparateSentencesAreNotGroupedPastPreferredTarget() throws {
        let (tokenizer, words) = try makeTokenizer(wordCount: 60)
        let first = words.prefix(30).joined(separator: " ") + "."
        let second = words.suffix(30).joined(separator: " ") + "."

        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            first + " " + second,
            tokenizer: tokenizer,
            maxTokens: PocketTtsConstants.maxTokensPerChunk,
            preferredMaxTokens: PocketTtsConstants.preferredTokensPerChunk)

        XCTAssertEqual(
            chunks,
            [
                .init(text: first, isMidSentence: false),
                .init(text: second, isMidSentence: false),
            ])
    }

    func testExplicitFiftyTokenCeilingStillSplitsLongSentence() throws {
        let (tokenizer, words) = try makeTokenizer(wordCount: 68)
        let text = words.joined(separator: " ") + "."

        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            text, tokenizer: tokenizer, maxTokens: 50)

        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(tokenizer.encode(chunk.text).count, 50)
        }
    }

    /// Build a tiny SentencePiece model with one token per generated word.
    /// This keeps the regression tests independent of downloaded model files.
    private func makeTokenizer(
        wordCount: Int
    ) throws -> (SentencePieceTokenizer, [String]) {
        let words = (0..<wordCount).map { "term\($0)" }
        let pieces = ["<unk>", "\u{2581}", "."] + words.map { "\u{2581}" + $0 }
        return (try SentencePieceTokenizer(modelData: modelData(pieces: pieces)), words)
    }

    /// Each `aa` word becomes two pieces.
    private func makeRepeatedWordTokenizer() throws -> SentencePieceTokenizer {
        try SentencePieceTokenizer(
            modelData: modelData(
                pieces: [
                    "<unk>", "\u{2581}", "\u{2581}a", "a", ".", "\u{2581}A",
                ]))
    }

    private func modelData(pieces: [String]) -> Data {
        var bytes: [UInt8] = []
        for piece in pieces {
            var body: [UInt8] = []
            let pieceBytes = Array(piece.utf8)
            body.append(contentsOf: tag(fieldNumber: 1, wireType: 2))
            body.append(contentsOf: varint(UInt64(pieceBytes.count)))
            body.append(contentsOf: pieceBytes)
            body.append(contentsOf: tag(fieldNumber: 2, wireType: 5))
            var score: Float = -1
            body.append(contentsOf: withUnsafeBytes(of: &score) { Array($0) })

            bytes.append(contentsOf: tag(fieldNumber: 1, wireType: 2))
            bytes.append(contentsOf: varint(UInt64(body.count)))
            bytes.append(contentsOf: body)
        }
        return Data(bytes)
    }

    private func tag(fieldNumber: Int, wireType: Int) -> [UInt8] {
        varint(UInt64((fieldNumber << 3) | wireType))
    }

    private func varint(_ value: UInt64) -> [UInt8] {
        var result: [UInt8] = []
        var remaining = value
        while remaining > 0x7F {
            result.append(UInt8(remaining & 0x7F) | 0x80)
            remaining >>= 7
        }
        result.append(UInt8(remaining))
        return result
    }
}
