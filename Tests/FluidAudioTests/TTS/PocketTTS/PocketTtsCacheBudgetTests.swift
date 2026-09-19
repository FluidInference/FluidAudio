import Foundation
import XCTest

@testable import FluidAudio

final class PocketTtsCacheBudgetTests: XCTestCase {

    private static let issue933Text =
        "The humidity feels comfortable despite the approaching storm, and the air carries "
        + "that distinctive electric charge that precedes rain — a mix of ozone and wet soil "
        + "that makes everything feel fresh and alive before the first drops hit the pavement."

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

    func testIssue933SentenceRemainsWholeWithRealEnglishTokenizer() throws {
        let tokenizer = try makeEnglishTokenizer()
        let text = Self.issue933Text
        let tokenCount = tokenizer.encode(text).count

        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            text,
            tokenizer: tokenizer,
            maxTokens: PocketTtsConstants.maxTokensPerChunk,
            preferredMaxTokens: PocketTtsConstants.preferredTokensPerChunk,
            voiceCachePosition: 126)

        XCTAssertEqual(tokenCount, 69)
        XCTAssertEqual(chunks, [.init(text: text, isMidSentence: false)])
        let requiredPositions =
            126 + tokenCount + PocketTtsSynthesizer.estimateRequiredCacheFrames(text: text)
        XCTAssertLessThanOrEqual(requiredPositions, PocketTtsConstants.kvCacheMaxLen)
    }

    func testIssue933SentenceSplitsWhenVoiceLeavesTooLittleCache() throws {
        let tokenizer = try makeEnglishTokenizer()
        let text = Self.issue933Text

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

    func testGenerationWithoutEOSThrowsAtCacheBoundary() {
        XCTAssertThrowsError(
            try PocketTtsSynthesizer.validateGenerationCompleted(
                generatedFrameLimit: 286,
                cachePosition: 226,
                eosStep: nil,
                framesAfterEos: 3))
    }

    func testGenerationThrowsWhenTrailingFramesExceedCacheBoundary() {
        XCTAssertThrowsError(
            try PocketTtsSynthesizer.validateGenerationCompleted(
                generatedFrameLimit: 286,
                cachePosition: 226,
                eosStep: 285,
                framesAfterEos: 2))
    }

    func testGenerationAcceptsEOSCompletedAtCacheBoundary() {
        XCTAssertNoThrow(
            try PocketTtsSynthesizer.validateGenerationCompleted(
                generatedFrameLimit: 286,
                cachePosition: 226,
                eosStep: 284,
                framesAfterEos: 2))
    }

    func testSeparateSentencesAreNotGroupedPastPreferredTarget() throws {
        let tokenizer = try makeEnglishTokenizer()
        let first =
            "The weather remains calm across the valley while scattered clouds drift slowly "
            + "toward the eastern hills."
        let second =
            "A gentle breeze carries the scent of rain and damp earth through the open window "
            + "this afternoon."

        XCTAssertLessThanOrEqual(
            tokenizer.encode(first).count, PocketTtsConstants.preferredTokensPerChunk)
        XCTAssertLessThanOrEqual(
            tokenizer.encode(second).count, PocketTtsConstants.preferredTokensPerChunk)
        XCTAssertGreaterThan(
            tokenizer.encode(first + " " + second).count,
            PocketTtsConstants.preferredTokensPerChunk)

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
        let tokenizer = try makeEnglishTokenizer()
        let text = Self.issue933Text

        let chunks = PocketTtsSynthesizer.chunkTextWithMetadata(
            text, tokenizer: tokenizer, maxTokens: 50)

        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(tokenizer.encode(chunk.text).count, 50)
        }
    }

    private func makeEnglishTokenizer() throws -> SentencePieceTokenizer {
        let modelURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "pocket_tts_english_tokenizer",
                withExtension: "model"))
        return try SentencePieceTokenizer(modelData: Data(contentsOf: modelURL))
    }
}
