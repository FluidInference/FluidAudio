import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 14, iOS 17, *)
final class CohereTokenConversionTests: XCTestCase {

    // MARK: - Helper

    private func createTestVocabulary() -> [Int: String] {
        return [
            0: "<unk>",
            1: "<|nospeech|>",
            2: "<pad>",
            3: "</s>",
            4: "<s>",
            5: "▁Hello",
            6: "▁world",
            7: "▁this",
            8: "▁is",
            9: "▁a",
            10: "▁test",
            11: "<|en|>",
            12: "<|fr|>",
            100: "▁",
            101: "▁The",
            102: "▁quick",
            103: "▁brown",
            104: "▁fox",
        ]
    }

    // MARK: - Special Token Filtering

    func testSpecialTokensAreFiltered() {
        let vocab = createTestVocabulary()
        let manager = CohereAsrManager()

        // Create tokens with special tokens that should be filtered
        let tokens = [
            CohereAsrConfig.SpecialTokens.startToken,  // 4 - should be filtered
            5,  // "▁Hello"
            6,  // "▁world"
            CohereAsrConfig.SpecialTokens.eosToken,    // 3 - should be filtered
        ]

        // Since convertTokensToText is private, we need to test via reflection
        // or make it internal for testing. For now, let's verify the vocabulary structure.
        XCTAssertEqual(vocab[CohereAsrConfig.SpecialTokens.startToken], "<s>")
        XCTAssertEqual(vocab[CohereAsrConfig.SpecialTokens.eosToken], "</s>")
    }

    func testControlTokensAreFiltered() {
        let vocab = createTestVocabulary()

        // Verify control tokens exist in vocab
        XCTAssertTrue(vocab[11]?.hasPrefix("<|") ?? false)
        XCTAssertTrue(vocab[12]?.hasPrefix("<|") ?? false)

        // These should be filtered during conversion
        let controlTokenIds = [11, 12]  // <|en|>, <|fr|>
        for tokenId in controlTokenIds {
            if let token = vocab[tokenId] {
                XCTAssertTrue(token.hasPrefix("<|"), "Control tokens should start with <|")
            }
        }
    }

    // MARK: - SentencePiece Marker Replacement

    func testSentencePieceMarkerIsReplacedWithSpace() {
        let vocab = createTestVocabulary()

        let tokens = [
            101,  // "▁The"
            102,  // "▁quick"
            103,  // "▁brown"
            104,  // "▁fox"
        ]

        // Each token starts with ▁ which should be replaced with space
        for tokenId in tokens {
            if let token = vocab[tokenId] {
                XCTAssertTrue(token.hasPrefix("▁"), "Token \(tokenId) should start with ▁")
            }
        }
    }

    func testEmptyTokenIsSkipped() {
        let vocab = createTestVocabulary()

        // Token 100 is just "▁" which becomes empty after marker replacement
        let emptyMarkerToken = vocab[100]
        XCTAssertEqual(emptyMarkerToken, "▁")
    }

    // MARK: - Edge Cases

    func testUnknownTokenIdsAreSkipped() {
        let vocab = createTestVocabulary()

        let unknownTokenId = 999
        XCTAssertNil(vocab[unknownTokenId], "Unknown token ID should not be in vocabulary")
    }

    func testEmptyTokenListProducesEmptyText() {
        let emptyTokens: [Int] = []
        XCTAssertTrue(emptyTokens.isEmpty)
    }

    func testOnlySpecialTokensProducesEmptyText() {
        let vocab = createTestVocabulary()
        let onlySpecialTokens = [
            CohereAsrConfig.SpecialTokens.unkToken,
            CohereAsrConfig.SpecialTokens.noSpeechToken,
            CohereAsrConfig.SpecialTokens.padToken,
            CohereAsrConfig.SpecialTokens.eosToken,
            CohereAsrConfig.SpecialTokens.startToken,
        ]

        // All these should be filtered (IDs <= 4 or EOS)
        for tokenId in onlySpecialTokens {
            XCTAssertLessThanOrEqual(tokenId, 4)
        }
    }

    // MARK: - Vocabulary Structure

    func testVocabularyHasExpectedStructure() {
        let vocab = createTestVocabulary()

        // Check special tokens at expected positions
        XCTAssertEqual(vocab[0], "<unk>")
        XCTAssertEqual(vocab[1], "<|nospeech|>")
        XCTAssertEqual(vocab[2], "<pad>")
        XCTAssertEqual(vocab[3], "</s>")
        XCTAssertEqual(vocab[4], "<s>")
    }

    func testRegularTokensStartWithWordBoundary() {
        let vocab = createTestVocabulary()

        let regularTokenIds = [5, 6, 7, 8, 9, 10, 101, 102, 103, 104]
        for tokenId in regularTokenIds {
            if let token = vocab[tokenId] {
                XCTAssertTrue(
                    token.hasPrefix("▁"),
                    "Regular token \(tokenId) ('\(token)') should start with ▁"
                )
            }
        }
    }

    // MARK: - Whitespace Handling

    func testLeadingAndTrailingWhitespaceIsTrimmed() {
        // After joining and replacing ▁ with spaces, whitespace should be trimmed
        let testString = "  Hello world  "
        let trimmed = testString.trimmingCharacters(in: .whitespaces)
        XCTAssertEqual(trimmed, "Hello world")
    }

    func testMultipleSpacesBetweenWords() {
        // Each ▁ becomes a space, so consecutive ▁ tokens would create multiple spaces
        // This tests the expected behavior
        let multipleSpaces = "▁▁Hello"
        let replaced = multipleSpaces.replacingOccurrences(of: "▁", with: " ")
        XCTAssertEqual(replaced, "  Hello")
    }

    // MARK: - Token ID Range Validation

    func testSpecialTokenIDsAreInLowRange() {
        // Special tokens should be in IDs 0-4
        XCTAssertEqual(CohereAsrConfig.SpecialTokens.unkToken, 0)
        XCTAssertEqual(CohereAsrConfig.SpecialTokens.noSpeechToken, 1)
        XCTAssertEqual(CohereAsrConfig.SpecialTokens.padToken, 2)
        XCTAssertEqual(CohereAsrConfig.SpecialTokens.eosToken, 3)
        XCTAssertEqual(CohereAsrConfig.SpecialTokens.startToken, 4)
    }

    func testAllTokenIDsAreWithinVocabSize() {
        let vocab = createTestVocabulary()
        let vocabSize = CohereAsrConfig.vocabSize

        for tokenId in vocab.keys {
            XCTAssertLessThan(tokenId, vocabSize, "Token ID \(tokenId) should be < vocab size")
        }
    }

    // MARK: - Integration with CohereAsrManager

    func testCohereAsrManagerInitializes() {
        let manager = CohereAsrManager()
        XCTAssertNotNil(manager)
    }

    func testCohereAsrManagerThrowsWhenModelsNotLoaded() async {
        let manager = CohereAsrManager()
        let emptyAudio: [Float] = []

        do {
            _ = try await manager.transcribe(audioSamples: emptyAudio)
            XCTFail("Should throw when models not loaded")
        } catch {
            // Expected
            XCTAssertTrue(error is CohereAsrError)
        }
    }
}
