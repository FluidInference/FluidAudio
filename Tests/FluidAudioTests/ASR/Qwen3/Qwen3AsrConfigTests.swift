import Foundation
import XCTest

@testable import FluidAudio

final class Qwen3AsrConfigTests: XCTestCase {

    // MARK: - Config Constants

    func testOutputFramesPerWindowComputation() {
        // ceil(100/8) = 13
        let expected = (100 + 8 - 1) / 8
        XCTAssertEqual(Qwen3AsrConfig.outputFramesPerWindow, expected)
        XCTAssertEqual(Qwen3AsrConfig.outputFramesPerWindow, 13)
    }

    func testSpecialTokenIdsAreInRange() {
        let vocabSize = Qwen3AsrConfig.vocabSize
        let tokenIds = [
            Qwen3AsrConfig.audioStartTokenId,
            Qwen3AsrConfig.audioEndTokenId,
            Qwen3AsrConfig.audioTokenId,
            Qwen3AsrConfig.asrTextTokenId,
            Qwen3AsrConfig.imStartTokenId,
            Qwen3AsrConfig.imEndTokenId,
            Qwen3AsrConfig.systemTokenId,
            Qwen3AsrConfig.userTokenId,
            Qwen3AsrConfig.assistantTokenId,
            Qwen3AsrConfig.newlineTokenId,
        ]

        for tokenId in tokenIds {
            XCTAssertGreaterThanOrEqual(tokenId, 0, "Token ID \(tokenId) should be non-negative")
            XCTAssertLessThan(tokenId, vocabSize, "Token ID \(tokenId) should be < vocabSize (\(vocabSize))")
        }

        for eosId in Qwen3AsrConfig.eosTokenIds {
            XCTAssertGreaterThanOrEqual(eosId, 0)
            XCTAssertLessThan(eosId, vocabSize)
        }
    }

    func testMropeSectionSumsToHalfHeadDim() {
        let sum = Qwen3AsrConfig.mropeSection.reduce(0, +)
        XCTAssertEqual(sum * 2, Qwen3AsrConfig.headDim, "mropeSection sums * 2 should equal headDim")
    }

    func testKVHeadsDivideAttentionHeads() {
        XCTAssertEqual(
            Qwen3AsrConfig.numAttentionHeads % Qwen3AsrConfig.numKVHeads, 0,
            "numAttentionHeads should be divisible by numKVHeads"
        )
    }

    // MARK: - Language

    func testLanguageFromIsoCode() {
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "en"), .english)
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "fr"), .french)
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "zh"), .chinese)
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "ja"), .japanese)
    }

    func testLanguageFromIsoCodeCaseInsensitive() {
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "EN"), .english)
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "Fr"), .french)
    }

    func testLanguageFromEnglishName() {
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "French"), .french)
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "English"), .english)
        XCTAssertEqual(Qwen3AsrConfig.Language(from: "japanese"), .japanese)
    }

    func testLanguageFromInvalidStringReturnsNil() {
        XCTAssertNil(Qwen3AsrConfig.Language(from: "klingon"))
        XCTAssertNil(Qwen3AsrConfig.Language(from: ""))
        XCTAssertNil(Qwen3AsrConfig.Language(from: "xx"))
    }

    func testAllLanguagesHaveEnglishNames() {
        for language in Qwen3AsrConfig.Language.allCases {
            XCTAssertFalse(language.englishName.isEmpty, "\(language) should have a non-empty English name")
        }
    }

    func testLanguageRawValueIsIsoCode() {
        XCTAssertEqual(Qwen3AsrConfig.Language.english.rawValue, "en")
        XCTAssertEqual(Qwen3AsrConfig.Language.french.rawValue, "fr")
        XCTAssertEqual(Qwen3AsrConfig.Language.cantonese.rawValue, "yue")
    }

    func testLanguageRoundTrip() {
        for language in Qwen3AsrConfig.Language.allCases {
            let fromIso = Qwen3AsrConfig.Language(from: language.rawValue)
            XCTAssertEqual(fromIso, language, "Round-trip via ISO code should work for \(language)")

            let fromName = Qwen3AsrConfig.Language(from: language.englishName)
            XCTAssertEqual(fromName, language, "Round-trip via English name should work for \(language)")
        }
    }
}
