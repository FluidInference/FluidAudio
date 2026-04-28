import Foundation
import XCTest

@testable import FluidAudio

final class CohereAsrConfigTests: XCTestCase {

    // MARK: - Config Constants

    func testSampleRateIs16kHz() {
        XCTAssertEqual(CohereAsrConfig.sampleRate, 16000)
    }

    func testMaxAudioDurationIs35Seconds() {
        // Matches the encoder mel input [1, 128, 3500] (3500 * 160 / 16000 = 35s).
        XCTAssertEqual(CohereAsrConfig.maxAudioSeconds, 35.0)
    }

    func testMaxSamplesMatchesDurationAndSampleRate() {
        let expectedSamples = Int(CohereAsrConfig.maxAudioSeconds * Float(CohereAsrConfig.sampleRate))
        XCTAssertEqual(CohereAsrConfig.maxSamples, expectedSamples)
        XCTAssertEqual(CohereAsrConfig.maxSamples, 560_000)
    }

    func testVocabSizeIs16384() {
        XCTAssertEqual(CohereAsrConfig.vocabSize, 16_384)
    }

    func testMaxSeqLenIs108() {
        // KV cache capacity
        XCTAssertEqual(CohereAsrConfig.maxSeqLen, 108)
    }

    func testHeadDimMatchesDecoderDimension() {
        let expectedHeadDim = CohereAsrConfig.decoderHiddenSize / CohereAsrConfig.numDecoderHeads
        XCTAssertEqual(CohereAsrConfig.headDim, expectedHeadDim)
        XCTAssertEqual(CohereAsrConfig.headDim, 128)
    }

    // MARK: - Special Tokens

    func testSpecialTokenIdsAreInRange() {
        let vocabSize = CohereAsrConfig.vocabSize
        let tokenIds = [
            CohereAsrConfig.SpecialTokens.unkToken,
            CohereAsrConfig.SpecialTokens.noSpeechToken,
            CohereAsrConfig.SpecialTokens.padToken,
            CohereAsrConfig.SpecialTokens.eosToken,
            CohereAsrConfig.SpecialTokens.startToken,
        ]

        for tokenId in tokenIds {
            XCTAssertGreaterThanOrEqual(tokenId, 0, "Token ID \(tokenId) should be non-negative")
            XCTAssertLessThan(tokenId, vocabSize, "Token ID \(tokenId) should be < vocabSize (\(vocabSize))")
        }
    }

    func testSpecialTokensAreUnique() {
        let tokens = Set([
            CohereAsrConfig.SpecialTokens.unkToken,
            CohereAsrConfig.SpecialTokens.noSpeechToken,
            CohereAsrConfig.SpecialTokens.padToken,
            CohereAsrConfig.SpecialTokens.eosToken,
            CohereAsrConfig.SpecialTokens.startToken,
        ])
        XCTAssertEqual(tokens.count, 5, "Special tokens should be unique")
    }

    func testEosTokenId() {
        XCTAssertEqual(CohereAsrConfig.SpecialTokens.eosToken, 3)
    }

    func testStartTokenId() {
        XCTAssertEqual(CohereAsrConfig.SpecialTokens.startToken, 4)
    }

    // MARK: - Mel Spectrogram Parameters

    func testMelSpecParametersAreValid() {
        XCTAssertEqual(CohereAsrConfig.MelSpec.nFFT, 512)
        XCTAssertEqual(CohereAsrConfig.MelSpec.hopLength, 160)
        XCTAssertEqual(CohereAsrConfig.MelSpec.nMels, 128)
        XCTAssertEqual(CohereAsrConfig.numMelBins, 128)
    }

    func testMelSpecFrequencyRange() {
        XCTAssertEqual(CohereAsrConfig.MelSpec.fMin, 0.0)
        XCTAssertEqual(CohereAsrConfig.MelSpec.fMax, 8000.0)
        XCTAssertLessThanOrEqual(
            CohereAsrConfig.MelSpec.fMax,
            Float(CohereAsrConfig.sampleRate) / 2.0,
            "fMax should not exceed Nyquist frequency"
        )
    }

    func testPreemphasisIsValid() {
        XCTAssertGreaterThan(CohereAsrConfig.MelSpec.preemphasis, 0.0)
        XCTAssertLessThanOrEqual(CohereAsrConfig.MelSpec.preemphasis, 1.0)
    }

    func testNFFTIsPowerOfTwo() {
        let nFFT = CohereAsrConfig.MelSpec.nFFT
        XCTAssertTrue(nFFT > 0 && (nFFT & (nFFT - 1)) == 0, "nFFT should be a power of 2")
    }

    // MARK: - Language

    func testLanguageRawValuesAreIsoCodes() {
        XCTAssertEqual(CohereAsrConfig.Language.english.rawValue, "en")
        XCTAssertEqual(CohereAsrConfig.Language.french.rawValue, "fr")
        XCTAssertEqual(CohereAsrConfig.Language.german.rawValue, "de")
        XCTAssertEqual(CohereAsrConfig.Language.spanish.rawValue, "es")
        XCTAssertEqual(CohereAsrConfig.Language.italian.rawValue, "it")
        XCTAssertEqual(CohereAsrConfig.Language.portuguese.rawValue, "pt")
        XCTAssertEqual(CohereAsrConfig.Language.dutch.rawValue, "nl")
        XCTAssertEqual(CohereAsrConfig.Language.polish.rawValue, "pl")
        XCTAssertEqual(CohereAsrConfig.Language.greek.rawValue, "el")
        XCTAssertEqual(CohereAsrConfig.Language.arabic.rawValue, "ar")
        XCTAssertEqual(CohereAsrConfig.Language.japanese.rawValue, "ja")
        XCTAssertEqual(CohereAsrConfig.Language.chinese.rawValue, "zh")
        XCTAssertEqual(CohereAsrConfig.Language.vietnamese.rawValue, "vi")
        XCTAssertEqual(CohereAsrConfig.Language.korean.rawValue, "ko")
    }

    func testAllLanguagesHaveEnglishNames() {
        for language in CohereAsrConfig.Language.allCases {
            XCTAssertFalse(language.englishName.isEmpty, "\(language) should have a non-empty English name")
        }
    }

    func testLanguageCount() {
        XCTAssertEqual(CohereAsrConfig.Language.allCases.count, 14, "Cohere supports 14 languages")
    }

    func testEnglishNameExamples() {
        XCTAssertEqual(CohereAsrConfig.Language.english.englishName, "English")
        XCTAssertEqual(CohereAsrConfig.Language.french.englishName, "French")
        XCTAssertEqual(CohereAsrConfig.Language.japanese.englishName, "Japanese")
    }

    // MARK: - Model Architecture

    func testEncoderParameters() {
        XCTAssertEqual(CohereAsrConfig.encoderHiddenSize, 1280)
        XCTAssertEqual(CohereAsrConfig.numEncoderLayers, 48)
    }

    func testDecoderParameters() {
        XCTAssertEqual(CohereAsrConfig.decoderHiddenSize, 1024)
        XCTAssertEqual(CohereAsrConfig.numDecoderLayers, 8)
        XCTAssertEqual(CohereAsrConfig.numDecoderHeads, 8)
    }
}
