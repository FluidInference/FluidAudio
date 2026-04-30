import XCTest

@testable import FluidAudio

final class MagpieConstantsTests: XCTestCase {

    func testForbiddenTokenIdsExcludeEos() {
        // The sampler masks these auxiliary tokens unconditionally; audioEosId is only
        // masked during the first `minFrames` steps, so it must NOT be in the forbidden list.
        XCTAssertFalse(
            MagpieConstants.forbiddenAudioIds.contains(MagpieConstants.audioEosId),
            "audioEosId must be sampleable outside the min-frames window"
        )
        XCTAssertTrue(
            MagpieConstants.forbiddenAudioIds.contains(MagpieConstants.audioBosId),
            "audioBosId should never be sampled"
        )
    }

    func testShapeRelationships() {
        XCTAssertEqual(MagpieConstants.dModel, MagpieConstants.numHeads * MagpieConstants.headDim)
        XCTAssertGreaterThan(MagpieConstants.maxCacheLength, MagpieConstants.speakerContextLength)
        XCTAssertEqual(MagpieConstants.numSpeakers, 5)
    }

    func testTokenizerNameMatchesNemoNaming() {
        // These strings are required by the mobius exporter (see
        // generate_coreml._tokenize_text); changing either side silently breaks parity.
        XCTAssertEqual(MagpieTokenizerFiles.tokenizerName(for: .english), "english_phoneme")
        XCTAssertEqual(MagpieTokenizerFiles.tokenizerName(for: .french), "french_chartokenizer")
        XCTAssertEqual(MagpieTokenizerFiles.tokenizerName(for: .mandarin), "mandarin_phoneme")
        XCTAssertEqual(MagpieTokenizerFiles.tokenizerName(for: .hindi), "hindi_chartokenizer")
    }

    func testTokenizerFilesCoverAllLanguages() {
        for lang in MagpieLanguage.allCases {
            let files = MagpieTokenizerFiles.files(for: lang)
            XCTAssertFalse(
                files.isEmpty,
                "Expected at least one tokenizer file for \(lang.rawValue)")
            XCTAssertTrue(
                files.contains { $0.hasSuffix("_token2id.json") },
                "Language \(lang.rawValue) must ship a token2id map")
        }
    }

    /// Magpie is the only repo that ships bare directory entries in its
    /// required-models set (the global `ModelNamesTests.testModelFileExtensions`
    /// excludes it). Pin the contract here so the experimental backend's
    /// layout stays valid without loosening assertions on production repos.
    func testRequiredModelsLayout() {
        let required = ModelNames.Magpie.requiredModels
        let validExtensions: Set<String> = [".mlmodelc", ".json", ".bin"]
        let allowedDirectories: Set<String> = [
            ModelNames.Magpie.constantsDir,
            ModelNames.Magpie.tokenizerDir,
        ]

        for entry in required {
            let hasValidExtension = validExtensions.contains(where: { entry.hasSuffix($0) })
            let isAllowedDirectory = allowedDirectories.contains(entry)
            XCTAssertTrue(
                hasValidExtension || isAllowedDirectory,
                "Magpie required entry '\(entry)' must have a valid file extension or be an allowed directory"
            )
        }

        // Sanity: the three core CoreML models + the constants directory are
        // the minimum required for English synthesis.
        XCTAssertTrue(required.contains(ModelNames.Magpie.textEncoderFile))
        XCTAssertTrue(required.contains(ModelNames.Magpie.decoderStepFile))
        XCTAssertTrue(required.contains(ModelNames.Magpie.nanocodecDecoderFile))
        XCTAssertTrue(required.contains(ModelNames.Magpie.constantsDir))
    }
}
