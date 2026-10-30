@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class UnifiedModelFamilyTests: XCTestCase {

    func testEnglishFamilyKeepsParakeetDefaults() {
        let config = UnifiedConfig()
        XCTAssertEqual(config.family, .parakeetEnglish)
        XCTAssertEqual(config.blankIdx, 1024)
        XCTAssertNil(config.family.promptId)
        XCTAssertEqual(config.family.repo, .parakeetUnified)
        // Same bundle names ModelNames.ParakeetUnified ships.
        XCTAssertEqual(
            config.family.streamingEncoderFile(precision: .int8, contextSuffix: "70_13_13"),
            ModelNames.ParakeetUnified.streamingEncoderInt8File)
        XCTAssertEqual(
            config.family.offlineEncoderFile(precision: .fp16), ModelNames.ParakeetUnified.offlineEncoderFp16File)
        XCTAssertEqual(config.family.decoderFile, ModelNames.ParakeetUnified.decoderFile)
        XCTAssertEqual(config.family.jointDecisionFile, ModelNames.ParakeetUnified.jointDecisionFile)
    }

    func testNemotronFamilyUsesItsBlankPromptAndBundles() {
        let config = UnifiedConfig(leftFrames: 70, chunkFrames: 7, rightFrames: 7, family: .nemotronMultilingual(.de))
        XCTAssertEqual(config.blankIdx, 4095)
        XCTAssertEqual(config.family.promptId, 9)
        XCTAssertEqual(config.family.repo, .nemotronUnified)
        XCTAssertEqual(
            config.family.streamingEncoderFile(precision: .fp16, contextSuffix: config.contextSuffix),
            "nemotron_unified_encoder_streaming_70_7_7.mlmodelc")
        XCTAssertEqual(config.family.offlineEncoderFile(precision: .int8), "nemotron_unified_encoder_int8.mlmodelc")
    }

    func testExplicitBlankIndexStillOverrides() {
        XCTAssertEqual(UnifiedConfig(blankIdx: 7, family: .nemotronMultilingual(.en)).blankIdx, 7)
    }

    func testPromptIdsMatchTheCheckpointDictionary() {
        // From nemotron-labs-unified-asr-0.6b model_config.yaml prompt_dictionary (also coreml/metadata.json).
        let expected: [NemotronUnifiedLanguage: Int] = [
            .auto: 127, .en: 0, .es: 3, .fr: 8, .de: 9, .ru: 11, .pt: 13, .it: 15, .nl: 16, .pl: 17, .uk: 19,
            .ro: 20, .el: 21, .cs: 22, .hu: 23, .sv: 24, .da: 25, .fi: 26, .sk: 28, .hr: 29, .bg: 30, .lt: 31,
            .et: 60, .lv: 61, .sl: 62, .mt: 102,
        ]
        XCTAssertEqual(NemotronUnifiedLanguage.allCases.count, 26)
        for language in NemotronUnifiedLanguage.allCases {
            XCTAssertEqual(language.promptId, expected[language], "\(language)")
        }
    }

    func testNemotronRepoLayoutAndRequiredFiles() {
        let repo = Repo.nemotronUnified
        XCTAssertEqual(repo.remotePath, "FluidInference/nemotron-labs-unified-asr-0.6b-coreml")
        XCTAssertEqual(repo.subPath, "coreml")
        let streaming = ModelNames.getRequiredModelNames(for: repo, variant: "fp16")
        XCTAssertEqual(
            streaming,
            [
                "nemotron_unified_decoder.mlmodelc", "nemotron_unified_joint_decision_single_step.mlmodelc",
                "vocab.json", "metadata.json",
            ])
        let offline = ModelNames.getRequiredModelNames(for: repo, variant: "offline-fp16")
        XCTAssertTrue(offline.contains("nemotron_unified_encoder.mlmodelc"))
        XCTAssertTrue(
            ModelNames.getRequiredModelNames(for: repo, variant: "offline").contains(
                "nemotron_unified_encoder_int8.mlmodelc"))
    }

    func testEncoderFeatureProviderAddsPromptIdOnlyWhenGiven() throws {
        let mel = try MLMultiArray(shape: [1, 128, 4], dataType: .float32)
        let melLength = try MLMultiArray(shape: [1], dataType: .int32)
        let english = UnifiedEncoderFeatureProvider(mel: mel, melLength: melLength)
        XCTAssertEqual(english.featureNames, ["mel", "mel_length"])
        XCTAssertNil(english.featureValue(for: "prompt_id"))

        let prompt = try MLMultiArray(shape: [1], dataType: .int32)
        prompt[0] = 17
        let polish = UnifiedEncoderFeatureProvider(mel: mel, melLength: melLength, promptId: prompt)
        XCTAssertEqual(polish.featureNames, ["mel", "mel_length", "prompt_id"])
        XCTAssertEqual(polish.featureValue(for: "prompt_id")?.multiArrayValue?[0].intValue, 17)
        XCTAssertNotNil(polish.featureValue(for: "mel_length"))
    }
}
