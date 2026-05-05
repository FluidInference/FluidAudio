import Foundation
import XCTest

@testable import FluidAudio

/// Pin the `ModelNames.StyleTTS2Ane` contract so accidental renames or
/// drops surface as a test failure rather than as a cryptic
/// "missing required asset after download" at runtime.
///
/// Mirrors `MagpieConstantsTests.swift` style: hard-coded literal expected
/// filenames + count, plus a few cross-references against
/// `StyleTTS2AneStage.bundleName` and `Repo.styleTts2Ane`.
final class StyleTTS2AneRequiredModelsTests: XCTestCase {

    // MARK: - Filename literals (intentionally hard-coded)

    func testRequiredCoreMLModelsCount() {
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.requiredCoreMLModels.count, 7,
            "StyleTTS2-ANE must require exactly 7 mlmodelcs (PLBert, PostBert, "
                + "Alignment, DiffusionStep, Prosody, Noise, Vocoder)")
    }

    func testRequiredCoreMLModelsContainsAllSevenStages() {
        let expected: Set<String> = [
            "styletts2_ane_plbert.mlmodelc",
            "styletts2_ane_postbert.mlmodelc",
            "styletts2_ane_alignment.mlmodelc",
            "styletts2_ane_diffusion_step.mlmodelc",
            "styletts2_ane_prosody.mlmodelc",
            "styletts2_ane_noise.mlmodelc",
            "styletts2_ane_vocoder.mlmodelc",
        ]
        XCTAssertEqual(ModelNames.StyleTTS2Ane.requiredCoreMLModels, expected)
    }

    func testIndividualFilenameConstantsMatchStageEnum() {
        // Each `StyleTTS2AneStage.bundleName` must be present in the
        // required-models set — otherwise the model store would attempt to
        // load a file the downloader didn't fetch.
        for stage in StyleTTS2AneStage.allCases {
            XCTAssertTrue(
                ModelNames.StyleTTS2Ane.requiredCoreMLModels.contains(stage.bundleName),
                "\(stage.bundleName) is referenced by StyleTTS2AneStage but missing "
                    + "from ModelNames.StyleTTS2Ane.requiredCoreMLModels")
        }
    }

    func testFilenameConstantsExposed() {
        // These constants are part of the public ModelNames contract; pin
        // them so renames force a deliberate API decision.
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.plBertFile, "styletts2_ane_plbert.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.postBertFile, "styletts2_ane_postbert.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.alignmentFile, "styletts2_ane_alignment.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.diffusionStepFile,
            "styletts2_ane_diffusion_step.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.prosodyFile, "styletts2_ane_prosody.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.noiseFile, "styletts2_ane_noise.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2Ane.vocoderFile, "styletts2_ane_vocoder.mlmodelc")
    }

    // MARK: - Repo wiring

    func testRepoStyleTts2AneFields() {
        XCTAssertEqual(Repo.styleTts2Ane.subPath, "ANE")
        XCTAssertEqual(Repo.styleTts2Ane.folderName, "styletts2/ANE")
        XCTAssertEqual(Repo.styleTts2Ane.remotePath, "FluidInference/StyleTTS-2-coreml")
        XCTAssertEqual(Repo.styleTts2Ane.name, "StyleTTS-2-coreml/ANE")
    }

    func testGetRequiredModelNamesRoutesStyleTts2Ane() {
        let names = ModelNames.getRequiredModelNames(for: .styleTts2Ane, variant: nil)
        XCTAssertEqual(names, ModelNames.StyleTTS2Ane.requiredCoreMLModels)
    }

    func testStyleTts2AneSetIsDisjointFromLegacyStyleTts2() {
        // The two backends share zero files on purpose — the ANE bundle
        // ships the 7 single-graph mlmodelcs while the legacy bundle ships
        // the 12 bucketed ones. If they ever overlap it means somebody
        // accidentally imported a legacy filename.
        let legacy = ModelNames.StyleTTS2.requiredModels
        let ane = ModelNames.StyleTTS2Ane.requiredCoreMLModels
        XCTAssertTrue(
            ane.isDisjoint(with: legacy),
            "StyleTTS2-ANE and legacy StyleTTS2 required-files must not overlap")
    }
}
