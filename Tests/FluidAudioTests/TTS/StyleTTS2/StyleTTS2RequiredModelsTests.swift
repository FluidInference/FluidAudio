import Foundation
import XCTest

@testable import FluidAudio

/// Pin the `ModelNames.StyleTTS2` contract so accidental renames or
/// drops surface as a test failure rather than as a cryptic
/// "missing required asset after download" at runtime.
///
/// Mirrors `MagpieConstantsTests.swift` style: hard-coded literal expected
/// filenames + count, plus a few cross-references against
/// `StyleTTS2Stage.bundleName` and `Repo.styleTts2`.
final class StyleTTS2RequiredModelsTests: XCTestCase {

    // MARK: - Filename literals (intentionally hard-coded)

    func testRequiredCoreMLModelsCount() {
        XCTAssertEqual(
            ModelNames.StyleTTS2.requiredCoreMLModels.count, 7,
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
        XCTAssertEqual(ModelNames.StyleTTS2.requiredCoreMLModels, expected)
    }

    func testIndividualFilenameConstantsMatchStageEnum() {
        // Each `StyleTTS2Stage.bundleName` must be present in the
        // required-models set — otherwise the model store would attempt to
        // load a file the downloader didn't fetch.
        for stage in StyleTTS2Stage.allCases {
            XCTAssertTrue(
                ModelNames.StyleTTS2.requiredCoreMLModels.contains(stage.bundleName),
                "\(stage.bundleName) is referenced by StyleTTS2Stage but missing "
                    + "from ModelNames.StyleTTS2.requiredCoreMLModels")
        }
    }

    func testFilenameConstantsExposed() {
        // These constants are part of the public ModelNames contract; pin
        // them so renames force a deliberate API decision.
        XCTAssertEqual(
            ModelNames.StyleTTS2.plBertFile, "styletts2_ane_plbert.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2.postBertFile, "styletts2_ane_postbert.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2.alignmentFile, "styletts2_ane_alignment.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2.diffusionStepFile,
            "styletts2_ane_diffusion_step.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2.prosodyFile, "styletts2_ane_prosody.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2.noiseFile, "styletts2_ane_noise.mlmodelc")
        XCTAssertEqual(
            ModelNames.StyleTTS2.vocoderFile, "styletts2_ane_vocoder.mlmodelc")
    }

    // MARK: - Repo wiring

    func testRepoStyleTts2AneFields() {
        XCTAssertEqual(Repo.styleTts2.subPath, "ANE")
        XCTAssertEqual(Repo.styleTts2.folderName, "styletts2/ANE")
        XCTAssertEqual(Repo.styleTts2.remotePath, "FluidInference/StyleTTS-2-coreml")
        XCTAssertEqual(Repo.styleTts2.name, "StyleTTS-2-coreml/ANE")
    }

    func testGetRequiredModelNamesRoutesStyleTts2Ane() {
        let names = ModelNames.getRequiredModelNames(for: .styleTts2, variant: nil)
        XCTAssertEqual(names, ModelNames.StyleTTS2.requiredCoreMLModels)
    }

    func testStyleTts2AneSetIsDisjointFromLegacyStyleTts2() {
        // The two backends share zero files on purpose — the ANE bundle
        // ships the 7 single-graph mlmodelcs while the legacy bundle ships
        // the 12 bucketed ones. If they ever overlap it means somebody
        // accidentally imported a legacy filename.
        let legacy = ModelNames.StyleTTS2.requiredModels
        let ane = ModelNames.StyleTTS2.requiredCoreMLModels
        XCTAssertTrue(
            ane.isDisjoint(with: legacy),
            "StyleTTS2-ANE and legacy StyleTTS2 required-files must not overlap")
    }
}
