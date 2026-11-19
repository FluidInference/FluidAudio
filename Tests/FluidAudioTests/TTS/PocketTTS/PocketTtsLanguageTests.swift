import Foundation
import XCTest

@testable import FluidAudio

/// Pure-logic unit tests for PocketTTS multi-language plumbing.
///
/// These tests exercise the path/filename/layer-count derivation that drives
/// HuggingFace downloads and CoreML model selection. They do not require any
/// model files or network access.
final class PocketTtsLanguageTests: XCTestCase {

    // MARK: - PocketTtsLanguage.repoSubdirectory

    func testAllLanguagesUseV21Subdirectory() {
        // v2.1 optimized packs live under `v2.1/<rawValue>/` on the HF repo, except
        // English: `.english` follows upstream's alias to english_2026-10, and the
        // April pack keeps the `v2.1/english/` folder it shipped in (#1087).
        XCTAssertEqual(PocketTtsLanguage.english.repoSubdirectory, "v2.1/english_2026-10")
        XCTAssertEqual(PocketTtsLanguage.english2026Apr.repoSubdirectory, "v2.1/english")
        for lang in PocketTtsLanguage.allCases where !lang.isEnglish {
            XCTAssertEqual(
                lang.repoSubdirectory, "v2.1/\(lang.rawValue)",
                "Language \(lang.rawValue) does not follow v2.1/<rawValue> convention")
        }
    }

    // MARK: - PocketTtsLanguage.transformerLayers

    func testTransformerLayerCounts() {
        // 6L variants
        let sixLayer: [PocketTtsLanguage] = [
            .english, .english2026Apr, .german, .italian, .portuguese, .spanish,
        ]
        for lang in sixLayer {
            XCTAssertEqual(
                lang.transformerLayers, 6,
                "\(lang.rawValue) should be a 6-layer pack")
        }

        // 24L variants (note: French ships only the 24L variant upstream)
        let twentyFourLayer: [PocketTtsLanguage] = [
            .french24L, .german24L, .italian24L, .portuguese24L, .spanish24L,
        ]
        for lang in twentyFourLayer {
            XCTAssertEqual(
                lang.transformerLayers, 24,
                "\(lang.rawValue) should be a 24-layer pack")
        }
    }

    // MARK: - English checkpoints (#1087)

    func testEnglishPacks() {
        XCTAssertEqual(PocketTtsLanguage(rawValue: "english"), .english)
        XCTAssertEqual(PocketTtsLanguage(rawValue: "english_2026-04"), .english2026Apr)
        for lang in [PocketTtsLanguage.english, .english2026Apr] {
            XCTAssertTrue(lang.isEnglish)
            XCTAssertEqual(
                PocketTtsSynthesizer.abbreviations(for: lang),
                PocketTtsSynthesizer.abbreviations(for: .german))
            XCTAssertEqual(PocketTtsSynthesizer.normalizeForLanguage("« 3 h »", language: lang), "« 3 h »")
        }
        XCTAssertFalse(PocketTtsLanguage.german.isEnglish)
    }

    func testDefaultTemperatures() {
        // Kyutai's english_2026-10 config sets default_temperature 0.3; every other pack keeps 0.7.
        XCTAssertEqual(PocketTtsLanguage.english.defaultTemperature, 0.3)
        for lang in PocketTtsLanguage.allCases where lang != .english {
            XCTAssertEqual(
                lang.defaultTemperature, PocketTtsConstants.temperature,
                "\(lang.rawValue) should keep the 0.7 default")
        }
    }

    func testPacksThatReprojectTheSharedEncoder() {
        // `.english` shares the April codec but has its own speaker projection, so cloning
        // with the shared encoder re-projects, like the 24-layer packs (#793, #1087).
        XCTAssertTrue(PocketTtsLanguage.english.reprojectsSharedEncoder)
        XCTAssertFalse(PocketTtsLanguage.english2026Apr.reprojectsSharedEncoder)
        XCTAssertFalse(PocketTtsLanguage.german.reprojectsSharedEncoder)
        XCTAssertTrue(PocketTtsLanguage.german24L.reprojectsSharedEncoder)
    }

    // MARK: - ModelNames.PocketTTS.requiredModels

    func testRequiredModels() {
        // v2.1 required set: cond_prefill + flow_decoder_fused replace the v2
        // cond_step + per-step flow_decoder.
        let models = ModelNames.PocketTTS.requiredModels
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.condPrefillFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.flowlmStepFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.flowDecoderFusedFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.mimiDecoderFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.constantsBinDir))
    }
}
