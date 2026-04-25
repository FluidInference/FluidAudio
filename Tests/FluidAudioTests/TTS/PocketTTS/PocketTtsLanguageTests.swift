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

    func testEnglishHasNoRepoSubdirectory() {
        // English keeps the legacy root-level layout so existing caches stay
        // valid without re-downloading into a `v2/english/` folder.
        XCTAssertNil(PocketTtsLanguage.english.repoSubdirectory)
    }

    func testNonEnglishLanguagesUseV2Subdirectory() {
        let expected: [(PocketTtsLanguage, String)] = [
            (.french24L, "v2/french_24l"),
            (.german, "v2/german"),
            (.german24L, "v2/german_24l"),
            (.italian, "v2/italian"),
            (.italian24L, "v2/italian_24l"),
            (.portuguese, "v2/portuguese"),
            (.portuguese24L, "v2/portuguese_24l"),
            (.spanish, "v2/spanish"),
            (.spanish24L, "v2/spanish_24l"),
        ]
        for (lang, path) in expected {
            XCTAssertEqual(
                lang.repoSubdirectory, path,
                "Unexpected repoSubdirectory for \(lang.rawValue)")
        }
    }

    func testAllNonEnglishLanguagesAreCovered() {
        // Guard against silent additions to the enum that forget to update
        // the v2/<id>/ mapping above.
        let nonEnglish = PocketTtsLanguage.allCases.filter { $0 != .english }
        XCTAssertEqual(nonEnglish.count, 9)
        for lang in nonEnglish {
            XCTAssertEqual(
                lang.repoSubdirectory, "v2/\(lang.rawValue)",
                "Language \(lang.rawValue) does not follow v2/<rawValue> convention")
        }
    }

    // MARK: - PocketTtsLanguage.transformerLayers

    func testTransformerLayerCounts() {
        // 6L variants (English plus 4 base non-English packs)
        let sixLayer: [PocketTtsLanguage] = [
            .english, .german, .italian, .portuguese, .spanish,
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

    func testEveryLanguageHasValidLayerCount() {
        // Sanity guard: enum may grow but every variant must be 6 or 24.
        for lang in PocketTtsLanguage.allCases {
            XCTAssertTrue(
                lang.transformerLayers == 6 || lang.transformerLayers == 24,
                "Unexpected layer count \(lang.transformerLayers) for \(lang.rawValue)")
        }
    }

    // MARK: - ModelNames.PocketTTS.requiredModels(for:)

    func testEnglishRequiredModelsUsesLegacyMimi() {
        // English keeps `mimi_decoder_v2.mlmodelc` for backward-compat with
        // the original repo layout on HuggingFace.
        let models = ModelNames.PocketTTS.requiredModels(for: .english)
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.mimiDecoderLegacyFile))
        XCTAssertFalse(models.contains(ModelNames.PocketTTS.mimiDecoderV2File))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.condStepFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.flowlmStepFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.flowDecoderFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.constantsBinDir))
    }

    func testNonEnglishRequiredModelsUsesNewMimi() {
        // Other languages ship `mimi_decoder.mlmodelc` (not the legacy `_v2`).
        for lang in PocketTtsLanguage.allCases where lang != .english {
            let models = ModelNames.PocketTTS.requiredModels(for: lang)
            XCTAssertTrue(
                models.contains(ModelNames.PocketTTS.mimiDecoderV2File),
                "\(lang.rawValue) should require mimi_decoder.mlmodelc")
            XCTAssertFalse(
                models.contains(ModelNames.PocketTTS.mimiDecoderLegacyFile),
                "\(lang.rawValue) should NOT require legacy mimi_decoder_v2.mlmodelc")
        }
    }

    func testRequiredModelsAlwaysHasFiveEntries() {
        // 4 model directories + 1 constants_bin/ directory.
        for lang in PocketTtsLanguage.allCases {
            let models = ModelNames.PocketTTS.requiredModels(for: lang)
            XCTAssertEqual(
                models.count, 5,
                "\(lang.rawValue) requiredModels should have 5 entries")
        }
    }

    // MARK: - ModelNames.PocketTTS.mimiDecoderFile(for:)

    func testMimiDecoderFilenameDispatch() {
        XCTAssertEqual(
            ModelNames.PocketTTS.mimiDecoderFile(for: .english),
            ModelNames.PocketTTS.mimiDecoderLegacyFile)
        for lang in PocketTtsLanguage.allCases where lang != .english {
            XCTAssertEqual(
                ModelNames.PocketTTS.mimiDecoderFile(for: lang),
                ModelNames.PocketTTS.mimiDecoderV2File,
                "\(lang.rawValue) should use mimi_decoder.mlmodelc")
        }
    }

    // MARK: - Backward-compat alias

    func testLegacyRequiredModelsMatchesEnglish() {
        // The legacy `requiredModels` (no language arg) must remain identical
        // to the English-language set so old callers keep working.
        XCTAssertEqual(
            ModelNames.PocketTTS.requiredModels,
            ModelNames.PocketTTS.requiredModels(for: .english))
    }
}
