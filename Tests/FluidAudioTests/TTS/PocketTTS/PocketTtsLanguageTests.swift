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

    func testAllLanguagesUseV2Subdirectory() {
        // Every language pack lives under `v2/<rawValue>/` on the HF repo.
        for lang in PocketTtsLanguage.allCases {
            XCTAssertEqual(
                lang.repoSubdirectory, "v2/\(lang.rawValue)",
                "Language \(lang.rawValue) does not follow v2/<rawValue> convention")
        }
    }

    // MARK: - PocketTtsLanguage.transformerLayers

    func testTransformerLayerCounts() {
        // 6L variants
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

    // MARK: - ModelNames.PocketTTS.requiredModels

    func testRequiredModels() {
        let models = ModelNames.PocketTTS.requiredModels
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.condStepFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.flowlmStepFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.flowDecoderFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.mimiDecoderFile))
        XCTAssertTrue(models.contains(ModelNames.PocketTTS.constantsBinDir))
    }
}
