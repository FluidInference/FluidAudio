import Foundation
import XCTest

@testable import FluidAudio

/// Config tests for the `.japanese` KokoroAne variant (issue #698).
///
/// The Japanese variant reuses the language-agnostic 7-stage chain and the
/// `synthesizeFromPhonemes` bypass; it ships no text frontend. These tests
/// pin the wiring (HF paths, default voice, required-file set, routing) so a
/// regression surfaces without needing the (separately uploaded) `ANE-ja/`
/// CoreML weights.
final class KokoroAneJapaneseVariantTests: XCTestCase {

    func testVariantIsEnumerated() {
        XCTAssertTrue(KokoroAneVariant.allCases.contains(.japanese))
    }

    func testVariantConfig() {
        XCTAssertEqual(KokoroAneVariant.japanese.defaultVoice, "jf_alpha")
        XCTAssertEqual(KokoroAneVariant.japanese.repo, .kokoroAneJa)
        // Voice packs nest under `voices/` (mirrors Mandarin).
        XCTAssertTrue(KokoroAneVariant.japanese.useVoicesSubdir)
    }

    func testRepoPaths() {
        let repo = Repo.kokoroAneJa
        XCTAssertEqual(repo.subPath, "ANE-ja")
        XCTAssertEqual(repo.folderName, "kokoro-82m-coreml/ANE-ja")
        XCTAssertEqual(repo.remotePath, "FluidInference/kokoro-82m-coreml")
        XCTAssertEqual(repo.name, "kokoro-82m-coreml/ANE-ja")
    }

    func testRequiredModels() {
        let required = ModelNames.KokoroAne.requiredModelsJa
        // 7 CoreML bundles + vocab + the nested default voice = 9 files.
        XCTAssertTrue(required.isSuperset(of: ModelNames.KokoroAne.requiredCoreMLModels))
        XCTAssertTrue(required.contains(ModelNames.KokoroAne.vocab))
        XCTAssertTrue(required.contains(ModelNames.KokoroAne.defaultVoiceFileJa))
        XCTAssertEqual(ModelNames.KokoroAne.defaultVoiceFileJa, "voices/jf_alpha.bin")
        // No Mandarin g2pW bundle — Japanese has no in-process G2P.
        XCTAssertFalse(required.contains(ModelNames.KokoroAne.g2pwModelZh))
        XCTAssertEqual(required.count, 9)
    }

    func testRequiredModelNamesRoutesToJa() {
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .kokoroAneJa, variant: nil),
            ModelNames.KokoroAne.requiredModelsJa)
    }

    /// The Japanese variant has no text→phoneme frontend; `phonemes(for:)`
    /// must throw rather than silently mis-synthesize. (Phoneme bypass via
    /// `synthesizeFromPhonemes` does not route through this method.)
    func testTextFrontendThrows() async {
        let manager = KokoroAneManager(variant: .japanese)
        do {
            _ = try await manager.phonemes(for: "ありがとう")
            XCTFail("expected phonemes(for:) to throw on the Japanese variant")
        } catch let error as KokoroAneError {
            guard case .inputProcessingFailed = error else {
                return XCTFail("expected inputProcessingFailed, got \(error)")
            }
        } catch {
            XCTFail("expected KokoroAneError.inputProcessingFailed, got \(error)")
        }
    }
}
