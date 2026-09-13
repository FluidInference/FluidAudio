import Foundation
import XCTest

@testable import FluidAudio

/// Config tests for the `.japanese` KokoroAne variant (issues #698 and #914).
///
/// The Japanese variant reuses the language-agnostic 7-stage chain and the
/// `synthesizeFromPhonemes` bypass and the lazy OpenJTalk text frontend. These
/// tests pin the wiring (HF paths, default voice, required-file set, routing) so a
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
        // OpenJTalk is a package runtime + lazy dictionary, not a CoreML model.
        XCTAssertFalse(required.contains(ModelNames.KokoroAne.g2pwModelZh))
        XCTAssertEqual(required.count, 9)
    }

    func testRequiredModelNamesRoutesToJa() {
        XCTAssertEqual(
            ModelNames.getRequiredModelNames(for: .kokoroAneJa, variant: nil),
            ModelNames.KokoroAne.requiredModelsJa)
    }

    func testPrecomputedPhonemesRemainPassThrough() async throws {
        let manager = KokoroAneManager(variant: .japanese)
        let ipa = "aɾʲiɡatoː"
        let result = try await manager.phonemes(for: ipa)
        XCTAssertEqual(result, ipa)
    }

    func testOpenJTalkDictionaryPin() {
        XCTAssertEqual(KokoroAneConstants.japaneseDictionaryArchiveBytes, 23_646_843)
        XCTAssertEqual(
            KokoroAneConstants.japaneseDictionaryArchiveSHA256,
            "fe6ba0e43542cef98339abdffd903e062008ea170b04e7e2a35da805902f382a")
        XCTAssertTrue(
            KokoroAneConstants.japaneseDictionaryArchiveURL.absoluteString.contains("v1.11.1"))
    }
}
