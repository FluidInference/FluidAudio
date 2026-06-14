import Foundation
import XCTest

@testable import FluidAudio

/// Phase-1 variant-plumbing tests for `KokoroAneManager` Mandarin support.
///
/// These tests are intentionally network-free — they verify that the variant
/// enum, repo wiring, and the Mandarin text-path rejection work correctly
/// without needing the HF bundle to be present locally. Live integration
/// tests land in Phase 2 alongside the Mandarin G2P implementation.
final class KokoroAneVariantTests: XCTestCase {

    // MARK: - KokoroAneVariant enum

    func testVariantDefaultVoice() {
        XCTAssertEqual(KokoroAneVariant.english.defaultVoice, "af_heart")
        XCTAssertEqual(KokoroAneVariant.mandarin.defaultVoice, "zf_001")
        XCTAssertEqual(
            KokoroAneVariant.english.defaultVoice,
            KokoroAneConstants.defaultVoice)
        XCTAssertEqual(
            KokoroAneVariant.mandarin.defaultVoice,
            KokoroAneConstants.defaultVoiceMandarin)
    }

    func testVariantUseVoicesSubdir() {
        XCTAssertFalse(KokoroAneVariant.english.useVoicesSubdir)
        XCTAssertTrue(KokoroAneVariant.mandarin.useVoicesSubdir)
    }

    func testVariantRepo() {
        XCTAssertEqual(KokoroAneVariant.english.repo, .kokoroAne)
        XCTAssertEqual(KokoroAneVariant.mandarin.repo, .kokoroAneZh)
    }

    func testVariantAllCases() {
        XCTAssertEqual(KokoroAneVariant.allCases.count, 3)
        XCTAssertTrue(KokoroAneVariant.allCases.contains(.english))
        XCTAssertTrue(KokoroAneVariant.allCases.contains(.mandarin))
        XCTAssertTrue(KokoroAneVariant.allCases.contains(.japanese))
    }

    // MARK: - Repo wiring

    func testRepoSubPathAndFolderName() {
        XCTAssertEqual(Repo.kokoroAneZh.subPath, "ANE-zh")
        XCTAssertEqual(Repo.kokoroAneZh.folderName, "kokoro-82m-coreml/ANE-zh")
        XCTAssertEqual(Repo.kokoroAneZh.remotePath, "FluidInference/kokoro-82m-coreml")
        // Sanity: existing English variant unchanged.
        XCTAssertEqual(Repo.kokoroAne.subPath, "ANE")
        XCTAssertEqual(Repo.kokoroAne.folderName, "kokoro-82m-coreml/ANE")
    }

    // MARK: - ModelNames.KokoroAne required-files set

    func testRequiredModelsZhContainsVoicesPrefix() {
        let required = ModelNames.KokoroAne.requiredModelsZh
        XCTAssertTrue(
            required.contains("voices/zf_001.bin"),
            "Mandarin required-models set must reference voices/<default>.bin so "
                + "the downloader's all-files-present check resolves to "
                + "<repoDir>/voices/zf_001.bin"
        )
        // English default voice file should NOT live in the Mandarin set.
        XCTAssertFalse(required.contains("af_heart.bin"))
        // All 7 mlmodelc bundles + vocab should still be present.
        XCTAssertTrue(required.contains(ModelNames.KokoroAne.albert))
        XCTAssertTrue(required.contains(ModelNames.KokoroAne.vocab))
    }

    func testRequiredModelsEnglishUnchanged() {
        let required = ModelNames.KokoroAne.requiredModels
        XCTAssertTrue(required.contains("af_heart.bin"))
        XCTAssertFalse(required.contains("voices/zf_001.bin"))
    }

    func testGetRequiredModelNamesRoutesByRepo() {
        let zh = ModelNames.getRequiredModelNames(for: .kokoroAneZh, variant: nil)
        XCTAssertTrue(zh.contains("voices/zf_001.bin"))
        let en = ModelNames.getRequiredModelNames(for: .kokoroAne, variant: nil)
        XCTAssertTrue(en.contains("af_heart.bin"))
    }

    // MARK: - Manager init signature

    func testManagerInitDefaultVariant() async {
        // Constructing the manager must not require any network access — only
        // verifies the init signature compiles + variant defaults to English.
        let manager = KokoroAneManager()
        let available = await manager.isAvailable()
        XCTAssertFalse(available, "Fresh manager should not yet be loaded")
    }

    func testManagerInitMandarinVariant() async {
        // Same — Mandarin construction is purely a property assignment.
        let manager = KokoroAneManager(variant: .mandarin)
        let available = await manager.isAvailable()
        XCTAssertFalse(available)
    }

    // Note: Phase-1 used to assert that `synthesize(text:)` on a Mandarin
    // manager throws (G2P deferred). Phase 2 replaced that with a real
    // MandarinG2P pipeline, so the rejection path no longer exists. The
    // G2P rules themselves are covered network-free by
    // `MandarinG2PTests`.
}
