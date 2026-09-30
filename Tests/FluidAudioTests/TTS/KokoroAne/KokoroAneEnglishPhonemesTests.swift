import Foundation
import XCTest

@testable import FluidAudio

/// Runs `englishPhonemes(for:)` on the Mandarin variant: English through the
/// English frontend without the English chain, in the Mandarin vocab. Skips
/// when the Mandarin bundle or the English G2P assets are not cached; set
/// `FLUIDAUDIO_RUN_TTS_E2E=1` to download them.
final class KokoroAneEnglishPhonemesTests: XCTestCase {
    private func mandarinManager() throws -> (KokoroAneManager, URL) {
        let repoDirectory = try KokoroAneResourceDownloader.repositoryDirectory(variant: .mandarin)
        let modelsCached = ModelNames.KokoroAne.requiredModelsZh.allSatisfy {
            FileManager.default.fileExists(atPath: repoDirectory.appendingPathComponent($0).path)
        }
        let g2pCached = FileManager.default.fileExists(
            atPath: try TtsCacheDirectory.ensure().appendingPathComponent("Models/kokoro").path)
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_TTS_E2E"] == "1"
        try XCTSkipUnless(
            (modelsCached && g2pCached) || allowDownload,
            "Mandarin bundle or English G2P assets not cached; set FLUIDAUDIO_RUN_TTS_E2E=1")
        return (KokoroAneManager(variant: .mandarin), repoDirectory)
    }

    func testEnglishPhonemesOnTheMandarinVariant() async throws {
        let (manager, _) = try mandarinManager()
        // NeMo normalization reads the time; the lexicon gives weak forms.
        let phonemes = try await manager.englishPhonemes(for: "It's 7:30 already.")
        XCTAssertEqual(phonemes, "ɪts sˈɛvən θˈɜɹɾi ˌɔlɹˈɛdi.")
    }

    func testEnglishPhonemesStayInTheMandarinVocab() async throws {
        let (manager, repoDirectory) = try mandarinManager()
        let phonemes = try await manager.englishPhonemes(
            for: "You're snuggling under the blanket, but you are not the bird!")
        let vocab = try KokoroAneVocab.load(
            from: repoDirectory.appendingPathComponent(ModelNames.KokoroAne.vocab))
        let unknown = phonemes.filter { $0 != " " && vocab.map[$0] == nil }
        XCTAssertTrue(unknown.isEmpty, "not in the Mandarin vocab: \(unknown)")
    }
}
