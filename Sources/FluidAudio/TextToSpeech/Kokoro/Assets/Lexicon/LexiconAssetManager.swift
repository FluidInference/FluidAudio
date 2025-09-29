import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
actor LexiconAssetManager {
    static let shared = LexiconAssetManager()

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "LexiconAssetManager")
    private func ensureLexiconFiles() async throws {
        try await TtsResourceDownloader.ensureLexiconFile(named: "us_gold.json")
        try await TtsResourceDownloader.ensureLexiconFile(named: "us_silver.json")
    }

    private func ensureLexiconCache() async {
        do {
            try await TtsResourceDownloader.ensureLexiconFile(named: "us_lexicon_cache.json")
        } catch {
            logger.warning("Failed to download lexicon cache (will fall back to merge): \(error.localizedDescription)")
        }
    }

    private func ensureEspeakAssets() async {
        let cacheDir = try? TtsModels.cacheDirectoryURL()
        guard let cacheDir else { return }
        let modelsDirectory = cacheDir.appendingPathComponent("Models")
        _ = try? await TtsResourceDownloader.ensureEspeakDataBundle(in: modelsDirectory)
    }

    func ensureCoreAssets() async throws {
        await ensureLexiconCache()
        try await ensureLexiconFiles()
        await ensureEspeakAssets()
    }

    static func ensureCoreAssets() async throws {
        try await LexiconAssetManager.shared.ensureCoreAssets()
    }
}
