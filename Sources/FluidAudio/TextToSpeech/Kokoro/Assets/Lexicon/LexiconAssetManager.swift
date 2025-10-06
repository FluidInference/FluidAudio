import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
actor LexiconAssetManager {

    private let logger = AppLogger(subsystem: "com.fluidaudio.tts", category: "LexiconAssetManager")
    private func ensureLexiconCache() async {
        do {
            try await TtsResourceDownloader.ensureLexiconFile(named: "us_lexicon_cache.json")
        } catch {
            logger.warning("Failed to download lexicon cache: \(error.localizedDescription)")
        }
    }

    func ensureCoreAssets() async throws {
        await ensureLexiconCache()
    }
}
