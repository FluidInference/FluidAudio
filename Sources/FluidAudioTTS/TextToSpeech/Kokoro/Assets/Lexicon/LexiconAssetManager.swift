import FluidAudio
import Foundation
import OSLog

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
