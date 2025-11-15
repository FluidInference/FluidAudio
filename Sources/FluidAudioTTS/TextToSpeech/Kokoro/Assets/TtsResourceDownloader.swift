import FluidAudio
import Foundation

/// Kokoro TTS resource downloader (lexicons, voice embeddings)
public enum TtsResourceDownloader {

    private static let logger = AppLogger(category: "TtsResourceDownloader")

    /// Download a voice embedding JSON file from HuggingFace
    public static func downloadVoiceEmbedding(voice: String) async throws -> Data {
        let url = try ModelRegistry.resolveModel(Repo.kokoro.remotePath, "voices/\(voice).json")

        do {
            let data = try await AssetDownloader.fetchData(
                from: url,
                description: "\(voice) voice embedding JSON",
                logger: logger
            )
            logger.info("Downloaded voice embedding JSON for \(voice)")
            return data
        } catch {
            throw TTSError.modelNotFound("Voice embedding JSON unavailable for \(voice): \(error.localizedDescription)")
        }
    }

    /// Ensure a voice embedding is available in cache
    public static func ensureVoiceEmbedding(voice: String = TtsConstants.recommendedVoice) async throws {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let voicesDir = cacheDir.appendingPathComponent("Models/kokoro/voices")

        // Create directory if needed
        try FileManager.default.createDirectory(at: voicesDir, withIntermediateDirectories: true)

        let jsonFile = "\(voice).json"
        let jsonURL = voicesDir.appendingPathComponent(jsonFile)

        // Skip if already cached
        if FileManager.default.fileExists(atPath: jsonURL.path) {
            return
        }

        // Try to download
        let data = try await downloadVoiceEmbedding(voice: voice)
        try data.write(to: jsonURL, options: [.atomic])
        logger.info("Voice embedding cached: \(voice)")
    }

    /// Ensure a Kokoro lexicon file exists locally (e.g. `us_gold.json`).
    @discardableResult
    public static func ensureLexiconFile(named filename: String) async throws -> URL {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let localURL = kokoroDir.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        let remoteURL = try ModelRegistry.resolveModel(Repo.kokoro.remotePath, filename)

        do {
            let descriptor = AssetDownloader.Descriptor(
                description: filename,
                remoteURL: remoteURL,
                destinationURL: localURL
            )
            return try await AssetDownloader.ensure(
                descriptor,
                logger: logger
            )
        } catch {
            throw TTSError.modelNotFound("Failed to download \(filename): \(error.localizedDescription)")
        }
    }

}
