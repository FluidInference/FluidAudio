import FluidAudio
import Foundation

/// Kokoro TTS resource downloader (lexicons, voice embeddings)
public enum TtsResourceDownloader {

    private static let logger = AppLogger(category: "TtsResourceDownloader")
    private static let zhRepoRemotePath = "alexwengg/tts-zh"

    /// Download a voice embedding JSON file from HuggingFace
    public static func downloadVoiceEmbedding(voice: String) async throws -> Data {
        // Primary repo (FluidInference/kokoro-82m-coreml)
        let primary = try ModelRegistry.resolveModel(Repo.kokoro.remotePath, "voices/\(voice).json")
        // Fallback repo for zh voices
        let fallback = try ModelRegistry.resolveModel(zhRepoRemotePath, "voices/\(voice).json")

        func attempt(_ url: URL) async throws -> Data {
            try await AssetDownloader.fetchData(
                from: url,
                description: "\(voice) voice embedding JSON",
                logger: logger
            )
        }

        do {
            let data = try await attempt(primary)
            logger.info("Downloaded voice embedding JSON for \(voice) from primary repo")
            return data
        } catch {
            logger.warning("Primary voice embedding missing for \(voice); trying zh repo")
            do {
                let data = try await attempt(fallback)
                logger.info("Downloaded voice embedding JSON for \(voice) from zh repo")
                return data
            } catch {
                throw TTSError.modelNotFound("Voice embedding JSON unavailable for \(voice)")
            }
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

// MARK: - zh assets (vocab + char lexicon)

extension TtsResourceDownloader {
    /// Ensure `zh_vocab_index.json` exists in Kokoro cache. Falls back to downloading `zh.json` and writing it
    /// as `zh_vocab_index.json` if needed.
    @discardableResult
    public static func ensureZhVocabularyInCache() async throws -> URL {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let zhVocabURL = kokoroDir.appendingPathComponent("zh_vocab_index.json")
        if FileManager.default.fileExists(atPath: zhVocabURL.path) {
            return zhVocabURL
        }

        let primary = try ModelRegistry.resolveModel(zhRepoRemotePath, "zh_vocab_index.json")
        let alt = try ModelRegistry.resolveModel(zhRepoRemotePath, "zh.json")

        do {
            let descriptor = AssetDownloader.Descriptor(
                description: "zh_vocab_index.json",
                remoteURL: primary,
                destinationURL: zhVocabURL
            )
            return try await AssetDownloader.ensure(descriptor, logger: logger)
        } catch {
            logger.warning("zh_vocab_index.json not found in zh repo; trying zh.json")
            let data = try await AssetDownloader.fetchData(
                from: alt,
                description: "zh.json",
                logger: logger
            )
            try data.write(to: zhVocabURL, options: [.atomic])
            logger.info("Cached zh vocabulary as zh_vocab_index.json")
            return zhVocabURL
        }
    }

    /// Ensure `zh_char_phonemes.json` exists in Kokoro cache, downloading from zh repo if missing.
    @discardableResult
    public static func ensureZhCharPhonemesInCache() async throws -> URL {
        let cacheDir = try TtsModels.cacheDirectoryURL()
        let kokoroDir = cacheDir.appendingPathComponent("Models/kokoro")
        try FileManager.default.createDirectory(at: kokoroDir, withIntermediateDirectories: true)

        let localURL = kokoroDir.appendingPathComponent("zh_char_phonemes.json")
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        let remoteURL = try ModelRegistry.resolveModel(zhRepoRemotePath, "zh_char_phonemes.json")
        let descriptor = AssetDownloader.Descriptor(
            description: "zh_char_phonemes.json",
            remoteURL: remoteURL,
            destinationURL: localURL
        )
        return try await AssetDownloader.ensure(descriptor, logger: logger)
    }

    /// Convenience: ensure both zh vocab and char phoneme lexicon exist in cache.
    public static func ensureZhAssetsInCache() async throws -> (vocabURL: URL, lexURL: URL) {
        async let v = ensureZhVocabularyInCache()
        async let l = ensureZhCharPhonemesInCache()
        return try await (vocabURL: v, lexURL: l)
    }
}
