import FluidAudio
import Foundation
import OSLog

/// Downloads PocketTTS models and constants from HuggingFace.
public enum PocketTtsResourceDownloader {

    private static let logger = AppLogger(category: "PocketTtsResourceDownloader")

    /// Ensure all PocketTTS models are downloaded and return the cache directory.
    public static func ensureModels() async throws -> URL {
        let cacheDirectory = try cacheDirectory()
        let modelsDirectory = cacheDirectory.appendingPathComponent(
            PocketTtsConstants.defaultModelsSubdirectory)

        let repoDir = modelsDirectory.appendingPathComponent(Repo.pocketTts.folderName)

        // Check that all required directories exist (models + constants_bin)
        let requiredModels = ModelNames.PocketTTS.requiredModels
        let allPresent = requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(model).path)
        }

        if !allPresent {
            logger.info("Downloading PocketTTS models from HuggingFace...")
            try await DownloadUtils.downloadRepo(.pocketTts, to: modelsDirectory)
        } else {
            logger.info("PocketTTS models found in cache")
        }

        return repoDir
    }

    /// Ensure constants (binary blobs + tokenizer) are available.
    public static func ensureConstants(repoDirectory: URL) throws -> PocketTtsConstantsBundle {
        try PocketTtsConstantsLoader.load(from: repoDirectory)
    }

    /// Ensure voice conditioning data is available.
    public static func ensureVoice(
        _ voice: String, repoDirectory: URL
    ) throws -> PocketTtsVoiceData {
        try PocketTtsConstantsLoader.loadVoice(voice, from: repoDirectory)
    }

    // MARK: - Private

    private static func cacheDirectory() throws -> URL {
        let baseDirectory: URL
        #if os(macOS)
        baseDirectory = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard
            let first = FileManager.default.urls(
                for: .cachesDirectory, in: .userDomainMask
            ).first
        else {
            throw TTSError.processingFailed("Failed to locate caches directory")
        }
        baseDirectory = first
        #endif

        let cacheDirectory = baseDirectory.appendingPathComponent("fluidaudio")
        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
        }
        return cacheDirectory
    }
}
