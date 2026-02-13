import FluidAudio
import Foundation
import OSLog

/// Downloads Qwen3-TTS models and data files from HuggingFace.
public enum Qwen3TtsResourceDownloader {

    private static let logger = AppLogger(category: "Qwen3TtsResourceDownloader")

    /// Ensure all Qwen3-TTS models are downloaded and return the cache directory.
    public static func ensureModels() async throws -> URL {
        let cacheDirectory = try cacheDirectory()
        let modelsDirectory = cacheDirectory.appendingPathComponent("Models")

        let repoDir = modelsDirectory.appendingPathComponent(Repo.qwen3Tts.folderName)

        // Check that all required files exist
        let requiredModels = ModelNames.Qwen3TTS.requiredModels
        let allPresent = requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(model).path)
        }

        if !allPresent {
            logger.info("Downloading Qwen3-TTS models from HuggingFace...")
            try await DownloadUtils.downloadRepo(.qwen3Tts, to: modelsDirectory)
        } else {
            logger.info("Qwen3-TTS models found in cache")
        }

        return repoDir
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
