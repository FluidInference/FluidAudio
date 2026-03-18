import Foundation
import OSLog

/// Downloads PocketTTS models and constants from HuggingFace.
public enum PocketTtsResourceDownloader {

    private static let logger = AppLogger(category: "PocketTtsResourceDownloader")

    /// Ensure all PocketTTS models are downloaded and return the cache directory.
    ///
    /// - Parameters:
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    ///   - progressHandler: Optional callback for download progress updates.
    public static func ensureModels(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
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
            try await DownloadUtils.downloadRepo(.pocketTts, to: modelsDirectory, progressHandler: progressHandler)
        } else {
            logger.info("PocketTTS models found in cache")
        }

        return repoDir
    }

    /// Ensure the Mimi encoder model is downloaded for voice cloning.
    ///
    /// This is an optional model that's only needed for voice cloning functionality.
    /// It's downloaded separately from the main models to reduce initial download size.
    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public static func ensureMimiEncoder(directory: URL? = nil) async throws -> URL {
        let repoDir = try await ensureModels(directory: directory)
        let encoderPath = repoDir.appendingPathComponent(ModelNames.PocketTTS.mimiEncoderFile)

        if FileManager.default.fileExists(atPath: encoderPath.path) {
            logger.info("Mimi encoder found in cache")
            return encoderPath
        }

        logger.info("Downloading Mimi encoder for voice cloning...")
        try await downloadMimiEncoder(to: repoDir)

        guard FileManager.default.fileExists(atPath: encoderPath.path) else {
            throw PocketTTSError.downloadFailed("Failed to download Mimi encoder model")
        }

        return encoderPath
    }

    /// Download the Mimi encoder model files from HuggingFace.
    private static func downloadMimiEncoder(to repoDir: URL) async throws {
        try await DownloadUtils.downloadSubdirectory(
            .pocketTts,
            subdirectory: ModelNames.PocketTTS.mimiEncoderFile,
            to: repoDir
        )
    }

    /// Ensure constants (binary blobs + tokenizer) are available.
    public static func ensureConstants(repoDirectory: URL) throws -> PocketTtsConstantsBundle {
        try PocketTtsConstantsLoader.load(from: repoDirectory)
    }

    /// Ensure voice conditioning data is available, downloading from HuggingFace if missing.
    public static func ensureVoice(
        _ voice: String, repoDirectory: URL
    ) async throws -> PocketTtsVoiceData {
        let sanitized = voice.filter { $0.isLetter || $0.isNumber || $0 == "_" }
        guard !sanitized.isEmpty else {
            throw PocketTTSError.processingFailed("Invalid voice name: \(voice)")
        }
        let constantsDir = repoDirectory.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let voiceFile = "\(sanitized)_audio_prompt.bin"
        let voiceURL = constantsDir.appendingPathComponent(voiceFile)

        if !FileManager.default.fileExists(atPath: voiceURL.path) {
            logger.info("Downloading voice '\(sanitized)' from HuggingFace...")
            let remotePath = "constants_bin/\(voiceFile)"
            let remoteURL = try ModelRegistry.resolveModel(Repo.pocketTts.remotePath, remotePath)
            let data = try await AssetDownloader.fetchData(
                from: remoteURL,
                description: "\(sanitized) voice prompt",
                logger: logger
            )
            try data.write(to: voiceURL, options: [.atomic])
            logger.info("Downloaded voice '\(sanitized)' (\(data.count / 1024) KB)")
        }

        return try PocketTtsConstantsLoader.loadVoice(voice, from: repoDirectory)
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
            throw PocketTTSError.processingFailed("Failed to locate caches directory")
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
