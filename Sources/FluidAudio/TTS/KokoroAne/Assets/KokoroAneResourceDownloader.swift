import Foundation

/// Downloads the laishere/kokoro 7-stage CoreML chain + auxiliary files
/// (`vocab.json`, voice packs) from HuggingFace.
public enum KokoroAneResourceDownloader {

    private static let logger = AppLogger(category: "KokoroAneResourceDownloader")

    /// Default cache subdirectory under the platform cache root.
    /// Resolves to `~/.cache/fluidaudio/Models/` on macOS,
    /// `<App caches>/fluidaudio/Models/` on iOS.
    public static let modelsSubdirectory = "Models"

    /// Ensure all required mlmodelc + vocab + default voice files are present.
    /// Returns the repo directory containing them.
    @discardableResult
    public static func ensureModels(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsDirectory = try directory ?? defaultModelsDirectory()
        let repoDir = modelsDirectory.appendingPathComponent(Repo.kokoroAne.folderName)

        let required = ModelNames.KokoroAne.requiredModels
        let allPresent = required.allSatisfy { name in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(name).path)
        }

        if !allPresent {
            logger.info("Downloading laishere Kokoro models from HuggingFace...")
            try await DownloadUtils.downloadRepo(
                .kokoroAne,
                to: modelsDirectory,
                progressHandler: progressHandler
            )
        } else {
            logger.info("laishere Kokoro models found in cache at \(repoDir.path)")
        }

        return repoDir
    }

    /// Ensure the shared G2P CoreML assets (encoder + decoder + vocab) exist
    /// in the kokoro cache directory. KokoroAne reuses `G2PModel` for text →
    /// IPA conversion, and `G2PModel.loadIfNeeded` only reads from cache —
    /// it never downloads. Without this call, a first-time KokoroAne user
    /// (who has never run the regular kokoro backend) would fail with
    /// `G2PModelError.vocabLoadFailed`.
    public static func ensureG2PAssets(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        let modelsDirectory = try directory ?? defaultModelsDirectory()
        let kokoroDir = modelsDirectory.appendingPathComponent(Repo.kokoro.folderName)
        let allPresent = ModelNames.G2P.requiredModels.allSatisfy { name in
            FileManager.default.fileExists(atPath: kokoroDir.appendingPathComponent(name).path)
        }
        if allPresent {
            return
        }
        logger.info("Downloading shared kokoro G2P assets from HuggingFace...")
        try await DownloadUtils.downloadRepo(
            .kokoro,
            to: modelsDirectory,
            variant: "g2p-only",
            progressHandler: progressHandler
        )
    }

    /// Ensure a specific voice pack `.bin` file exists, downloading if missing.
    /// Default voice (`af_heart.bin`) is included in `requiredModels`; this
    /// helper covers any additional voice that ships separately.
    @discardableResult
    public static func ensureVoicePack(
        _ voice: String,
        repoDirectory: URL
    ) async throws -> URL {
        let sanitized = voice.filter { $0.isLetter || $0.isNumber || $0 == "_" }
        guard !sanitized.isEmpty else {
            throw KokoroAneError.downloadFailed("Invalid voice name: \(voice)")
        }
        let filename = "\(sanitized).bin"
        let localURL = repoDirectory.appendingPathComponent(filename)

        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        logger.info("Downloading voice pack '\(sanitized)' from HuggingFace...")
        let remoteFilePath: String
        if let sub = Repo.kokoroAne.subPath {
            remoteFilePath = "\(sub)/\(filename)"
        } else {
            remoteFilePath = filename
        }
        let remoteURL = try ModelRegistry.resolveModel(Repo.kokoroAne.remotePath, remoteFilePath)
        let data = try await AssetDownloader.fetchData(
            from: remoteURL,
            description: "\(sanitized) voice pack",
            logger: logger
        )
        try data.write(to: localURL, options: [.atomic])
        logger.info("Downloaded voice pack '\(sanitized)' (\(data.count / 1024) KB)")
        return localURL
    }

    // MARK: - Private

    private static func defaultModelsDirectory() throws -> URL {
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
            throw KokoroAneError.downloadFailed("Failed to locate caches directory")
        }
        baseDirectory = first
        #endif

        let cacheDirectory = baseDirectory.appendingPathComponent("fluidaudio")
        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
        }
        return cacheDirectory.appendingPathComponent(modelsSubdirectory)
    }
}
