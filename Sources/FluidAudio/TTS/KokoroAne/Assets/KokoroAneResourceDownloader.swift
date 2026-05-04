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
        variant: KokoroAneVariant = .english,
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsDirectory = try directory ?? defaultModelsDirectory()
        let repo = variant.repo
        let repoDir = modelsDirectory.appendingPathComponent(repo.folderName)

        let required: Set<String>
        switch variant {
        case .english:
            required = ModelNames.KokoroAne.requiredModels
        case .mandarin:
            required = ModelNames.KokoroAne.requiredModelsZh
        }
        let allPresent = required.allSatisfy { name in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(name).path)
        }

        if !allPresent {
            logger.info("Downloading laishere Kokoro models (\(variant.rawValue)) from HuggingFace...")
            try await DownloadUtils.downloadRepo(
                repo,
                to: modelsDirectory,
                progressHandler: progressHandler
            )
        } else {
            logger.info("laishere Kokoro models (\(variant.rawValue)) found in cache at \(repoDir.path)")
        }

        return repoDir
    }

    /// Ensure the Mandarin G2P binary dictionaries (`pinyin_phrases.bin`,
    /// `pinyin_single.bin`) are resident under
    /// `<repoDir>/g2p/`. The compressed `.bin.gz` artefacts are pulled
    /// from `FluidInference/kokoro-82m-v1.1-zh-mlx/g2p/` (the MLX repo
    /// already hosts them — sharing avoids duplicating ~3.6 MB across
    /// HuggingFace bundles), then inflated once on download so runtime
    /// loads stay zero-copy.
    ///
    /// Returns `<repoDir>/g2p/`. Idempotent.
    @discardableResult
    public static func ensureMandarinG2P(
        repoDirectory: URL,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let g2pDir = repoDirectory.appendingPathComponent(KokoroAneConstants.g2pSubdir)
        if !FileManager.default.fileExists(atPath: g2pDir.path) {
            try FileManager.default.createDirectory(
                at: g2pDir, withIntermediateDirectories: true)
        }

        let needed = [
            (
                local: KokoroAneConstants.g2pPinyinPhrasesFile,
                remote: KokoroAneConstants.g2pPinyinPhrasesRemoteFile
            ),
            (
                local: KokoroAneConstants.g2pPinyinSingleFile,
                remote: KokoroAneConstants.g2pPinyinSingleRemoteFile
            ),
        ]

        for entry in needed {
            let localURL = g2pDir.appendingPathComponent(entry.local)
            if FileManager.default.fileExists(atPath: localURL.path) { continue }

            logger.info(
                "Downloading Mandarin G2P asset '\(entry.remote)' from "
                    + "\(KokoroAneConstants.g2pRemoteRepo)/\(KokoroAneConstants.g2pRemoteSubdir)/...")
            let remotePath = "\(KokoroAneConstants.g2pRemoteSubdir)/\(entry.remote)"
            let remoteURL = try ModelRegistry.resolveModel(
                KokoroAneConstants.g2pRemoteRepo, remotePath)
            let compressed = try await AssetDownloader.fetchData(
                from: remoteURL,
                description: "Mandarin G2P asset \(entry.remote)",
                logger: logger
            )
            let inflated: Data
            do {
                inflated = try GzipDecompressor.decompress(compressed)
            } catch {
                throw KokoroAneError.downloadFailed(
                    "Failed to gunzip Mandarin G2P asset \(entry.remote): "
                        + error.localizedDescription)
            }
            try inflated.write(to: localURL, options: [.atomic])
            logger.info(
                "Cached \(entry.local) (\(inflated.count / 1024) KB inflated, "
                    + "\(compressed.count / 1024) KB on the wire)")
        }

        return g2pDir
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
    /// Default voice for each variant is included in `requiredModels(Zh)`; this
    /// helper covers any additional voice that ships separately.
    ///
    /// Mandarin (`ANE-zh/`) voice packs live under a `voices/` subdirectory,
    /// both remotely and on disk. English (`ANE/`) voice packs sit at the
    /// bundle root.
    @discardableResult
    public static func ensureVoicePack(
        _ voice: String,
        repoDirectory: URL,
        variant: KokoroAneVariant = .english
    ) async throws -> URL {
        let sanitized = voice.filter { $0.isLetter || $0.isNumber || $0 == "_" }
        guard !sanitized.isEmpty else {
            throw KokoroAneError.downloadFailed("Invalid voice name: \(voice)")
        }
        let filename = "\(sanitized).bin"
        let relativePath = variant.useVoicesSubdir ? "voices/\(filename)" : filename
        let localURL = repoDirectory.appendingPathComponent(relativePath)

        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }

        // Ensure the parent dir (`voices/`) exists for Mandarin voices that
        // are downloaded individually rather than via the bulk repo grab.
        let parentDir = localURL.deletingLastPathComponent()
        if !FileManager.default.fileExists(atPath: parentDir.path) {
            try FileManager.default.createDirectory(
                at: parentDir, withIntermediateDirectories: true)
        }

        logger.info("Downloading voice pack '\(sanitized)' (\(variant.rawValue)) from HuggingFace...")
        let repo = variant.repo
        let remoteFilePath: String
        if let sub = repo.subPath {
            remoteFilePath = "\(sub)/\(relativePath)"
        } else {
            remoteFilePath = relativePath
        }
        let remoteURL = try ModelRegistry.resolveModel(repo.remotePath, remoteFilePath)
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
