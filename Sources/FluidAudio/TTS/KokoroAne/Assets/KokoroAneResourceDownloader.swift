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
        progressHandler: ProgressHandler? = nil
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
        case .japanese:
            required = ModelNames.KokoroAne.requiredModelsJa
        }
        let allPresent = required.allSatisfy { name in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(name).path)
        }

        if !allPresent {
            logger.info("Downloading laishere Kokoro models (\(variant.rawValue)) from HuggingFace...")
            try await ModelHub.download(
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
    /// `pinyin_single.bin`) are resident under `<repoDir>/g2p/`. The
    /// uncompressed `.bin` artefacts are pulled from
    /// `FluidInference/kokoro-82m-coreml/ANE-zh/assets/` (co-located with
    /// the CoreML weights so the Mandarin variant has a single HF
    /// dependency).
    ///
    /// Returns `<repoDir>/g2p/`. Idempotent.
    @discardableResult
    public static func ensureMandarinG2P(
        repoDirectory: URL,
        progressHandler: ProgressHandler? = nil
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
            let data = try await AssetDownloader.fetchData(
                from: remoteURL,
                description: "Mandarin G2P asset \(entry.remote)",
                logger: logger
            )
            try data.write(to: localURL, options: [.atomic])
            logger.info("Cached \(entry.local) (\(data.count / 1024) KB)")
        }

        return g2pDir
    }

    /// Best-effort fetch of the jieba HMM tables (start / trans / emit)
    /// into the same `<repoDir>/g2p/` cache.
    ///
    /// Returns the cache directory when all three artefacts are
    /// resident locally (either pre-cached or freshly downloaded).
    /// Returns `nil` when any artefact is missing both locally and
    /// remotely — the caller is expected to fall back to the
    /// FMM/single-char-only segmentation path. The Mandarin variant
    /// stays usable in that case; HMM is a quality booster, not a
    /// hard dependency.
    public static func ensureMandarinJiebaHmm(
        repoDirectory: URL
    ) async -> URL? {
        let g2pDir = repoDirectory.appendingPathComponent(KokoroAneConstants.g2pSubdir)
        if !FileManager.default.fileExists(atPath: g2pDir.path) {
            do {
                try FileManager.default.createDirectory(
                    at: g2pDir, withIntermediateDirectories: true)
            } catch {
                logger.warning(
                    "Could not create jieba HMM cache dir: \(error.localizedDescription)")
                return nil
            }
        }

        let needed = [
            (
                local: KokoroAneConstants.jiebaHmmStartFile,
                remote: KokoroAneConstants.jiebaHmmStartRemoteFile
            ),
            (
                local: KokoroAneConstants.jiebaHmmTransFile,
                remote: KokoroAneConstants.jiebaHmmTransRemoteFile
            ),
            (
                local: KokoroAneConstants.jiebaHmmEmitFile,
                remote: KokoroAneConstants.jiebaHmmEmitRemoteFile
            ),
        ]

        for entry in needed {
            let localURL = g2pDir.appendingPathComponent(entry.local)
            if FileManager.default.fileExists(atPath: localURL.path) { continue }
            do {
                logger.info(
                    "Downloading jieba HMM asset '\(entry.remote)' from "
                        + "\(KokoroAneConstants.g2pRemoteRepo)/\(KokoroAneConstants.g2pRemoteSubdir)/...")
                let remotePath = "\(KokoroAneConstants.g2pRemoteSubdir)/\(entry.remote)"
                let remoteURL = try ModelRegistry.resolveModel(
                    KokoroAneConstants.g2pRemoteRepo, remotePath)
                let data = try await AssetDownloader.fetchData(
                    from: remoteURL,
                    description: "jieba HMM asset \(entry.remote)",
                    logger: logger
                )
                try data.write(to: localURL, options: [.atomic])
                logger.info("Cached \(entry.local) (\(data.count / 1024) KB)")
            } catch {
                logger.warning(
                    "Jieba HMM asset '\(entry.remote)' unavailable "
                        + "(\(error.localizedDescription)); HMM segmentation disabled.")
                return nil
            }
        }
        return g2pDir
    }

    /// Ensure the Mandarin g2pW polyphone disambiguator assets are
    /// resident under `<repoDir>/g2pw/`. Returns the directory URL on
    /// success, or `nil` if any required artefact is unavailable
    /// (network failure, asset not yet published, …) so callers can
    /// fall back to the dict-only Mandarin pipeline without throwing.
    ///
    /// The CoreML bundle (`g2pw.mlmodelc/`) is a directory and is
    /// expected to land via the bulk `ensureModels` repo grab once the
    /// asset is added to the `requiredModelsZh` set. This helper only
    /// fetches the two auxiliary text files (`vocab.txt`,
    /// `POLYPHONIC_CHARS.txt`) that ship alongside the model and then
    /// validates the bundle is on disk.
    @discardableResult
    public static func ensureMandarinG2pw(
        repoDirectory: URL
    ) async -> URL? {
        let g2pwDir = repoDirectory.appendingPathComponent(KokoroAneConstants.g2pwSubdir)
        if !FileManager.default.fileExists(atPath: g2pwDir.path) {
            do {
                try FileManager.default.createDirectory(
                    at: g2pwDir, withIntermediateDirectories: true)
            } catch {
                logger.info(
                    "g2pW assets unavailable (could not create cache dir: \(error.localizedDescription))"
                )
                return nil
            }
        }

        let needed = [
            (
                local: KokoroAneConstants.g2pwVocabFile,
                remote: KokoroAneConstants.g2pwVocabRemoteFile
            ),
            (
                local: KokoroAneConstants.g2pwPolyphonicCharsFile,
                remote: KokoroAneConstants.g2pwPolyphonicCharsRemoteFile
            ),
        ]

        for entry in needed {
            let localURL = g2pwDir.appendingPathComponent(entry.local)
            if FileManager.default.fileExists(atPath: localURL.path) { continue }

            do {
                let remotePath = "\(KokoroAneConstants.g2pwRemoteSubdir)/\(entry.remote)"
                let remoteURL = try ModelRegistry.resolveModel(
                    KokoroAneConstants.g2pRemoteRepo, remotePath)
                let data = try await AssetDownloader.fetchData(
                    from: remoteURL,
                    description: "Mandarin g2pW asset \(entry.remote)",
                    logger: logger
                )
                try data.write(to: localURL, options: [.atomic])
                logger.info("Cached \(entry.local) (\(data.count / 1024) KB)")
            } catch {
                logger.info(
                    "g2pW asset '\(entry.local)' unavailable (\(error.localizedDescription))"
                        + " — Mandarin G2P will run dict-only"
                )
                return nil
            }
        }

        // The CoreML bundle is required for the model to actually run.
        // Without it, return nil and let the caller skip g2pW entirely.
        let modelURL =
            repoDirectory
            .appendingPathComponent(KokoroAneConstants.g2pwSubdir)
            .appendingPathComponent(KokoroAneConstants.g2pwModelBundle)
        if !FileManager.default.fileExists(atPath: modelURL.path) {
            logger.info(
                "g2pW CoreML bundle missing at \(modelURL.path) — Mandarin G2P will run dict-only"
            )
            return nil
        }
        return g2pwDir
    }

    /// Ensure the shared G2P CoreML assets (encoder + decoder + vocab) exist
    /// in the kokoro cache directory. KokoroAne reuses `G2PModel` for text →
    /// IPA conversion, and `G2PModel.loadIfNeeded` only reads from cache —
    /// it never downloads. Without this call, a first-time KokoroAne user
    /// (who has never run the regular kokoro backend) would fail with
    /// `G2PModelError.vocabLoadFailed`.
    public static func ensureG2PAssets(
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
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
        try await ModelHub.download(
            .kokoro,
            to: modelsDirectory,
            variant: "g2p-only",
            progressHandler: progressHandler
        )
    }

    /// Best-effort fetch of Kokoro's preprocessed Misaki lexicon cache
    /// (`us_lexicon_cache.json`) into the shared kokoro cache directory
    /// (next to the G2P CoreML assets — same file StyleTTS2 consumes via
    /// `LexiconAssetCache`).
    ///
    /// Returns the kokoro cache directory when the file is resident
    /// (pre-cached or freshly downloaded), or `nil` when it is missing
    /// and could not be fetched — the English frontend then falls back
    /// to BART-G2P-only phonemization. The lexicon is a pronunciation
    /// quality booster (Misaki weak forms for function words, issue
    /// #691), not a hard dependency.
    public static func ensureEnglishLexicon(
        directory: URL? = nil
    ) async -> URL? {
        let filename = "us_lexicon_cache.json"
        do {
            let modelsDirectory = try directory ?? defaultModelsDirectory()
            let kokoroDir = modelsDirectory.appendingPathComponent(Repo.kokoro.folderName)
            try FileManager.default.createDirectory(
                at: kokoroDir, withIntermediateDirectories: true)

            let localURL = kokoroDir.appendingPathComponent(filename)
            if FileManager.default.fileExists(atPath: localURL.path) {
                return kokoroDir
            }

            let remoteURL = try ModelRegistry.resolveModel(Repo.kokoro.remotePath, filename)
            let descriptor = AssetDownloader.Descriptor(
                description: filename,
                remoteURL: remoteURL,
                destinationURL: localURL
            )
            _ = try await AssetDownloader.ensure(descriptor, logger: logger)
            return kokoroDir
        } catch {
            logger.warning(
                "English lexicon cache unavailable (\(error.localizedDescription)) — "
                    + "falling back to BART G2P only")
            return nil
        }
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
        // Delegate to the shared TTS cache root (Application Support on iOS,
        // ~/.cache/fluidaudio on macOS) so all backends share one location.
        return try TtsCacheDirectory.ensure().appendingPathComponent(modelsSubdirectory)
    }
}
