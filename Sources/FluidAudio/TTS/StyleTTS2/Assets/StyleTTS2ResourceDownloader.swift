import Foundation

/// Downloads the StyleTTS2 *shared assets* (vocab, bundle config, and
/// `voices/*.bin` reference blobs) from the legacy HuggingFace repo.
///
/// Repo layout (`FluidInference/StyleTTS-2-coreml`):
/// ```
/// /constants/text_cleaner_vocab.json
/// /config.json
/// /voices/ref_s_<id>.bin
/// ```
///
/// The legacy 4-graph CoreML bundles (`compiled/styletts2_*.mlmodelc`) are
/// no longer fetched — the StyleTTS2-ANE 7-graph re-cut is the sole shipping
/// backend. The vocab + config + voices live in the same repo because they
/// are shared with the ANE checkpoint (same LibriTTS espeak-ng tokenizer,
/// same `ref_s.bin` style blobs).
public enum StyleTTS2ResourceDownloader {

    private static let logger = AppLogger(category: "StyleTTS2ResourceDownloader")

    /// Ensure the shared StyleTTS2 vocab, config, and voice presets are
    /// downloaded.
    ///
    /// - Parameters:
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    ///   - progressHandler: Optional callback for download progress updates.
    /// - Returns: The repo root directory containing `config.json`,
    ///   `constants/text_cleaner_vocab.json`, and the `voices/` directory.
    public static func ensureModels(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            StyleTTS2Constants.defaultModelsSubdirectory)
        let repoDir = modelsDirectory.appendingPathComponent(Repo.styleTts2.folderName)

        let allPresent = ModelNames.StyleTTS2.requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(model).path)
        }

        guard !allPresent else {
            logger.info("StyleTTS2 shared assets found in cache")
            return repoDir
        }

        try FileManager.default.createDirectory(
            at: modelsDirectory, withIntermediateDirectories: true)

        logger.info("Downloading StyleTTS2 shared assets from HuggingFace...")
        try await DownloadUtils.downloadRepo(
            .styleTts2,
            to: modelsDirectory,
            progressHandler: progressHandler
        )

        // Verify after download — `downloadRepo` checks each pattern but the
        // explicit re-check surfaces clearer errors when the network ack races
        // ahead of disk visibility (seen on slow filesystems).
        for model in ModelNames.StyleTTS2.requiredModels {
            let path = repoDir.appendingPathComponent(model).path
            guard FileManager.default.fileExists(atPath: path) else {
                throw StyleTTS2Error.downloadFailed(
                    "Missing required asset after download: \(model)")
            }
        }

        // The `voices/` directory existence check above is necessary but not
        // sufficient — the directory walker may have created the parent
        // without populating its contents. Verify each preset blob lands.
        let voicesDir = repoDir.appendingPathComponent(
            StyleTTS2VoicePresets.directoryName, isDirectory: true)
        for filename in StyleTTS2VoicePresets.requiredFilenames {
            let path = voicesDir.appendingPathComponent(filename).path
            guard FileManager.default.fileExists(atPath: path) else {
                throw StyleTTS2Error.downloadFailed(
                    "Missing voice preset after download: \(filename)")
            }
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
            throw StyleTTS2Error.processingFailed("Failed to locate caches directory")
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
