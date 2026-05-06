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
public enum StyleTTS2AssetDownloader {

    private static let logger = AppLogger(category: "StyleTTS2AssetDownloader")

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
        let repoDir = modelsDirectory.appendingPathComponent(Repo.styleTts2Assets.folderName)

        // `ModelNames.StyleTTS2.requiredModels` lists files plus the `voices/`
        // *directory*; that list is the allowlist passed to `downloadRepo`, not
        // a sufficient cache check on its own (an empty `voices/` dir would
        // make `fileExists` return true). We separately enumerate
        // `StyleTTS2VoicePresets.requiredFilenames` to validate the actual
        // preset blobs.
        //
        // Atomic rename in `DownloadUtils` rules out 0-byte stragglers from
        // interrupted writes, so `fileExists` is sufficient — every file that
        // exists at this path was fully written.
        let voicesDir = repoDir.appendingPathComponent(
            StyleTTS2VoicePresets.directoryName, isDirectory: true)
        let sharedPresent = ModelNames.StyleTTS2.requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(model).path)
        }
        let voicesPresent = StyleTTS2VoicePresets.requiredFilenames.allSatisfy { filename in
            FileManager.default.fileExists(
                atPath: voicesDir.appendingPathComponent(filename).path)
        }

        if sharedPresent && voicesPresent {
            logger.info("StyleTTS2 shared assets found in cache")
            return repoDir
        }

        try FileManager.default.createDirectory(
            at: modelsDirectory, withIntermediateDirectories: true)

        logger.info("Downloading StyleTTS2 shared assets from HuggingFace...")
        try await DownloadUtils.downloadRepo(
            .styleTts2Assets,
            to: modelsDirectory,
            progressHandler: progressHandler
        )

        // Defense-in-depth verification. `downloadRepo` already checks its own
        // glob/allowlist, but if that allowlist ever drifts from the
        // `requiredModels` list this loop pinpoints the missing path with a
        // clearer error than the consumer would otherwise see.
        for model in ModelNames.StyleTTS2.requiredModels {
            let path = repoDir.appendingPathComponent(model).path
            guard FileManager.default.fileExists(atPath: path) else {
                throw StyleTTS2Error.downloadFailed(
                    "Missing required asset after download: \(model)")
            }
        }

        // Same defense-in-depth for voice presets — the `voices/` allowlist
        // entry only covers the directory, not its contents.
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

    /// Resolve the FluidAudio cache root. macOS uses `~/.cache/fluidaudio`
    /// (Linux/XDG-style — matches HuggingFace's default and is the
    /// project-wide convention used by every other downloader). iOS falls
    /// back to the platform Caches directory.
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
            throw StyleTTS2Error.downloadFailed("Failed to locate caches directory")
        }
        baseDirectory = first
        #endif

        let cacheDirectory = baseDirectory.appendingPathComponent("fluidaudio")
        try FileManager.default.createDirectory(
            at: cacheDirectory, withIntermediateDirectories: true)
        return cacheDirectory
    }
}
