import Foundation

/// Downloads the StyleTTS2-ANE 7-stage CoreML chain from HuggingFace.
///
/// Repo layout (`FluidInference/StyleTTS-2-coreml/ANE/`):
/// ```
/// /styletts2_ane_plbert.mlmodelc/
/// /styletts2_ane_postbert.mlmodelc/
/// /styletts2_ane_alignment.mlmodelc/
/// /styletts2_ane_diffusion_step.mlmodelc/
/// /styletts2_ane_prosody.mlmodelc/
/// /styletts2_ane_noise.mlmodelc/
/// /styletts2_ane_vocoder.mlmodelc/
/// ```
///
/// `ref_s.bin` voice blobs are *not* fetched here — they ship per-voice and
/// are loaded via `StyleTTS2VoiceStyle.load(from:)` from a caller-supplied
/// URL. The text-cleaner vocab + tokenizer state are reused from the legacy
/// 4-graph repo (already downloaded by `StyleTTS2AssetDownloader`).
public enum StyleTTS2CoreMLDownloader {

    private static let logger = AppLogger(category: "StyleTTS2CoreMLDownloader")

    /// Ensure the 7 mlmodelcs are present locally; download if any is missing.
    /// - Returns: The directory containing all 7 `.mlmodelc` bundles.
    @discardableResult
    public static func ensureAssets(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsDirectory = try directory ?? defaultModelsDirectory()
        let repo = Repo.styleTts2
        let repoDir = modelsDirectory.appendingPathComponent(repo.folderName)

        let required = ModelNames.StyleTTS2.requiredCoreMLModels
        let allPresent = required.allSatisfy { name in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(name).path)
        }

        if allPresent {
            logger.info("StyleTTS2-ANE models found in cache at \(repoDir.path)")
            return repoDir
        }

        try FileManager.default.createDirectory(
            at: modelsDirectory, withIntermediateDirectories: true)

        logger.info("Downloading StyleTTS2-ANE bundle from HuggingFace...")
        try await DownloadUtils.downloadRepo(
            repo,
            to: modelsDirectory,
            progressHandler: progressHandler
        )

        // Defense-in-depth verification. `downloadRepo` already enforces its
        // own glob/allowlist, but if that allowlist ever drifts from
        // `requiredCoreMLModels` this loop pinpoints the missing bundle with
        // a clearer error than the consumer would otherwise see.
        for name in required {
            let path = repoDir.appendingPathComponent(name).path
            guard FileManager.default.fileExists(atPath: path) else {
                throw StyleTTS2Error.downloadFailed(
                    "Missing required asset after download: \(name)")
            }
        }
        return repoDir
    }

    // MARK: - Private

    /// Resolve the FluidAudio cache root and append the StyleTTS2 models
    /// subdirectory.
    ///
    /// Returns the *fully qualified* models path (`~/.cache/fluidaudio/<sub>`),
    /// while `StyleTTS2AssetDownloader.cacheDirectory()` returns just the base
    /// (`~/.cache/fluidaudio`) and lets its caller append the subdir. Both
    /// shapes coexist intentionally — the asset downloader owns multiple
    /// repos under the same base, while this downloader maps 1:1 to a single
    /// model bundle. Pulling these two helpers (plus the analogous ones in
    /// every other downloader under `Sources/FluidAudio/TTS/`) into a single
    /// shared cache-path utility is a worthwhile project-wide refactor but
    /// out of scope here.
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
            throw StyleTTS2Error.downloadFailed("Failed to locate caches directory")
        }
        baseDirectory = first
        #endif

        let cacheDirectory = baseDirectory.appendingPathComponent("fluidaudio")
        try FileManager.default.createDirectory(
            at: cacheDirectory, withIntermediateDirectories: true)
        return cacheDirectory.appendingPathComponent(
            StyleTTS2Constants.defaultModelsSubdirectory)
    }
}
