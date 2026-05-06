import Foundation

/// Downloads the StyleTTS2-ANE 7-stage CoreML chain from HuggingFace.
///
/// Repo layout (`FluidInference/StyleTTS-2-coreml/ANE/`):
/// ```
/// /styletts2_ane_plbert.mlmodelc/
/// /styletts2_ane_postbert_v2.mlmodelc/
/// /styletts2_ane_alignment.mlmodelc/
/// /styletts2_ane_diffusion_step.mlmodelc/
/// /styletts2_ane_prosody.mlmodelc/
/// /styletts2_ane_noise.mlmodelc/
/// /styletts2_ane_vocoder_v2.mlmodelc/
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

        // Re-check after download surfaces clearer errors when the network ack
        // races ahead of disk visibility (seen on slow filesystems).
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
        if !FileManager.default.fileExists(atPath: cacheDirectory.path) {
            try FileManager.default.createDirectory(
                at: cacheDirectory, withIntermediateDirectories: true)
        }
        return cacheDirectory.appendingPathComponent(
            StyleTTS2Constants.defaultModelsSubdirectory)
    }
}
