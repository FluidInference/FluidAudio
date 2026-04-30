import Foundation

/// Downloads StyleTTS2 CoreML models, vocab, and bundle config from HuggingFace.
///
/// Repo layout (`FluidInference/StyleTTS-2-coreml`):
/// ```
/// /compiled/styletts2_text_predictor_{32,64,128,256,512}.mlmodelc/
/// /compiled/styletts2_diffusion_step_512.mlmodelc/
/// /compiled/styletts2_f0n_energy.mlmodelc/
/// /compiled/styletts2_decoder_{256,512,1024,2048,4096}.mlmodelc/
/// /constants/text_cleaner_vocab.json
/// /config.json
/// /styletts2_*.mlpackage/      (portability/debugging artifacts; not fetched here)
/// ```
/// We grab only the precompiled `compiled/*.mlmodelc` bundles to avoid the
/// cold-start `anecompilerservice` hit on first synthesis.
public enum StyleTTS2ResourceDownloader {

    private static let logger = AppLogger(category: "StyleTTS2ResourceDownloader")

    /// Ensure all StyleTTS2 models, vocab, and config are downloaded.
    ///
    /// - Parameters:
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    ///   - progressHandler: Optional callback for download progress updates.
    /// - Returns: The repo root directory containing all `.mlpackage` bundles
    ///   plus `config.json` and `constants/text_cleaner_vocab.json`.
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
            logger.info("StyleTTS2 models found in cache")
            return repoDir
        }

        try FileManager.default.createDirectory(
            at: modelsDirectory, withIntermediateDirectories: true)

        logger.info("Downloading StyleTTS2 bundle from HuggingFace...")
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
