import Foundation

/// Downloads VoxCPM models and constants from HuggingFace.
public enum VoxCpmResourceDownloader {

    private static let logger = AppLogger(category: "VoxCpmResourceDownloader")

    /// Ensure all VoxCPM models are downloaded and return the cache directory.
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
            VoxCpmConstants.defaultModelsSubdirectory)

        let repoDir = modelsDirectory.appendingPathComponent(Repo.voxCpm.folderName)

        // Check that all required directories/files exist
        let requiredModels = ModelNames.VoxCPM.requiredModels
        let allPresent = requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(model).path)
        }

        if !allPresent {
            logger.info("Downloading VoxCPM models from HuggingFace...")
            try await DownloadUtils.downloadRepo(
                .voxCpm, to: modelsDirectory, progressHandler: progressHandler)
        } else {
            logger.info("VoxCPM models found in cache")
        }

        return repoDir
    }

    /// Ensure constants (binary blobs + tokenizer) are available.
    public static func ensureConstants(repoDirectory: URL) async throws -> VoxCpmConstantsBundle {
        try await VoxCpmConstantsLoader.load(from: repoDirectory)
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
            throw VoxCpmError.processingFailed("Failed to locate caches directory")
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
