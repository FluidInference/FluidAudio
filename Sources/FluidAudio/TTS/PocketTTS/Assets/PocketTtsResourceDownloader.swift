import Foundation
import OSLog

/// Downloads PocketTTS models and constants from HuggingFace.
public enum PocketTtsResourceDownloader {

    private static let logger = AppLogger(category: "PocketTtsResourceDownloader")

    /// Ensure all PocketTTS models are downloaded and return the cache directory.
    ///
    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public static func ensureModels(directory: URL? = nil) async throws -> URL {
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
            try await DownloadUtils.downloadRepo(.pocketTts, to: modelsDirectory)
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
        let modelName = ModelNames.PocketTTS.mimiEncoderFile
        let modelDir = repoDir.appendingPathComponent(modelName)

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        // List files in the mimi_encoder.mlmodelc directory
        let apiPath = "tree/main/\(modelName)"
        let dirURL = try ModelRegistry.apiModels(Repo.pocketTts.remotePath, apiPath)

        let (dirData, _) = try await DownloadUtils.fetchWithAuth(from: dirURL)

        guard let items = try JSONSerialization.jsonObject(with: dirData) as? [[String: Any]] else {
            throw PocketTTSError.downloadFailed("Failed to list Mimi encoder files")
        }

        // Collect all files recursively
        var filesToDownload: [(path: String, size: Int)] = []

        func collectFiles(from items: [[String: Any]], basePath: String) async throws {
            for item in items {
                guard let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String
                else { continue }

                if itemType == "directory" {
                    let subDirURL = try ModelRegistry.apiModels(Repo.pocketTts.remotePath, "tree/main/\(itemPath)")
                    let (subDirData, _) = try await DownloadUtils.fetchWithAuth(from: subDirURL)
                    if let subItems = try JSONSerialization.jsonObject(with: subDirData) as? [[String: Any]] {
                        try await collectFiles(from: subItems, basePath: itemPath)
                    }
                } else if itemType == "file" {
                    let fileSize = item["size"] as? Int ?? -1
                    filesToDownload.append((path: itemPath, size: fileSize))
                }
            }
        }

        try await collectFiles(from: items, basePath: modelName)
        logger.info("Found \(filesToDownload.count) files in Mimi encoder")

        // Download each file
        for (index, file) in filesToDownload.enumerated() {
            // Local path relative to modelName
            let relativePath =
                file.path.hasPrefix("\(modelName)/")
                ? String(file.path.dropFirst(modelName.count + 1))
                : file.path
            let destPath = modelDir.appendingPathComponent(relativePath)

            if FileManager.default.fileExists(atPath: destPath.path) {
                continue
            }

            try FileManager.default.createDirectory(
                at: destPath.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            // Handle empty files
            if file.size == 0 {
                FileManager.default.createFile(atPath: destPath.path, contents: Data())
                continue
            }

            let encodedPath = file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            let fileURL = try ModelRegistry.resolveModel(Repo.pocketTts.remotePath, encodedPath)

            let (tempURL, response) = try await DownloadUtils.sharedSession.download(
                for: URLRequest(url: fileURL, timeoutInterval: 1800)
            )

            guard let httpResponse = response as? HTTPURLResponse,
                (200..<300).contains(httpResponse.statusCode)
            else {
                throw PocketTTSError.downloadFailed("Failed to download \(file.path)")
            }

            if FileManager.default.fileExists(atPath: destPath.path) {
                try? FileManager.default.removeItem(at: destPath)
            }
            try FileManager.default.moveItem(at: tempURL, to: destPath)

            if (index + 1) % 5 == 0 || index == filesToDownload.count - 1 {
                logger.info("Downloaded \(index + 1)/\(filesToDownload.count) Mimi encoder files")
            }
        }

        logger.info("Mimi encoder download complete")
    }

    /// Ensure constants (binary blobs + tokenizer) are available.
    public static func ensureConstants(repoDirectory: URL) throws -> PocketTtsConstantsBundle {
        try PocketTtsConstantsLoader.load(from: repoDirectory)
    }

    /// Ensure voice conditioning data is available.
    public static func ensureVoice(
        _ voice: String, repoDirectory: URL
    ) throws -> PocketTtsVoiceData {
        try PocketTtsConstantsLoader.loadVoice(voice, from: repoDirectory)
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
