import Foundation
import OSLog

/// Downloads PocketTTS models and constants from HuggingFace.
public enum PocketTtsResourceDownloader {

    private static let logger = AppLogger(category: "PocketTtsResourceDownloader")

    /// Ensure all PocketTTS models for the given language are downloaded and
    /// return the **language root** directory (`<repoDir>/v2/<lang>/`).
    ///
    /// - Parameters:
    ///   - language: Which upstream language pack to fetch.
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    ///   - progressHandler: Optional callback for download progress updates.
    /// - Returns: The directory that contains the four `.mlmodelc` packages
    ///   plus `constants_bin/` for the requested language.
    public static func ensureModels(
        language: PocketTtsLanguage = .english,
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            PocketTtsConstants.defaultModelsSubdirectory)

        let repoDir = modelsDirectory.appendingPathComponent(Repo.pocketTts.folderName)
        let subdir = language.repoSubdirectory
        let languageRoot = repoDir.appendingPathComponent(subdir)

        let allPresent = ModelNames.PocketTTS.requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: languageRoot.appendingPathComponent(model).path)
        }

        if !allPresent {
            logger.info(
                "Downloading PocketTTS \(language.rawValue) language pack from HuggingFace (\(subdir))..."
            )
            try await DownloadUtils.downloadSubdirectory(
                .pocketTts,
                subdirectory: subdir,
                to: repoDir,
                progressHandler: progressHandler
            )
        } else {
            logger.info(
                "PocketTTS \(language.rawValue) models found in cache")
        }

        return languageRoot
    }

    /// Ensure the Mimi encoder model is downloaded for voice cloning.
    ///
    /// This is an optional model that's only needed for voice cloning functionality.
    /// It's downloaded separately from the main models to reduce initial download size.
    /// The encoder is shared across all language packs and lives at the legacy
    /// repo root regardless of which language is currently loaded — so a Spanish
    /// (or any non-English) user can clone a voice without pulling in the
    /// English language pack.
    /// - Parameter directory: Optional override for the base cache directory.
    ///   When `nil`, uses the default platform cache location.
    public static func ensureMimiEncoder(directory: URL? = nil) async throws -> URL {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            PocketTtsConstants.defaultModelsSubdirectory)
        let repoDir = modelsDirectory.appendingPathComponent(Repo.pocketTts.folderName)
        let encoderPath = repoDir.appendingPathComponent(ModelNames.PocketTTS.mimiEncoderFile)

        if FileManager.default.fileExists(atPath: encoderPath.path) {
            logger.info("Mimi encoder found in cache")
            return encoderPath
        }

        // Make sure the parent directory exists — the user may not have
        // downloaded any language pack yet.
        try FileManager.default.createDirectory(
            at: repoDir, withIntermediateDirectories: true)

        logger.info("Downloading Mimi encoder for voice cloning...")
        try await downloadMimiEncoder(to: repoDir)

        guard FileManager.default.fileExists(atPath: encoderPath.path) else {
            throw PocketTTSError.downloadFailed("Failed to download Mimi encoder model")
        }

        return encoderPath
    }

    /// Download the Mimi encoder model files from HuggingFace.
    private static func downloadMimiEncoder(to repoDir: URL) async throws {
        try await DownloadUtils.downloadSubdirectory(
            .pocketTts,
            subdirectory: ModelNames.PocketTTS.mimiEncoderFile,
            to: repoDir
        )
    }

    /// Ensure constants (binary blobs + tokenizer) are available.
    ///
    /// - Parameter languageRoot: The directory returned by `ensureModels(...)`,
    ///   which contains the language-specific `constants_bin/`.
    public static func ensureConstants(languageRoot: URL) throws -> PocketTtsConstantsBundle {
        try PocketTtsConstantsLoader.load(from: languageRoot)
    }

    /// Ensure voice conditioning data for the given language is available,
    /// downloading from HuggingFace if missing.
    ///
    /// - Parameters:
    ///   - voice: Voice name (e.g. `"alba"`, `"michael"`).
    ///   - language: Language pack the voice belongs to. Voice files are
    ///     per-language (same names, different acoustic embeddings).
    ///   - languageRoot: The directory returned by `ensureModels(language:)`.
    public static func ensureVoice(
        _ voice: String,
        language: PocketTtsLanguage = .english,
        languageRoot: URL
    ) async throws -> PocketTtsVoiceData {
        let sanitized = voice.filter { $0.isLetter || $0.isNumber || $0 == "_" }
        guard !sanitized.isEmpty else {
            throw PocketTTSError.processingFailed("Invalid voice name: \(voice)")
        }
        let constantsDir = languageRoot.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let safetensorsFile = "\(sanitized).safetensors"
        let safetensorsURL = constantsDir.appendingPathComponent(safetensorsFile)

        if !FileManager.default.fileExists(atPath: safetensorsURL.path) {
            let remotePath = "\(language.repoSubdirectory)/constants_bin/\(safetensorsFile)"
            let remoteURL = try ModelRegistry.resolveModel(Repo.pocketTts.remotePath, remotePath)
            logger.info(
                "Downloading voice '\(sanitized)' for \(language.rawValue) from HuggingFace (\(safetensorsFile))..."
            )
            let data = try await AssetDownloader.fetchData(
                from: remoteURL,
                description: "\(sanitized) voice prompt (\(language.rawValue))",
                logger: logger
            )
            // Make sure the parent directory exists in case this is a fresh
            // language pack that hasn't materialized constants_bin/ yet.
            try FileManager.default.createDirectory(
                at: constantsDir, withIntermediateDirectories: true)
            try data.write(to: safetensorsURL, options: [.atomic])
            logger.info("Downloaded voice '\(sanitized)' (\(data.count / 1024) KB)")
        }

        return try PocketTtsConstantsLoader.loadVoice(voice, from: languageRoot)
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
