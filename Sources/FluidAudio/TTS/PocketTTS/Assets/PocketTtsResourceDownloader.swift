import Foundation
import OSLog

/// Resolved on-disk locations for PocketTTS submodels, honoring the requested
/// per-submodel quantization configuration.
public struct PocketTtsResolvedModels: Sendable {
    /// Local cache directory mirroring the HuggingFace repo root.
    public let repoDir: URL
    /// Language root: legacy repo root for English, `<repoDir>/v2/<lang>/` otherwise.
    /// Always contains `constants_bin/` regardless of quantization choices.
    public let languageRoot: URL
    /// Resolved `.mlmodelc` URL for `cond_step` (fp16 or int8).
    public let condStepURL: URL
    /// Resolved `.mlmodelc` URL for `flowlm_step` (fp16 or int8).
    public let flowlmStepURL: URL
    /// Resolved `.mlmodelc` URL for `flow_decoder` (fp16 or int8).
    public let flowDecoderURL: URL
    /// Resolved `.mlmodelc` URL for `mimi_decoder` (fp16 or int8).
    public let mimiDecoderURL: URL
    /// Quantization that was applied to resolve the URLs above.
    public let quantization: PocketTtsQuantization
}

/// Downloads PocketTTS models and constants from HuggingFace.
public enum PocketTtsResourceDownloader {

    private static let logger = AppLogger(category: "PocketTtsResourceDownloader")

    /// Ensure all PocketTTS models for the given language are downloaded and
    /// return the **language root** directory.
    ///
    /// Backwards-compatible overload: defaults to all-fp16 quantization and
    /// returns just the language root. New callers should prefer
    /// ``ensureResolvedModels(language:quantization:directory:progressHandler:)``.
    ///
    /// - Parameters:
    ///   - language: Which upstream language pack to fetch.
    ///   - directory: Optional override for the base cache directory.
    ///     When `nil`, uses the default platform cache location.
    ///   - progressHandler: Optional callback for download progress updates.
    /// - Returns: The directory that contains the four `.mlmodelc` packages
    ///   plus `constants_bin/` for the requested language. For English this
    ///   is the legacy repo root; for other languages it's
    ///   `<repoDir>/v2/<lang>/`.
    public static func ensureModels(
        language: PocketTtsLanguage = .english,
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let resolved = try await ensureResolvedModels(
            language: language,
            quantization: .allFp16,
            directory: directory,
            progressHandler: progressHandler
        )
        return resolved.languageRoot
    }

    /// Ensure all PocketTTS submodels and constants are downloaded for the
    /// given language and quantization configuration, and return resolved
    /// per-submodel URLs.
    ///
    /// The fp16 language pack (and its `constants_bin/`) is always ensured,
    /// since constants and tokenizer live there. For each submodel marked
    /// `int8`, the corresponding subtree under
    /// `languages/<lang>/int8/<basename>.mlmodelc` is fetched on top.
    public static func ensureResolvedModels(
        language: PocketTtsLanguage = .english,
        quantization: PocketTtsQuantization = .allFp16,
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> PocketTtsResolvedModels {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            PocketTtsConstants.defaultModelsSubdirectory)

        let repoDir = modelsDirectory.appendingPathComponent(Repo.pocketTts.folderName)

        let languageRoot: URL
        if let subdir = language.repoSubdirectory {
            languageRoot = repoDir.appendingPathComponent(subdir)
        } else {
            languageRoot = repoDir
        }

        // Always ensure the fp16 language pack — constants_bin/ lives there
        // and is needed regardless of which submodels are int8.
        let requiredModels = ModelNames.PocketTTS.requiredModels(for: language)
        let allPresent = requiredModels.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: languageRoot.appendingPathComponent(model).path)
        }

        if !allPresent {
            if let subdir = language.repoSubdirectory {
                logger.info(
                    "Downloading PocketTTS \(language.rawValue) language pack from HuggingFace (\(subdir))..."
                )
                try await DownloadUtils.downloadSubdirectory(
                    .pocketTts,
                    subdirectory: subdir,
                    to: repoDir
                )
            } else {
                logger.info("Downloading PocketTTS English models from HuggingFace...")
                try await DownloadUtils.downloadRepo(
                    .pocketTts, to: modelsDirectory, progressHandler: progressHandler)
            }
        } else {
            logger.info(
                "PocketTTS \(language.rawValue) models found in cache")
        }

        // Fetch any int8 variants requested. Each int8 submodel's mlmodelc is
        // a self-contained subtree at `languages/<lang>/int8/<basename>.mlmodelc`.
        if quantization.hasAnyInt8 {
            let int8RemoteRoot = language.int8RepoSubdirectory
            let int8LocalRoot = repoDir.appendingPathComponent(int8RemoteRoot)

            var int8Files: [String] = []
            if quantization.condStep == .int8 {
                int8Files.append(ModelNames.PocketTTS.condStepInt8File)
            }
            if quantization.flowlmStep == .int8 {
                int8Files.append(ModelNames.PocketTTS.flowlmStepInt8File)
            }
            if quantization.flowDecoder == .int8 {
                int8Files.append(ModelNames.PocketTTS.flowDecoderInt8File)
            }
            if quantization.mimiDecoder == .int8 {
                int8Files.append(ModelNames.PocketTTS.mimiDecoderInt8File)
            }

            for file in int8Files {
                let localPath = int8LocalRoot.appendingPathComponent(file)
                if FileManager.default.fileExists(atPath: localPath.path) {
                    continue
                }
                let remoteSub = "\(int8RemoteRoot)/\(file)"
                logger.info(
                    "Downloading PocketTTS \(language.rawValue) int8 submodel: \(remoteSub)..."
                )
                try await DownloadUtils.downloadSubdirectory(
                    .pocketTts,
                    subdirectory: remoteSub,
                    to: repoDir
                )
            }
        }

        // Resolve per-submodel URLs.
        let int8Root = repoDir.appendingPathComponent(language.int8RepoSubdirectory)

        func resolved(_ precision: PocketTtsModelPrecision, fp16File: String, int8File: String) -> URL {
            switch precision {
            case .fp16:
                return languageRoot.appendingPathComponent(fp16File)
            case .int8:
                return int8Root.appendingPathComponent(int8File)
            }
        }

        let condURL = resolved(
            quantization.condStep,
            fp16File: ModelNames.PocketTTS.condStepFile,
            int8File: ModelNames.PocketTTS.condStepInt8File)
        let flowlmURL = resolved(
            quantization.flowlmStep,
            fp16File: ModelNames.PocketTTS.flowlmStepFile,
            int8File: ModelNames.PocketTTS.flowlmStepInt8File)
        let flowURL = resolved(
            quantization.flowDecoder,
            fp16File: ModelNames.PocketTTS.flowDecoderFile,
            int8File: ModelNames.PocketTTS.flowDecoderInt8File)
        let mimiURL = resolved(
            quantization.mimiDecoder,
            fp16File: ModelNames.PocketTTS.mimiDecoderFile(for: language),
            int8File: ModelNames.PocketTTS.mimiDecoderInt8File)

        return PocketTtsResolvedModels(
            repoDir: repoDir,
            languageRoot: languageRoot,
            condStepURL: condURL,
            flowlmStepURL: flowlmURL,
            flowDecoderURL: flowURL,
            mimiDecoderURL: mimiURL,
            quantization: quantization
        )
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
        let voiceFile = "\(sanitized)_audio_prompt.bin"
        let voiceURL = constantsDir.appendingPathComponent(voiceFile)

        if !FileManager.default.fileExists(atPath: voiceURL.path) {
            logger.info(
                "Downloading voice '\(sanitized)' for \(language.rawValue) from HuggingFace...")
            let remotePrefix: String
            if let subdir = language.repoSubdirectory {
                remotePrefix = "\(subdir)/"
            } else {
                remotePrefix = ""
            }
            let remotePath = "\(remotePrefix)constants_bin/\(voiceFile)"
            let remoteURL = try ModelRegistry.resolveModel(Repo.pocketTts.remotePath, remotePath)
            let data = try await AssetDownloader.fetchData(
                from: remoteURL,
                description: "\(sanitized) voice prompt (\(language.rawValue))",
                logger: logger
            )
            // Make sure the parent directory exists in case this is a fresh
            // language pack that hasn't materialized constants_bin/ yet.
            try FileManager.default.createDirectory(
                at: constantsDir, withIntermediateDirectories: true)
            try data.write(to: voiceURL, options: [.atomic])
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
