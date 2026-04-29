import Foundation

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
    ///   - precision: Which precision variant to load (default: `.fp16`,
    ///     matching upstream's on-disk weight format).
    ///     `.int8` reuses the same upstream `v2/<lang>/` directory but loads
    ///     `flowlm_stepv2.mlmodelc` (int8 weight quantization on the FlowLM
    ///     transformer's attention + FFN linears, per kyutai-labs/pocket-tts#147)
    ///     instead of `flowlm_step.mlmodelc`. The other three models stay
    ///     at the default fp16 precision.
    ///   - progressHandler: Optional callback for download progress updates.
    /// - Returns: The directory that contains the requested `.mlmodelc`
    ///   packages plus `constants_bin/` for the requested language.
    ///
    /// Note: the upstream `v2/<lang>/` directory ships both flowlm variants,
    /// so a fresh download pulls the unused variant too. After download
    /// completes, the unused FlowLM `.mlmodelc` and `.mlpackage` directories
    /// are deleted so only the requested precision occupies disk
    /// (~217 MB savings for `.int8`, ~75 MB savings for `.fp16`).
    public static func ensureModels(
        language: PocketTtsLanguage,
        directory: URL? = nil,
        precision: PocketTtsPrecision = .fp16,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            PocketTtsConstants.defaultModelsSubdirectory)

        let repoDir = modelsDirectory.appendingPathComponent(Repo.pocketTts.folderName)
        let subdir = language.repoSubdirectory
        let languageRoot = repoDir.appendingPathComponent(subdir)

        let required = ModelNames.PocketTTS.requiredModels(precision: precision)
        let allPresent = required.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: languageRoot.appendingPathComponent(model).path)
        }

        guard !allPresent else {
            logger.info(
                "PocketTTS \(language.rawValue) (\(precision)) models found in cache")
            return languageRoot
        }

        logger.info(
            "Downloading PocketTTS \(language.rawValue) (\(precision)) language pack from HuggingFace (\(subdir))..."
        )
        try await DownloadUtils.downloadSubdirectory(
            .pocketTts,
            subdirectory: subdir,
            to: repoDir,
            progressHandler: progressHandler
        )

        // The HF subdir contains both FlowLM precisions; delete the one we
        // don't need so disk usage matches the loaded models.
        removeUnusedFlowlmVariant(at: languageRoot, keeping: precision)

        return languageRoot
    }

    /// Delete the FlowLM `.mlmodelc` and `.mlpackage` directories that don't
    /// match the requested precision. Idempotent — silently skips paths that
    /// don't exist.
    private static func removeUnusedFlowlmVariant(
        at languageRoot: URL,
        keeping precision: PocketTtsPrecision
    ) {
        let unusedNames: [String]
        switch precision {
        case .fp16:
            // Loading flowlm_step.mlmodelc; drop the int8 variant.
            unusedNames = [
                ModelNames.PocketTTS.flowlmStepV2 + ".mlmodelc",
                ModelNames.PocketTTS.flowlmStepV2 + ".mlpackage",
            ]
        case .int8:
            // Loading flowlm_stepv2.mlmodelc; drop the default variant.
            unusedNames = [
                ModelNames.PocketTTS.flowlmStep + ".mlmodelc",
                ModelNames.PocketTTS.flowlmStep + ".mlpackage",
            ]
        }
        for name in unusedNames {
            let url = languageRoot.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            do {
                try FileManager.default.removeItem(at: url)
                logger.info("Removed unused PocketTTS variant on disk: \(name)")
            } catch {
                logger.warning(
                    "Failed to remove unused PocketTTS variant \(name): \(error.localizedDescription)"
                )
            }
        }
    }

    /// Ensure the Mimi encoder model is downloaded for voice cloning.
    ///
    /// This is an optional model that's only needed for voice cloning
    /// functionality. It's downloaded separately from the main models to
    /// reduce initial download size. The encoder is shared across all
    /// language packs and lives at the repo root, so users on any language
    /// can clone a voice without pulling in another language pack.
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
        try await DownloadUtils.downloadSubdirectory(
            .pocketTts,
            subdirectory: ModelNames.PocketTTS.mimiEncoderFile,
            to: repoDir
        )

        guard FileManager.default.fileExists(atPath: encoderPath.path) else {
            throw PocketTTSError.downloadFailed("Failed to download Mimi encoder model")
        }

        return encoderPath
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
        language: PocketTtsLanguage,
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
