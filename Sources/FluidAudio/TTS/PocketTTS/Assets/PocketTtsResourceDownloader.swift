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
    /// completes, the unused FlowLM `.mlmodelc` directory is deleted so only
    /// the requested precision occupies disk (~140 MB savings for `.int8`,
    /// ~75 MB savings for `.fp16`).
    ///
    /// The downloader also skips redundant repo artifacts entirely:
    /// `.mlpackage` sources (CoreML never loads them — the compiled
    /// `.mlmodelc` next to each one is what Swift uses), `constants/`
    /// `.npy/.npz` intermediates (`constants_bin/*.bin` files are the
    /// loaded form), and incidental files (`verify.wav`, `.DS_Store`).
    public static func ensureModels(
        language: PocketTtsLanguage,
        directory: URL? = nil,
        precision: PocketTtsPrecision = .fp16,
        placement: PocketTtsModelPlacement = .gpu,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            PocketTtsConstants.defaultModelsSubdirectory)

        let repoDir = modelsDirectory.appendingPathComponent(Repo.pocketTts.folderName)
        let subdir = language.repoSubdirectory
        let languageRoot = repoDir.appendingPathComponent(subdir)

        let required = ModelNames.PocketTTS.requiredModels(
            precision: precision, placement: placement)
        let allPresent = required.allSatisfy { model in
            FileManager.default.fileExists(
                atPath: languageRoot.appendingPathComponent(model).path)
        }

        if allPresent {
            logger.info(
                "PocketTTS \(language.rawValue) (\(precision)) models found in cache")
            // Pre-#592 caches lack `constants_bin/bos_before_voice.bin`. The
            // language-pack files are otherwise complete, so try to fetch just
            // the missing constant rather than re-downloading the whole subdir.
            //
            // Best-effort: shipped snapshot voices don't need this file at all,
            // and the v1 cloned-voice prefill path enforces presence at use
            // time (PocketTtsConstantsLoader returns nil gracefully). Failing
            // the fetch here — e.g. offline, or before the file lands on HF —
            // must not block users who only synthesize with shipped voices.
            do {
                try await ensureBosBeforeVoice(language: language, languageRoot: languageRoot)
            } catch {
                logger.warning(
                    "Failed to backfill bos_before_voice.bin for \(language.rawValue): "
                        + "\(error.localizedDescription). Cloned-voice v1 prefill will fail "
                        + "until this file is available; shipped snapshot voices are unaffected."
                )
            }
            return languageRoot
        }

        logger.info(
            "Downloading PocketTTS \(language.rawValue) (\(precision)) language pack from HuggingFace (\(subdir))..."
        )
        try await DownloadUtils.downloadSubdirectory(
            .pocketTts,
            subdirectory: subdir,
            to: repoDir,
            progressHandler: progressHandler,
            shouldSkip: Self.shouldSkipAsset(at:)
        )

        // The HF subdir contains both FlowLM precisions; delete the one we
        // don't need so disk usage matches the loaded models.
        removeUnusedFlowlmVariant(at: languageRoot, keeping: precision)

        // The Trial 23 multifunction state package is not published on
        // HuggingFace yet, so the subdir download above cannot provide it.
        // Fail loudly with install instructions instead of letting the
        // model load die with a bare file-not-found.
        if placement == .aneState {
            let statePath = languageRoot.appendingPathComponent(
                ModelNames.PocketTTS.pocketStateFile)
            guard FileManager.default.fileExists(atPath: statePath.path) else {
                throw PocketTTSError.modelNotFound(
                    "\(ModelNames.PocketTTS.pocketStateFile) is required for the "
                        + "`.aneState` placement but is not published upstream. Install the "
                        + "Trial 23 multifunction artifact (mobius bench_pipeline_mlstate.py, "
                        + "pocket_flowlm_mf_state.mlmodelc) at \(statePath.path)."
                )
            }
        }

        return languageRoot
    }

    /// Skip filter applied to every remote path the HF lister walks for a
    /// PocketTTS language pack. Excluded categories were identified during
    /// the v2 repo cleanup audit: every `.mlpackage` ships with a compiled
    /// `.mlmodelc` next to it and CoreML only loads the latter; `constants/`
    /// holds intermediate `.npy/.npz` files whose binary equivalents live
    /// under `constants_bin/`; `verify.wav` is an upstream debug artifact;
    /// `.DS_Store` is macOS junk.
    @Sendable
    private static func shouldSkipAsset(at path: String) -> Bool {
        let basename = (path as NSString).lastPathComponent
        if basename == ".DS_Store" || basename == "verify.wav" {
            return true
        }
        if basename.hasSuffix(".mlpackage") || path.contains(".mlpackage/") {
            return true
        }
        // Only skip the intermediate "constants/" subdirectory, never
        // "constants_bin/" (which contains the .bin and voice .safetensors
        // files Swift loads at runtime).
        for component in path.split(separator: "/") where component == "constants" {
            return true
        }
        return false
    }

    /// Backfill `constants_bin/bos_before_voice.bin` for cached language packs
    /// that were downloaded before the FluidAudio #592 fix. New downloads pick
    /// it up via `downloadSubdirectory` — this helper exists only to upgrade
    /// older caches without a full re-download.
    private static func ensureBosBeforeVoice(
        language: PocketTtsLanguage,
        languageRoot: URL
    ) async throws {
        let constantsDir = languageRoot.appendingPathComponent(ModelNames.PocketTTS.constantsBinDir)
        let bosURL = constantsDir.appendingPathComponent("bos_before_voice.bin")
        if FileManager.default.fileExists(atPath: bosURL.path) {
            return
        }
        try FileManager.default.createDirectory(
            at: constantsDir, withIntermediateDirectories: true)
        let remotePath = "\(language.repoSubdirectory)/constants_bin/bos_before_voice.bin"
        let remoteURL = try ModelRegistry.resolveModel(Repo.pocketTts.remotePath, remotePath)
        logger.info(
            "Backfilling bos_before_voice.bin for cached \(language.rawValue) pack...")
        let data = try await AssetDownloader.fetchData(
            from: remoteURL,
            description: "bos_before_voice.bin (\(language.rawValue))",
            logger: logger
        )
        try data.write(to: bosURL, options: [.atomic])
        logger.info("Wrote bos_before_voice.bin (\(data.count) bytes)")
    }

    /// Delete the FlowLM `.mlmodelc` directory that doesn't match the
    /// requested precision. Idempotent — silently skips paths that don't
    /// exist. `.mlpackage` siblings are no longer downloaded (see
    /// `shouldSkipAsset`), so this only touches `.mlmodelc` directories.
    private static func removeUnusedFlowlmVariant(
        at languageRoot: URL,
        keeping precision: PocketTtsPrecision
    ) {
        let unusedNames: [String]
        switch precision {
        case .fp16:
            // Loading flowlm_step.mlmodelc; drop the int8 variant.
            unusedNames = [ModelNames.PocketTTS.flowlmStepV2 + ".mlmodelc"]
        case .int8:
            // Loading flowlm_stepv2.mlmodelc; drop the default variant.
            unusedNames = [ModelNames.PocketTTS.flowlmStep + ".mlmodelc"]
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
        // Delegate to the shared TTS cache root (Application Support on iOS,
        // ~/.cache/fluidaudio on macOS) so all backends share one location.
        return try TtsCacheDirectory.ensure()
    }
}
