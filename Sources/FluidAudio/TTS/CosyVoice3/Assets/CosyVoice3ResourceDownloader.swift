import Foundation

/// Pulls CosyVoice3 CoreML models + runtime assets from the
/// `FluidInference/CosyVoice3-0.5B-coreml` HuggingFace repo.
///
/// Layout produced on disk (relative to `ensureCoreModels(...)`'s return URL):
///
/// ```
/// <repoDirectory>/
/// ├── LLM-Prefill-T256-M768-fp16.mlmodelc/
/// ├── LLM-Decode-M768-fp16.mlmodelc/
/// ├── Flow-N250-fp16.mlmodelc/
/// ├── HiFT-T500-fp16.mlmodelc/
/// ├── embeddings/
/// │   ├── speech_embedding-fp16.safetensors
/// │   └── embeddings-runtime-fp32.safetensors   (text-mode only)
/// ├── tokenizer/
/// │   ├── vocab.json, merges.txt, tokenizer_config.json, special_tokens.json
/// └── voices/
///     ├── cosyvoice3-default-zh.safetensors + .json   (default voice, eager)
///     └── <voice-id>.safetensors + .json              (optional, on-demand)
/// ```
public enum CosyVoice3ResourceDownloader {

    private static let logger = AppLogger(
        subsystem: "com.fluidaudio.tts", category: "CosyVoice3ResourceDownloader")

    /// Path bundle produced by `ensureTextFrontendAssets`.
    public struct TextFrontendPaths: Sendable {
        public let tokenizerDirectory: URL
        public let runtimeEmbeddingsFile: URL
        public let specialTokensFile: URL
    }

    // MARK: - Core models + speech embedding table

    /// Ensure the four `.mlmodelc` bundles and `speech_embedding-fp16.safetensors`
    /// are cached locally. Returns the repository root directory.
    ///
    /// - Parameters:
    ///   - directory: Optional base cache dir. When `nil`, defaults to
    ///     `~/.cache/fluidaudio` (macOS) or `Caches/fluidaudio` (iOS).
    ///   - progressHandler: Forwarded to `DownloadUtils.downloadRepo`.
    @discardableResult
    public static func ensureCoreModels(
        directory: URL? = nil,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let targetDir = try directory ?? cacheDirectory()
        let modelsDirectory = targetDir.appendingPathComponent(
            CosyVoice3Constants.defaultModelsSubdirectory)
        let repoDir = modelsDirectory.appendingPathComponent(Repo.cosyvoice3.folderName)

        // 1. Fetch the four .mlmodelc bundles via the standard repo downloader.
        let modelsPresent = ModelNames.CosyVoice3.requiredModels.allSatisfy { name in
            FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(name).path)
        }
        if !modelsPresent {
            logger.info("Downloading CosyVoice3 .mlmodelc bundles from HuggingFace...")
            try await DownloadUtils.downloadRepo(
                .cosyvoice3,
                to: modelsDirectory,
                progressHandler: progressHandler)
        } else {
            logger.info("CosyVoice3 .mlmodelc bundles found in cache")
        }

        // 2. Fetch the small speech-embedding table (sidecar, not a model).
        _ = try await ensureSidecarFile(
            subdir: ModelNames.CosyVoice3.Sidecar.embeddingsDir,
            name: ModelNames.CosyVoice3.Sidecar.speechEmbeddings,
            repoDirectory: repoDir,
            description: "CosyVoice3 speech embedding table")

        return repoDir
    }

    // MARK: - Text-mode assets (tokenizer + 542 MB runtime embeddings)

    /// Ensure tokenizer assets + `embeddings-runtime-fp32.safetensors` are on
    /// disk. Only required when using `CosyVoice3TtsManager.synthesize(text:…)`;
    /// fixture-mode callers may skip this.
    public static func ensureTextFrontendAssets(
        repoDirectory: URL
    ) async throws -> TextFrontendPaths {
        // Tokenizer subdirectory: vocab.json + merges.txt + special_tokens.json
        // + tokenizer_config.json. `downloadSubdirectory` walks the tree and
        // skips files already on disk.
        let tokenizerDir = repoDirectory.appendingPathComponent(
            ModelNames.CosyVoice3.Sidecar.tokenizerDir)
        let tokenizerRequired = [
            ModelNames.CosyVoice3.Sidecar.vocab,
            ModelNames.CosyVoice3.Sidecar.merges,
            ModelNames.CosyVoice3.Sidecar.specialTokens,
        ]
        let tokenizerPresent = tokenizerRequired.allSatisfy { name in
            FileManager.default.fileExists(
                atPath: tokenizerDir.appendingPathComponent(name).path)
        }
        if !tokenizerPresent {
            logger.info("Downloading CosyVoice3 tokenizer assets…")
            try await DownloadUtils.downloadSubdirectory(
                .cosyvoice3,
                subdirectory: ModelNames.CosyVoice3.Sidecar.tokenizerDir,
                to: repoDirectory)
        }

        // Runtime text-embedding table (542 MB). Pulled as a file download so
        // it never has to sit in RAM during transfer.
        let runtimeEmbeddings = try await ensureSidecarFile(
            subdir: ModelNames.CosyVoice3.Sidecar.embeddingsDir,
            name: ModelNames.CosyVoice3.Sidecar.runtimeEmbeddings,
            repoDirectory: repoDirectory,
            description: "CosyVoice3 runtime text embedding table (542 MB)")

        return TextFrontendPaths(
            tokenizerDirectory: tokenizerDir,
            runtimeEmbeddingsFile: runtimeEmbeddings,
            specialTokensFile: tokenizerDir.appendingPathComponent(
                ModelNames.CosyVoice3.Sidecar.specialTokens))
    }

    // MARK: - Voice bundles

    /// Ensure the requested zero-shot voice bundle (`<id>.safetensors` +
    /// `<id>.json`) is cached. Returns the `.safetensors` URL that
    /// `CosyVoice3PromptAssets.load(from:)` expects — the loader derives the
    /// `.json` sidecar path from it.
    @discardableResult
    public static func ensureVoice(
        voiceId: String = ModelNames.CosyVoice3.Sidecar.defaultVoiceId,
        repoDirectory: URL
    ) async throws -> URL {
        let sanitized = voiceId.filter { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" }
        guard !sanitized.isEmpty, sanitized == voiceId else {
            throw CosyVoice3Error.invalidShape("invalid voice id: \(voiceId)")
        }

        let voicesDir = repoDirectory.appendingPathComponent(
            ModelNames.CosyVoice3.Sidecar.voicesDir)
        try FileManager.default.createDirectory(
            at: voicesDir, withIntermediateDirectories: true)

        let tensorsURL = voicesDir.appendingPathComponent("\(voiceId).safetensors")
        let metadataURL = voicesDir.appendingPathComponent("\(voiceId).json")

        for (local, remoteName, desc) in [
            (tensorsURL, "\(voiceId).safetensors", "voice tensors"),
            (metadataURL, "\(voiceId).json", "voice metadata"),
        ] {
            if FileManager.default.fileExists(atPath: local.path) { continue }
            let remotePath = "\(ModelNames.CosyVoice3.Sidecar.voicesDir)/\(remoteName)"
            let remoteURL = try ModelRegistry.resolveModel(
                Repo.cosyvoice3.remotePath, remotePath)
            let descriptor = AssetDownloader.Descriptor(
                description: "\(voiceId) \(desc)",
                remoteURL: remoteURL,
                destinationURL: local,
                transferMode: .file())
            _ = try await AssetDownloader.ensure(descriptor, logger: logger)
        }

        return tensorsURL
    }

    // MARK: - Helpers

    private static func ensureSidecarFile(
        subdir: String,
        name: String,
        repoDirectory: URL,
        description: String
    ) async throws -> URL {
        let localDir = repoDirectory.appendingPathComponent(subdir)
        try FileManager.default.createDirectory(
            at: localDir, withIntermediateDirectories: true)
        let localURL = localDir.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: localURL.path) {
            return localURL
        }
        let remotePath = "\(subdir)/\(name)"
        let remoteURL = try ModelRegistry.resolveModel(
            Repo.cosyvoice3.remotePath, remotePath)
        let descriptor = AssetDownloader.Descriptor(
            description: description,
            remoteURL: remoteURL,
            destinationURL: localURL,
            transferMode: .file())
        return try await AssetDownloader.ensure(descriptor, logger: logger)
    }

    /// `~/.cache/fluidaudio` (macOS) / `Caches/fluidaudio` (iOS) — matches the
    /// convention used by `TtsResourceDownloader` and `PocketTtsResourceDownloader`.
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
            throw CosyVoice3Error.invalidShape("failed to locate caches directory")
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
