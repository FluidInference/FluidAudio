import Foundation

/// Downloads the MOSS-TTS-Nano CoreML assets from `FluidInference/moss-tts-nano-coreml`.
///
/// Required on first use: `Prefill`, `Step`, `Frame`, `CodecStep` bundles plus
/// `config.json` and `tokenizer.model`. The fp32 `CodecEncoder` is fetched lazily
/// by `MossTtsNanoModelStore.encoder()` (custom voice cloning only), and built-in
/// voices come from `voices/<name>.json`.
public enum MossTtsNanoResourceDownloader {

    private static let logger = AppLogger(category: "MossTtsNanoResourceDownloader")

    @discardableResult
    public static func ensureModels(
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.mossTtsNano.folderName)
        let missing = ModelNames.MossTtsNano.requiredFiles.filter {
            !FileManager.default.fileExists(atPath: repoDir.appendingPathComponent($0).path)
        }
        if !missing.isEmpty {
            logger.info("Downloading MOSS-TTS-Nano CoreML assets from HuggingFace (\(missing.count) missing)…")
            do {
                try await ModelHub.download(.mossTtsNano, to: modelsRoot, progressHandler: progressHandler)
            } catch {
                throw MossTtsNanoError.downloadFailed("\(error)")
            }
        } else {
            logger.info("MOSS-TTS-Nano assets found in cache at \(repoDir.path)")
        }
        return repoDir
    }

    /// Fetch the fp32 codec encoder used to turn a reference clip into voice codes.
    @discardableResult
    public static func ensureEncoder(
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.mossTtsNano.folderName)
        let encoderURL = repoDir.appendingPathComponent(ModelNames.MossTtsNano.codecEncoderFile)
        if FileManager.default.fileExists(atPath: encoderURL.path) { return encoderURL }
        logger.info("Downloading MOSS-TTS-Nano codec encoder from HuggingFace…")
        do {
            try await ModelHub.download(
                .mossTtsNano, to: modelsRoot,
                additionalModelNames: [ModelNames.MossTtsNano.codecEncoderFile],
                progressHandler: progressHandler)
        } catch {
            throw MossTtsNanoError.downloadFailed("codec encoder: \(error)")
        }
        guard FileManager.default.fileExists(atPath: encoderURL.path) else {
            throw MossTtsNanoError.downloadFailed("codec encoder missing after download")
        }
        return encoderURL
    }

    /// Download (if needed) a built-in voice JSON and return its local URL.
    @discardableResult
    public static func ensureVoice(
        _ voice: MossTtsNanoBuiltInVoice,
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.mossTtsNano.folderName)
        let localURL = repoDir.appendingPathComponent(voice.fileName)
        if FileManager.default.fileExists(atPath: localURL.path) { return localURL }
        logger.info("Downloading MOSS-TTS-Nano voice \(voice.rawValue) from HuggingFace…")
        do {
            try await ModelHub.download(
                .mossTtsNano,
                subdirectory: MossTtsNanoBuiltInVoiceFiles.subdirectory,
                to: repoDir,
                progressHandler: progressHandler,
                shouldSkip: { $0 != voice.fileName }
            )
        } catch {
            throw MossTtsNanoError.downloadFailed("voice \(voice.rawValue): \(error)")
        }
        guard FileManager.default.fileExists(atPath: localURL.path) else {
            throw MossTtsNanoError.downloadFailed("voice \(voice.rawValue) missing after download")
        }
        return localURL
    }

    public static func loadVoice(
        _ voice: MossTtsNanoBuiltInVoice,
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> MossTtsNanoVoice {
        let url = try await ensureVoice(voice, directory: directory, progressHandler: progressHandler)
        return try MossTtsNanoVoice.load(from: url)
    }

    static func defaultCacheRoot() throws -> URL {
        let root = try TtsCacheDirectory.ensure().appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}
