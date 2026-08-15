import Foundation

/// Downloads the Supertonic-3 CoreML assets from HuggingFace.
///
/// FluidAudio republishes the upstream ONNX checkpoint as four `.mlmodelc`
/// bundles plus the original `tts.json` + `unicode_indexer.json` companion
/// files at `FluidInference/supertonic-3-coreml`. The bundle layout is
/// produced by `Scripts/convert_supertonic3_to_coreml.py`; see that script
/// for conversion details.
public enum Supertonic3ResourceDownloader {

    private static let logger = AppLogger(category: "Supertonic3ResourceDownloader")

    /// Ensure all required Supertonic-3 model + companion files are present
    /// locally. Returns the resolved repo directory.
    @discardableResult
    public static func ensureModels(
        directory: URL? = nil,
        veVariant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.supertonic3.folderName)

        let required = ModelNames.Supertonic3.requiredFiles(veVariant: veVariant)
        let allPresent = required.allSatisfy { file in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(file).path)
        }

        if !allPresent {
            logger.info("Downloading Supertonic-3 CoreML assets from HuggingFace…")
            do {
                try await ModelHub.download(
                    .supertonic3, to: modelsRoot, variant: veVariant,
                    progressHandler: progressHandler)
            } catch {
                throw Supertonic3Error.downloadFailed("\(error)")
            }
        } else {
            logger.info("Supertonic-3 assets found in cache at \(repoDir.path)")
        }

        return repoDir
    }

    /// Ensure a built-in voice style JSON is present locally, downloading it
    /// from `FluidInference/supertonic-3-coreml/voice_styles/` if missing, and
    /// return its local file URL. Custom voices can skip this and load any file
    /// directly via `Supertonic3VoiceStyle.load(from:)`.
    @discardableResult
    public static func downloadVoiceStyle(
        _ voice: Supertonic3Voice,
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.supertonic3.folderName)
        let localURL = repoDir.appendingPathComponent(voice.fileName)

        if FileManager.default.fileExists(atPath: localURL.path) {
            logger.info("Supertonic-3 voice \(voice.rawValue) found in cache")
            return localURL
        }

        logger.info("Downloading Supertonic-3 voice \(voice.rawValue) from HuggingFace…")
        do {
            // The HF tree API only lists directories, so pull the single file
            // out of voice_styles/ by skipping every other entry.
            try await ModelHub.download(
                .supertonic3,
                subdirectory: "voice_styles",
                to: repoDir,
                progressHandler: progressHandler,
                shouldSkip: { $0 != voice.fileName }
            )
        } catch {
            throw Supertonic3Error.downloadFailed("voice \(voice.rawValue): \(error)")
        }

        guard FileManager.default.fileExists(atPath: localURL.path) else {
            throw Supertonic3Error.downloadFailed(
                "voice \(voice.rawValue) missing after download")
        }
        return localURL
    }

    /// Download (if needed) and decode a built-in voice style in one call.
    public static func loadVoiceStyle(
        _ voice: Supertonic3Voice,
        directory: URL? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> Supertonic3VoiceStyle {
        let url = try await downloadVoiceStyle(
            voice, directory: directory, progressHandler: progressHandler)
        return try Supertonic3VoiceStyle.load(from: url)
    }

    private static func defaultCacheRoot() throws -> URL {
        // Delegate to the shared TTS cache root (Application Support on iOS,
        // ~/.cache/fluidaudio on macOS) so all backends share one location.
        let root = try TtsCacheDirectory.ensure().appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}
