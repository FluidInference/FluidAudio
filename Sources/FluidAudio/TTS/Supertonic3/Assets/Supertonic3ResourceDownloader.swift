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
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> URL {
        let modelsRoot = try directory ?? defaultCacheRoot()
        let repoDir = modelsRoot.appendingPathComponent(Repo.supertonic3.folderName)

        let allPresent = ModelNames.Supertonic3.requiredFiles.allSatisfy { file in
            FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(file).path)
        }

        if !allPresent {
            logger.info("Downloading Supertonic-3 CoreML assets from HuggingFace…")
            do {
                try await DownloadUtils.downloadRepo(
                    .supertonic3, to: modelsRoot, progressHandler: progressHandler)
            } catch {
                throw Supertonic3Error.downloadFailed("\(error)")
            }
        } else {
            logger.info("Supertonic-3 assets found in cache at \(repoDir.path)")
        }

        return repoDir
    }

    private static func defaultCacheRoot() throws -> URL {
        let base: URL
        #if os(macOS)
        base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache")
        #else
        guard
            let first = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        else {
            throw Supertonic3Error.downloadFailed("failed to locate caches directory")
        }
        base = first
        #endif
        let root = base.appendingPathComponent("fluidaudio").appendingPathComponent("Models")
        if !FileManager.default.fileExists(atPath: root.path) {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        }
        return root
    }
}
