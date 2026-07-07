import Foundation

/// On-disk model-cache knowledge for the download stack (#765 Wave 4):
/// existence/completeness checks, corrupt-cache purging, robust directory
/// creation, and the cache-clearing operations behind the ModelHub
/// public API.
enum ModelCache {

    /// Historical log category retained deliberately across the 0.16 rename so
    /// existing `category == "DownloadUtils"` predicates keep capturing the
    /// whole download trail; renaming it is a separate, opt-in decision.
    private static let logger = AppLogger(category: "DownloadUtils")

    /// Robustly create a directory, removing any conflicting files in the path.
    ///
    /// This handles cases where a file exists where a directory should be, which can happen
    /// during corrupted cache recovery when partial deletion leaves files in place of directories.
    ///
    /// - Parameter url: The directory path to create
    /// - Throws: Errors from FileManager if directory creation fails after cleanup
    static func createDirectoryRobustly(at url: URL) throws {
        let fm = FileManager.default

        // Hot path: the directory usually already exists (one stat instead of
        // one per ancestor component on every per-file call).
        var isDirectory: ObjCBool = false
        if fm.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
            return
        }

        var pathComponents = url.pathComponents

        // Remove leading "/" if present
        if pathComponents.first == "/" {
            pathComponents.removeFirst()
        }

        // Build path incrementally, checking each component
        var currentPath = "/"
        for component in pathComponents {
            currentPath = (currentPath as NSString).appendingPathComponent(component)
            let componentURL = URL(fileURLWithPath: currentPath)

            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: currentPath, isDirectory: &isDirectory) {
                if !isDirectory.boolValue {
                    // A file exists where a directory should be - remove it
                    logger.warning("Removing file blocking directory creation: \(currentPath)")
                    try fm.removeItem(at: componentURL)
                    try fm.createDirectory(at: componentURL, withIntermediateDirectories: false)
                }
                // If it's already a directory, continue
            } else {
                // Path doesn't exist, create remaining path with intermediate directories
                try fm.createDirectory(at: url, withIntermediateDirectories: true)
                return
            }
        }
    }

    /// `true` when every model in `models` exists under `repoPath`.
    static func allModelsExist(at repoPath: URL, models: Set<String>) -> Bool {
        missingModels(at: repoPath, models: models).isEmpty
    }

    /// The subset of `models` missing under `repoPath`, sorted for stable
    /// error reporting.
    static func missingModels(at repoPath: URL, models: Set<String>) -> [String] {
        models.filter { model in
            !FileManager.default.fileExists(atPath: repoPath.appendingPathComponent(model).path)
        }.sorted()
    }

    /// Throw `modelNotFound` for the (deterministically first, sorted)
    /// required model absent under `repoPath` — the post-download verify pass.
    static func verifyModelsPresent(at repoPath: URL, models: Set<String>) throws {
        if let missing = missingModels(at: repoPath, models: models).first {
            throw DownloadError.modelNotFound(path: missing)
        }
    }

    /// Validate the on-disk shape of a compiled CoreML model before loading:
    /// it must be a directory containing `coremldata.bin`.
    static func validateCompiledModelLayout(at modelPath: URL, name: String) throws {
        guard FileManager.default.fileExists(atPath: modelPath.path) else {
            throw CocoaError(
                .fileNoSuchFile,
                userInfo: [
                    NSFilePathErrorKey: modelPath.path,
                    NSLocalizedDescriptionKey: "Model file not found: \(name)",
                ])
        }

        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(atPath: modelPath.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw CocoaError(
                .fileReadCorruptFile,
                userInfo: [
                    NSFilePathErrorKey: modelPath.path,
                    NSLocalizedDescriptionKey: "Model path is not a directory: \(name)",
                ])
        }

        let coremlDataPath = modelPath.appendingPathComponent("coremldata.bin")
        guard FileManager.default.fileExists(atPath: coremlDataPath.path) else {
            logger.error("Missing coremldata.bin in \(name)")
            throw CocoaError(
                .fileReadCorruptFile,
                userInfo: [
                    NSFilePathErrorKey: coremlDataPath.path,
                    NSLocalizedDescriptionKey: "Missing coremldata.bin in model: \(name)",
                ])
        }
    }

    /// Delete a corrupted repo cache, tolerating an already-missing path
    /// (robust directory creation handles any remnants on re-download).
    ///
    /// Callers MUST check `RetryPolicy.isCancellation` first: cancellation is
    /// not corruption, and purging on a cancelled load threw away valid
    /// multi-hundred-MB caches before the guard existed (see loadModels).
    static func purgeCorruptedCache(at repoPath: URL) {
        do {
            try FileManager.default.removeItem(at: repoPath)
            logger.info("Successfully deleted corrupted cache at \(repoPath.path)")
        } catch {
            let nsError = error as NSError
            if nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileNoSuchFileError {
                // Already gone — fine.
            } else {
                logger.warning("Failed to delete cache: \(error.localizedDescription)")
                logger.info("Will attempt to overwrite during re-download")
            }
        }
    }
}
