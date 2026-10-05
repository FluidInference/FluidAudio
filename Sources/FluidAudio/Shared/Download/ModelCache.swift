import CryptoKit
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
    private static let revisionMarkerName = ".fluidaudio-revision"

    /// Whether the managed cache contains files from `revision`.
    ///
    /// Historical `main` caches predate revision markers and remain valid. A
    /// pinned revision always requires an exact marker so an SDK revision bump
    /// cannot silently reuse files downloaded by an older release.
    static func matchesRevision(at repoPath: URL, revision: String) -> Bool {
        let marker = repoPath.appendingPathComponent(revisionMarkerName)
        guard let data = try? Data(contentsOf: marker) else {
            return revision == "main"
        }
        guard let storedRevision = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            return false
        }
        return storedRevision == revision
    }

    /// Inventory every cached compiled bundle and plain file before listing a
    /// legacy pinned cache. A revision marker covers all variants in this folder.
    static func legacyCacheContents(
        at repoPath: URL, revision: String
    ) throws -> (bundles: Set<String>, files: Set<String>)? {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard revision != "main",
            !fm.fileExists(atPath: repoPath.appendingPathComponent(revisionMarkerName).path),
            fm.fileExists(atPath: repoPath.path, isDirectory: &isDirectory), isDirectory.boolValue
        else { return nil }

        let root = repoPath.resolvingSymlinksInPath()
        var bundles: Set<String> = root.pathExtension == "mlmodelc" ? [""] : []
        var files: Set<String> = []
        guard let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: nil) else {
            return (bundles, files)
        }
        for case let item as URL in enumerator {
            let item = item.resolvingSymlinksInPath()
            let relative = String(item.path.dropFirst(root.path.count + 1))
            if item.pathExtension == "mlmodelc" {
                bundles.insert(relative)
                enumerator.skipDescendants()
                continue
            }
            guard !relative.hasSuffix(".partial"), !relative.hasSuffix(".partial.etag"),
                let attributes = try attributesIfPresent(at: item),
                attributes[.type] as? FileAttributeType == .typeRegular
            else { continue }
            files.insert(relative)
        }
        return (bundles, files)
    }

    /// Adopt only after the pinned listing covers every locally cached bundle.
    /// Incomplete bundles are removed as a unit so the model-existence gate
    /// re-enters download after an interruption. Existing markers remain authoritative.
    static func adoptLegacyCache(
        at repoPath: URL, revision: String, files: [RemoteFile], subPath: String? = nil
    ) throws {
        let fm = FileManager.default
        guard let contents = try legacyCacheContents(at: repoPath, revision: revision) else { return }
        let marker = repoPath.appendingPathComponent(revisionMarkerName)
        var listedBundles: Set<String> = []
        var listedFiles: Set<String> = []
        var invalidBundles: Set<String> = []
        var invalidFiles: Set<URL> = []
        var keptFiles: [(url: URL, bundle: String?)] = []
        for file in files {
            let localPath = localPath(for: file.path, subPath: subPath)
            let components = localPath.split(separator: "/")
            let bundle =
                repoPath.pathExtension == "mlmodelc"
                ? ""
                : components.firstIndex(where: { $0.hasSuffix(".mlmodelc") }).map {
                    components[...$0].joined(separator: "/")
                }
            if let bundle { listedBundles.insert(bundle) }
            listedFiles.insert(localPath)
            let destination = repoPath.appendingPathComponent(localPath)
            guard try hasMatchingContent(file, at: destination) else {
                if let bundle {
                    invalidBundles.insert(bundle)
                } else {
                    invalidFiles.insert(destination)
                }
                continue
            }
            keptFiles.append((destination, bundle))
        }
        invalidBundles.formUnion(contents.bundles.subtracting(listedBundles))
        for file in contents.files.subtracting(listedFiles) {
            invalidFiles.insert(repoPath.appendingPathComponent(file))
        }
        var removals = invalidBundles.sorted().map {
            $0.isEmpty ? repoPath : repoPath.appendingPathComponent($0)
        }
        removals.append(contentsOf: invalidFiles)
        // Rejected or missing files must not return through finished-partial reuse.
        for file in invalidFiles {
            removals.append(file.appendingPathExtension("partial"))
            removals.append(file.appendingPathExtension("partial.etag"))
        }
        for file in keptFiles where file.bundle.map({ !invalidBundles.contains($0) }) ?? true {
            removals.append(file.url.appendingPathExtension("partial"))
            removals.append(file.url.appendingPathExtension("partial.etag"))
        }
        for path in removals {
            do {
                try fm.removeItem(at: path)
            } catch {
                guard isMissingFile(error) else { throw error }
            }
        }
        try fm.createDirectory(at: repoPath, withIntermediateDirectories: true)
        try Data((revision + "\n").utf8).write(to: marker, options: .atomic)
    }

    /// Size is only a pre-check: an unknown size can still match its content ID.
    private static func hasMatchingContent(_ file: RemoteFile, at destination: URL) throws -> Bool {
        guard let contentID = file.contentID,
            let attributes = try attributesIfPresent(at: destination),
            attributes[.type] as? FileAttributeType == .typeRegular,
            let size = (attributes[.size] as? NSNumber)?.int64Value,
            file.size < 0 || size == Int64(file.size)
        else { return false }

        let expected: String
        let hexLength: Int
        switch contentID {
        case .lfsSHA256(let oid):
            expected = oid.lowercased()
            hexLength = 64
        case .gitBlobSHA1(let oid):
            expected = oid.lowercased()
            hexLength = 40
        }
        guard expected.utf8.count == hexLength,
            expected.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
        else { return false }

        do {
            switch contentID {
            case .lfsSHA256:
                return try hashFile(at: destination, using: SHA256()) == expected
            case .gitBlobSHA1:
                var hasher = Insecure.SHA1()
                hasher.update(data: Data("blob \(size)\0".utf8))
                return try hashFile(at: destination, using: hasher) == expected
            }
        } catch {
            guard isMissingFile(error) else { throw error }
            return false
        }
    }

    /// Stream large weights in bounded chunks rather than materializing them in memory.
    private static func hashFile<H: HashFunction>(at url: URL, using initialHasher: H) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = initialHasher
        while let chunk = try handle.read(upToCount: 1_048_576), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func localPath(for remotePath: String, subPath: String?) -> String {
        guard let subPath, remotePath.hasPrefix("\(subPath)/") else { return remotePath }
        return String(remotePath.dropFirst(subPath.count + 1))
    }

    /// Missing-file races are harmless; permission and other I/O errors still propagate.
    static func isMissingFile(_ error: Error) -> Bool {
        let error = error as NSError
        if error.domain == NSCocoaErrorDomain,
            error.code == NSFileNoSuchFileError || error.code == NSFileReadNoSuchFileError
        {
            return true
        }
        if error.domain == NSPOSIXErrorDomain, error.code == Int(POSIXErrorCode.ENOENT.rawValue) { return true }
        guard let underlying = error.userInfo[NSUnderlyingErrorKey] as? Error else { return false }
        return isMissingFile(underlying)
    }

    private static func attributesIfPresent(at url: URL) throws -> [FileAttributeKey: Any]? {
        do {
            return try FileManager.default.attributesOfItem(atPath: url.path)
        } catch {
            guard isMissingFile(error) else { throw error }
            return nil
        }
    }

    /// Prepare a managed cache for downloads from one resolved revision.
    /// Existing files are preserved when the marker matches and replaced when
    /// the requested revision changes. The marker is written before downloads
    /// begin so interrupted files can resume on the next attempt.
    static func prepareForDownload(at repoPath: URL, revision: String) throws {
        let fm = FileManager.default
        guard !matchesRevision(at: repoPath, revision: revision) else {
            try fm.createDirectory(at: repoPath, withIntermediateDirectories: true)
            return
        }

        if fm.fileExists(atPath: repoPath.path) {
            try fm.removeItem(at: repoPath)
        }
        try fm.createDirectory(at: repoPath, withIntermediateDirectories: true)
        guard revision != "main" else { return }

        let marker = repoPath.appendingPathComponent(revisionMarkerName)
        try Data((revision + "\n").utf8).write(to: marker, options: .atomic)
    }

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

    /// The subset of `requiredFiles` that is missing or not load-ready under
    /// `repoPath`, sorted for stable error reporting. Compiled bundles
    /// (`.mlmodelc`) must pass `validateCompiledModelLayout` and contain no
    /// `*.partial` download staging file; plain files must exist.
    ///
    /// This is the cache-validity check behind `ModelHub.loadWithRecovery`.
    /// A bare directory-existence check passes for a bundle whose download
    /// was interrupted mid-weights — leaving `weights/weight.bin.partial`
    /// and no root `coremldata.bin` — which then fails every subsequent
    /// `MLModel.load` while the downloader believes the cache is warm
    /// (issue #819). This check reports such a bundle as incomplete so the
    /// downloader re-runs and resumes the partial file.
    static func incompleteFiles(at repoPath: URL, requiredFiles: Set<String>) -> [String] {
        requiredFiles.filter { file in
            let path = repoPath.appendingPathComponent(file)
            if file.hasSuffix(".mlmodelc") {
                guard (try? validateCompiledModelLayout(at: path, name: file)) != nil else { return true }
                return containsPartialDownload(at: path)
            }
            return !FileManager.default.fileExists(atPath: path.path)
        }.sorted()
    }

    /// `true` when every file in `requiredFiles` is present and load-ready
    /// under `repoPath` — see `incompleteFiles(at:requiredFiles:)`.
    static func isCacheComplete(at repoPath: URL, requiredFiles: Set<String>) -> Bool {
        incompleteFiles(at: repoPath, requiredFiles: requiredFiles).isEmpty
    }

    /// `true` when any `*.partial` staging file from an interrupted
    /// `FileDownloader` fetch remains under `url`.
    static func containsPartialDownload(at url: URL) -> Bool {
        guard
            let enumerator = FileManager.default.enumerator(
                at: url, includingPropertiesForKeys: nil)
        else { return false }
        for case let item as URL in enumerator where item.pathExtension == "partial" {
            return true
        }
        return false
    }

    /// Files whose on-disk size is smaller than (or missing versus) the
    /// published remote size, as `"path (local/remote bytes)"`, sorted.
    ///
    /// The pure comparison behind `ModelHub.logLoadFailureSizeDiagnosis`:
    /// a truncated weight file and a full-size model that cannot run on
    /// this hardware produce the same CoreML "Unable to load model" error
    /// (issue #819 discussion / #828); byte counts are what separates
    /// them. `subPath` is stripped from remote paths to form local paths
    /// under `repoPath`; remote entries with unreported sizes (-1) are
    /// skipped.
    static func undersizedFiles(
        remote: [RemoteFile], at repoPath: URL, subPath: String?
    ) -> [String] {
        var short: [String] = []
        for file in remote where file.size > 0 {
            var localRel = file.path
            if let sub = subPath, localRel.hasPrefix("\(sub)/") {
                localRel = String(localRel.dropFirst(sub.count + 1))
            }
            let localPath = repoPath.appendingPathComponent(localRel).path
            let attributes = try? FileManager.default.attributesOfItem(atPath: localPath)
            let localSize = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
            if localSize < Int64(file.size) {
                short.append("\(localRel) (\(localSize)/\(file.size) bytes)")
            }
        }
        return short.sorted()
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
