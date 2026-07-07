import CoreML
import Foundation

/// The model-download surface for FluidAudio (#765 Wave 6): loading CoreML
/// model repos from HuggingFace into the local cache, targeted subdirectory
/// and single-file fetches, offline enforcement, and cache management.
///
/// Replaces the pre-0.16 `ModelHub` class — see the 0.16.0 migration
/// table for the mechanical old→new spellings.
public enum ModelHub {

    /// Historical log category retained deliberately across the 0.16 rename:
    /// existing `log stream --predicate 'category == "ModelHub"'`
    /// diagnostics keep capturing the whole download trail. Renaming the
    /// category is a separate, opt-in decision.
    private static let logger = AppLogger(category: "DownloadUtils")

    /// Shared URLSession with registry and proxy configuration. Advanced
    /// plumbing — exposed for tooling (the FluidAudio CLI's dataset
    /// downloads); apps normally never touch it.
    public static var session: URLSession { HFClient.session }

    /// Offline-only mode. When true, every download surface (`fetchWithAuth`,
    /// `download`, `fetchFile`) and the `loadModels` retry-with-redownload
    /// fallback throws `DownloadError.networkDisabled` / `.modelMissing`
    /// instead of touching the network. Applications that bundle their own
    /// model assets should set this once at startup and route loading through
    /// manual APIs (e.g. `MLModel(contentsOf:)`, `VadManager(config:vadModel:)`)
    /// so a corrupt-detected `.mlmodelc` never silently re-downloads at
    /// runtime. Set before any FluidAudio loaders are touched.
    public static var offlineMode: Bool {
        get { HFClient.offlineMode }
        set { HFClient.offlineMode = newValue }
    }

    /// Throws `DownloadError.networkDisabled` if `offlineMode` is on.
    /// Call this at the top of any path that would touch the network.
    private static func ensureOnlineAllowed(_ operation: String) throws {
        if offlineMode {
            throw DownloadError.networkDisabled(operation: operation)
        }
    }

    /// Fetch data from a URL with HuggingFace authentication if available.
    /// Advanced plumbing for API calls needing auth tokens (private repos,
    /// higher rate limits); prefer `fetchFile` for content.
    public static func fetchWithAuth(from url: URL) async throws -> (Data, URLResponse) {
        try ensureOnlineAllowed("fetchWithAuth(\(url.absoluteString))")
        return try await HFClient.fetchWithAuth(from: url)
    }

    public static func clearCache(for repo: Repo, directory: URL) {
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try? FileManager.default.removeItem(at: repoPath)
    }

    /// Remove all downloaded models and caches. Clears both cache locations:
    /// `~/Library/Application Support/FluidAudio/Models/` (ASR, VAD,
    /// Diarization) and the shared TTS root — `~/.cache/fluidaudio/` on
    /// macOS, `Application Support/fluidaudio/` on iOS.
    public static func clearAllCaches() {
        let fm = FileManager.default

        // ASR, VAD, Diarization models
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let modelsDir = appSupport.appendingPathComponent("FluidAudio/Models")
            try? fm.removeItem(at: modelsDir)
        }

        // TTS models (Kokoro, PocketTTS, Supertonic3, StyleTTS2).
        #if os(macOS)
        let home = fm.homeDirectoryForCurrentUser
        let ttsCache = home.appendingPathComponent(".cache/fluidaudio")
        try? fm.removeItem(at: ttsCache)
        #else
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let ttsCache = appSupport.appendingPathComponent("fluidaudio")
            try? fm.removeItem(at: ttsCache)
        }
        #endif

        logger.info("All model caches cleared")
    }

    public static func loadModels(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> [String: MLModel] {
        await SystemInfo.logOnce(using: logger)
        do {
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        } catch {
            // In offline mode never delete cache + re-download. Surface
            // the original load failure so the caller can decide.
            if offlineMode {
                logger.warning(
                    "Offline mode: load failed and re-download blocked. \(error.localizedDescription)"
                )
                throw error
            }

            // Cancellation is not corruption. A cancelled first load (app
            // teardown, user abort) must never wipe a valid cache — deleting
            // here threw away fully-downloaded multi-hundred-MB repos.
            if RetryPolicy.isCancellation(error) {
                logger.info(
                    "Load cancelled; preserving model cache. \(error.localizedDescription)")
                throw error
            }

            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)

            ModelCache.purgeCorruptedCache(at: repoPath)

            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        }
    }
    private static func loadModelsOnce(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws -> [String: MLModel] {
        await SystemInfo.logOnce(using: logger)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let repoPath = directory.appendingPathComponent(repo.folderName)
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
        // The caller-supplied `modelNames` may include files outside the repo's
        // default "required" set (e.g. CtcHead.mlmodelc inside parakeet-ctc-110m
        // when loaded by the TDT-CTC manager — see issue #524). Union them in
        // so the cache-validity check and the download filter both consider
        // every model the caller actually needs.
        let extraModelNames = Set(modelNames).subtracting(requiredModels)
        let effectiveModels = requiredModels.union(extraModelNames)
        let reporter = ProgressReporter(handler: progressHandler, downloadPhaseWeight: 0.5)

        if !ModelCache.allModelsExist(at: repoPath, models: effectiveModels) {
            // In offline mode surface a typed error listing the
            // missing files instead of attempting a HuggingFace fetch.
            if offlineMode {
                let missing = ModelCache.missingModels(at: repoPath, models: effectiveModels)
                logger.error(
                    "Offline mode: required models missing at \(repoPath.path): \(missing)"
                )
                throw DownloadError.modelMissing(repo: repo.folderName, missing: missing)
            }
            logger.info("Models not found in cache at \(repoPath.path)")
            try await download(
                repo, to: directory, variant: variant,
                additionalModelNames: extraModelNames,
                progressHandler: progressHandler)
        } else {
            logger.info("Found \(repo.folderName) locally, no download needed")
            reporter.cachedModelsAvailable()
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.allowLowPrecisionAccumulationOnGPU = true

        var models: [String: MLModel] = [:]
        for (index, name) in modelNames.enumerated() {
            let modelPath = repoPath.appendingPathComponent(name)
            try ModelCache.validateCompiledModelLayout(at: modelPath, name: name)

            reporter.compiling(name: name, index: index, count: modelNames.count)

            let start = Date()
            let model = try MLModel(contentsOf: modelPath, configuration: config)
            let elapsed = Date().timeIntervalSince(start)

            models[name] = model

            let ms = elapsed * 1000
            let formatted = String(format: "%.2f", ms)
            logger.info("Compiled model \(name) in \(formatted) ms :: \(SystemInfo.summary())")
        }

        reporter.finished()
        return models
    }

    /// Download a HuggingFace repository using URLSession (does not load models).
    ///
    /// - Parameter additionalModelNames: Extra model directory names (e.g.
    ///   `"CtcHead.mlmodelc"`) to fetch in addition to the repo's default
    ///   `ModelNames.getRequiredModelNames(...)` set. Used by `loadModels` to
    ///   forward caller-requested files that are not part of the repo's
    ///   baseline required set.
    public static func download(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        additionalModelNames: Set<String> = [],
        progressHandler: ProgressHandler? = nil
    ) async throws {
        try await download(
            repo, to: directory, variant: variant,
            additionalModelNames: additionalModelNames,
            progressHandler: progressHandler,
            configuration: nil)
    }

    /// Internal seam: `configuration` overrides the session used for tree
    /// listing and per-file downloads so characterization tests can drive the
    /// full listing/filtering/download pipeline with a stub `URLProtocol`
    /// (#765 Wave 1). `nil` (the public path) uses the shared session.
    static func download(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        additionalModelNames: Set<String> = [],
        progressHandler: ProgressHandler? = nil,
        configuration: URLSessionConfiguration?
    ) async throws {
        try ensureOnlineAllowed("download(\(repo.folderName))")
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let listingSession = configuration.map { URLSession(configuration: $0) } ?? session
        defer {
            if configuration != nil { listingSession.finishTasksAndInvalidate() }
        }

        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
            .union(additionalModelNames)
        let subPath = repo.subPath  // e.g., "160ms" for parakeetEou160

        // Build patterns for filtering (relative to subPath if present)
        var patterns: [String] = []
        for model in requiredModels {
            if let sub = subPath {
                patterns.append("\(sub)/\(model)/")
            } else {
                patterns.append("\(model)/")
            }
        }

        // File selection rules for repo downloads (subPath scoping, required-
        // model patterns, metadata-extension allowances);
        // DownloadFilterCharacterizationTests pins them.
        let include: (String, Bool) -> Bool = { itemPath, isDirectory in
            if isDirectory {
                // For subPath repos, only process paths within the subPath
                if let sub = subPath {
                    return itemPath == sub || itemPath.hasPrefix("\(sub)/")
                        || patterns.contains { itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/") }
                }
                return patterns.isEmpty
                    || patterns.contains { itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/") }
            }
            // For subPath repos, only include files within the subPath
            if let sub = subPath {
                let isInSubPath = itemPath.hasPrefix("\(sub)/")
                let matchesPattern =
                    patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                let isMetadata =
                    itemPath.hasSuffix(".json") || itemPath.hasSuffix(".model") || itemPath.hasSuffix(".bin")
                return isInSubPath && (matchesPattern || isMetadata)
            }
            return patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                || itemPath.hasSuffix(".json") || itemPath.hasSuffix(".txt")
        }

        // Repo loads: download occupies 0-0.5, CoreML compile 0.5-1.0.
        let reporter = ProgressReporter(handler: progressHandler, downloadPhaseWeight: 0.5)

        // Start listing from subPath if specified, otherwise from root
        reporter.listing()
        let treeFetch = HFTreeLister.fetch(using: listingSession)
        var filesToDownload: [RemoteFile] = try await HFTreeLister.listTree(
            repoRemotePath: repo.remotePath,
            startingAt: subPath ?? "",
            include: include,
            fetch: treeFetch
        )

        // Some subPath repos keep shared auxiliary files (e.g. vocab.json) at the
        // repo *root* rather than inside the precision subdirectory — the bundled
        // .mlmodelc dirs live under `q8/`, but the tokenizer vocab is shared across
        // precisions and published once at the root. The subPath traversal above
        // never visits the root, so those files are missed and the verify pass
        // below throws `modelNotFound` (issue #649). For any required *file*
        // (i.e. not an .mlmodelc/.mlpackage bundle) that the subPath sweep did not
        // already collect, fall back to grabbing a matching root-level file.
        if subPath != nil {
            let collected = Set(filesToDownload.map { ($0.path as NSString).lastPathComponent })
            let missingAux = requiredModels.filter { model in
                !model.hasSuffix(".mlmodelc") && !model.hasSuffix(".mlpackage")
                    && !collected.contains((model as NSString).lastPathComponent)
            }
            if !missingAux.isEmpty {
                // Root-level pass only: directories are pruned; a root file is
                // pulled when its name equals a missing required aux file's
                // FULL name. Slash-containing required paths (e.g.
                // voices/zf_001.bin) therefore never match a root file — a
                // same-named root file would land at the wrong local path, so
                // the loud modelNotFound from the verify pass is preferable.
                let names = Set(missingAux)
                filesToDownload += try await HFTreeLister.listTree(
                    repoRemotePath: repo.remotePath,
                    include: { itemPath, isDirectory in
                        !isDirectory && names.contains((itemPath as NSString).lastPathComponent)
                    },
                    fetch: treeFetch
                )
            }
        }

        logger.info("Found \(filesToDownload.count) files to download")

        // Compute total known bytes for byte-weighted progress.
        // Files with unknown sizes (size == -1) are treated as 0 for weighting.
        let totalBytes: Int64 = filesToDownload.reduce(0) { $0 + Int64(max(0, $1.size)) }
        var completedBytes: Int64 = 0

        // Download each file
        for (index, file) in filesToDownload.enumerated() {
            // Strip subPath prefix when saving locally
            var localPath = file.path
            if let sub = subPath, file.path.hasPrefix("\(sub)/") {
                localPath = String(file.path.dropFirst(sub.count + 1))
            }
            let destPath = repoPath.appendingPathComponent(localPath)

            let onBytes = reporter.liveBytesCallback(
                baseBytes: completedBytes,
                totalBytes: totalBytes,
                fileIndex: index,
                totalFiles: filesToDownload.count)

            // Repo caches keep the historical corrupt-recovery behavior:
            // a regular file blocking a path component is replaced.
            let outcome = try await FileDownloader.ensure(
                file: file,
                from: repo.remotePath,
                at: destPath,
                recoveringBlockedPaths: true,
                configuration: configuration,
                onBytes: onBytes
            )
            completedBytes += Int64(max(0, file.size))

            // Pinned asymmetry vs download(subdirectory:): cached/empty files
            // emit no boundary here (the pre-#765 behavior ProgressSequence
            // relies on); the subdirectory loop emits for every outcome.
            guard outcome == .downloaded else { continue }

            if (index + 1) % 10 == 0 || index == filesToDownload.count - 1 {
                logger.info("Downloaded \(index + 1)/\(filesToDownload.count) files")
            }

            reporter.fileBoundary(
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                completedFiles: index + 1,
                totalFiles: filesToDownload.count)
        }

        // Verify required models are present
        try ModelCache.verifyModelsPresent(at: repoPath, models: requiredModels)

        logger.info("Downloaded all required models for \(repo.folderName)")
    }

    /// Download a specific subdirectory from a HuggingFace repository.
    ///
    /// Use this for optional model components that aren't part of the required model set
    /// (e.g., the Mimi encoder for PocketTTS voice cloning).
    ///
    /// - Parameters:
    ///   - repo: The HuggingFace repository.
    ///   - subdirectory: Path within the repo to download (e.g. `"mimi_encoder.mlmodelc"`).
    ///   - repoDirectory: Local directory corresponding to the repo root.
    ///     Files are saved at `repoDirectory/<remote_path>`.
    ///   - shouldSkip: Optional predicate evaluated on each remote path
    ///     (both files and directories). Returning `true` excludes the file
    ///     or, for directories, skips the whole subtree without recursing.
    ///     Used to avoid pulling redundant artifacts (e.g. `.mlpackage`
    ///     sources next to compiled `.mlmodelc`).
    public static func download(
        _ repo: Repo,
        subdirectory: String,
        to repoDirectory: URL,
        progressHandler: ProgressHandler? = nil,
        shouldSkip: (@Sendable (String) -> Bool)? = nil
    ) async throws {
        try ensureOnlineAllowed("download(\(repo.folderName)/\(subdirectory))")
        // Subdirectory downloads have no compile phase: download spans 0-1.
        let reporter = ProgressReporter(handler: progressHandler, downloadPhaseWeight: 1.0)
        reporter.listing()
        let filesToDownload: [RemoteFile] = try await HFTreeLister.listTree(
            repoRemotePath: repo.remotePath,
            startingAt: subdirectory,
            include: { itemPath, _ in shouldSkip?(itemPath) != true },
            fetch: HFTreeLister.fetch(using: session)
        )
        let totalFiles = filesToDownload.count
        logger.info("Found \(totalFiles) files in \(subdirectory)")

        // Compute total known bytes for byte-weighted progress.
        // Files with unknown sizes (size == -1) are treated as 0 for weighting.
        let totalBytes: Int64 = filesToDownload.reduce(0) { $0 + Int64(max(0, $1.size)) }
        var completedBytes: Int64 = 0

        reporter.fileBoundary(
            completedBytes: completedBytes,
            totalBytes: totalBytes,
            completedFiles: 0,
            totalFiles: totalFiles)

        for (index, file) in filesToDownload.enumerated() {
            let destPath = repoDirectory.appendingPathComponent(file.path)

            // Only stream live byte progress for files with a known size: an
            // unknown-size file (-1) carries zero weight in totalBytes, so its
            // real bytesWritten would inflate the fraction mid-file and snap
            // back at the boundary. Boundary emits keep progress monotonic.
            let onBytes =
                file.size > 0
                ? reporter.liveBytesCallback(
                    baseBytes: completedBytes,
                    totalBytes: totalBytes,
                    fileIndex: index,
                    totalFiles: totalFiles)
                : nil

            // Fail loudly on blocked paths: subdirectory downloads land in
            // caller-provided directories, so a regular file where a directory
            // belongs is surfaced, never silently deleted.
            let outcome = try await FileDownloader.ensure(
                file: file,
                from: repo.remotePath,
                at: destPath,
                recoveringBlockedPaths: false,
                onBytes: onBytes
            )
            completedBytes += Int64(max(0, file.size))

            reporter.fileBoundary(
                completedBytes: completedBytes,
                totalBytes: totalBytes,
                completedFiles: index + 1,
                totalFiles: totalFiles)

            if outcome != .alreadyPresent, (index + 1) % 5 == 0 || index == totalFiles - 1 {
                logger.info("Downloaded \(index + 1)/\(totalFiles) \(subdirectory) files")
            }
        }

        logger.info("Downloaded \(subdirectory) from \(repo.folderName)")
    }

    /// Fetch a single file from HuggingFace with the converged retry policy
    /// (#765 Wave 5): permanent errors (404s) fail fast instead of consuming
    /// the backoff budget, 5xx/rate-limits retry with Retry-After pacing, and
    /// HTML/empty bodies are rejected instead of returned as content.
    public static func fetchFile(
        from url: URL,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0
    ) async throws -> Data {
        try await fetchFile(
            from: url, description: description,
            maxAttempts: maxAttempts, minBackoff: minBackoff,
            configuration: nil)
    }

    /// Internal seam: `configuration` lets tests stub the transport.
    static func fetchFile(
        from url: URL,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0,
        configuration: URLSessionConfiguration?
    ) async throws -> Data {
        try ensureOnlineAllowed("fetchFile(\(description))")
        return try await FileDownloader.fetchData(
            from: url, description: description,
            maxAttempts: maxAttempts, minBackoff: minBackoff,
            configuration: configuration)
    }
}
