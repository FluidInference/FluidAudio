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
        config: DownloadConfig = .default,
        progressHandler: ProgressHandler? = nil
    ) async throws -> [String: MLModel] {
        await SystemInfo.logOnce(using: logger)
        do {
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                config: config, progressHandler: progressHandler)
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

            // Transient network failures are not corruption either. An error
            // reaching here has already exhausted RetryPolicy's in-flight
            // retries, but the bytes on disk are valid and resumable
            // (FileDownloader streams into `.partial` files with HTTP Range
            // resume). Wiping would discard exactly the bytes that make the
            // caller's next attempt cheap — on the flaky networks most
            // likely to land here.
            if RetryPolicy.isRetryable(error) {
                logger.warning(
                    "Load failed with transient network error; preserving model cache for resume. \(error.localizedDescription)"
                )
                throw error
            }

            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)

            ModelCache.purgeCorruptedCache(at: repoPath)

            do {
                return try await loadModelsOnce(
                    repo, modelNames: modelNames,
                    directory: directory, computeUnits: computeUnits, variant: variant,
                    config: config, progressHandler: progressHandler)
            } catch let retryError {
                if !RetryPolicy.isCancellation(retryError) {
                    let required = ModelNames.getRequiredModelNames(for: repo, variant: variant)
                        .union(modelNames)
                    await logLoadFailureSizeDiagnosis(
                        repo, directory: directory, requiredFiles: required)
                }
                throw retryError
            }
        }
    }
    /// Distinguish the two look-alike load-failure classes from the #819
    /// discussion: CoreML's "Unable to load model" reads the same whether
    /// the bytes on disk are short (truncated download — refetching helps)
    /// or the full-size model cannot build an execution plan on this
    /// hardware (#828 — refetching cannot help). Called after a
    /// purge-and-redownload retry has ALSO failed; compares each required
    /// file's on-disk size against the published HuggingFace size and logs
    /// which class this failure is. Best-effort diagnostics only: needs
    /// the network for the tree listing, silent in offline mode, never
    /// throws.
    static func logLoadFailureSizeDiagnosis(
        _ repo: Repo, directory: URL, requiredFiles: Set<String>
    ) async {
        guard !offlineMode else { return }
        let repoPath = directory.appendingPathComponent(repo.folderName)
        let subPath = repo.subPath
        var patterns: [String] = []
        for model in requiredFiles {
            if let sub = subPath {
                patterns.append("\(sub)/\(model)")
            } else {
                patterns.append(model)
            }
        }
        do {
            let remote = try await HFTreeLister.listTree(
                repoRemotePath: repo.remotePath,
                startingAt: subPath ?? "",
                include: { itemPath, isDirectory in
                    if isDirectory {
                        // Descend into ancestors of a pattern and into the
                        // required bundles themselves.
                        return patterns.contains {
                            itemPath == $0 || itemPath.hasPrefix($0 + "/") || $0.hasPrefix(itemPath + "/")
                        }
                    }
                    return patterns.contains { itemPath == $0 || itemPath.hasPrefix($0 + "/") }
                },
                fetch: HFTreeLister.fetch(using: session)
            )
            let undersized = ModelCache.undersizedFiles(remote: remote, at: repoPath, subPath: subPath)
            if undersized.isEmpty {
                logger.error(
                    "Load still failing but all \(remote.count) required \(repo.folderName) files match their published HuggingFace sizes — the download is intact and re-downloading cannot fix this. The model likely cannot run on this hardware/OS (see issue #828); try a different precision or variant."
                )
            } else {
                logger.error(
                    "Load still failing and \(undersized.count) \(repo.folderName) file(s) are smaller than their published HuggingFace size — the cache is truncated; clear it (ModelHub.clearCache) and re-download: \(undersized.joined(separator: ", "))"
                )
            }
        } catch {
            logger.warning(
                "Load-failure size diagnosis unavailable: \(error.localizedDescription)")
        }
    }

    /// Ensure a complete cache for `repo`, run `load`, and recover from a
    /// corrupted cache by purging and re-downloading once.
    ///
    /// The recovery wrapper for managers that load models themselves with
    /// per-model `MLModelConfiguration`s (e.g. encoder on ANE, decoder on
    /// CPU) and therefore cannot use `loadModels(_:modelNames:...)`. It
    /// provides the same guarantees that path has (issue #819):
    ///
    /// - Cache validity is judged per required file — every `.mlmodelc`
    ///   must have its root `coremldata.bin` and no `*.partial` staging
    ///   file — not by bare directory existence. An interrupted download
    ///   therefore triggers a re-download (resuming any partial file via
    ///   HTTP Range) instead of being mistaken for a warm cache.
    /// - When `load` throws, the repo cache is purged and re-downloaded
    ///   and `load` retried once — except in offline mode, on
    ///   cancellation, and on transient network errors, where the cache
    ///   is preserved and the error rethrown.
    ///
    /// - Parameters:
    ///   - requiredFiles: Paths relative to the repo cache directory that
    ///     `load` needs (bundle names like `"encoder.mlmodelc"`, nested
    ///     paths like `"encoder/encoder_int8.mlmodelc"`, or plain files
    ///     like `"vocab.json"`). Entries outside the repo's registry set
    ///     are forwarded to the download as `additionalModelNames`.
    ///   - load: Loads the models from the cache; its result is returned.
    public static func loadWithRecovery<T: Sendable>(
        _ repo: Repo,
        directory: URL,
        requiredFiles: Set<String>,
        variant: String? = nil,
        config: DownloadConfig = .default,
        progressHandler: ProgressHandler? = nil,
        load: @Sendable () async throws -> T
    ) async throws -> T {
        await SystemInfo.logOnce(using: logger)
        let repoPath = directory.appendingPathComponent(repo.folderName)
        let additionalModelNames = requiredFiles.subtracting(
            ModelNames.getRequiredModelNames(for: repo, variant: variant))

        func ensureCacheComplete() async throws {
            let incomplete = ModelCache.incompleteFiles(at: repoPath, requiredFiles: requiredFiles)
            if incomplete.isEmpty {
                logger.info("Found \(repo.folderName) locally, no download needed")
                return
            }
            if offlineMode {
                logger.error(
                    "Offline mode: required models missing or incomplete at \(repoPath.path): \(incomplete)"
                )
                throw DownloadError.modelMissing(repo: repo.folderName, missing: incomplete)
            }
            logger.info("Models missing or incomplete in cache at \(repoPath.path): \(incomplete)")
            try await download(
                repo, to: directory, variant: variant,
                additionalModelNames: additionalModelNames,
                config: config,
                progressHandler: progressHandler)
        }

        try await ensureCacheComplete()
        do {
            return try await load()
        } catch {
            // Mirror loadModels: offline mode never purges or re-downloads;
            // cancellation and transient network failures are not corruption
            // and must preserve the (valid, resumable) bytes on disk.
            if offlineMode {
                logger.warning(
                    "Offline mode: load failed and re-download blocked. \(error.localizedDescription)"
                )
                throw error
            }
            if RetryPolicy.isCancellation(error) {
                logger.info(
                    "Load cancelled; preserving model cache. \(error.localizedDescription)")
                throw error
            }
            if RetryPolicy.isRetryable(error) {
                logger.warning(
                    "Load failed with transient network error; preserving model cache for resume. \(error.localizedDescription)"
                )
                throw error
            }

            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            ModelCache.purgeCorruptedCache(at: repoPath)

            try await ensureCacheComplete()
            do {
                return try await load()
            } catch let retryError {
                if !RetryPolicy.isCancellation(retryError) {
                    await logLoadFailureSizeDiagnosis(
                        repo, directory: directory, requiredFiles: requiredFiles)
                }
                throw retryError
            }
        }
    }

    private static func loadModelsOnce(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil,
        config: DownloadConfig = .default,
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
                config: config,
                progressHandler: progressHandler)
        } else {
            logger.info("Found \(repo.folderName) locally, no download needed")
            reporter.cachedModelsAvailable()
        }

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits
        mlConfig.allowLowPrecisionAccumulationOnGPU = true

        var models: [String: MLModel] = [:]
        for (index, name) in modelNames.enumerated() {
            let modelPath = repoPath.appendingPathComponent(name)
            try ModelCache.validateCompiledModelLayout(at: modelPath, name: name)

            reporter.compiling(name: name, index: index, count: modelNames.count)

            let start = Date()
            let model = try MLModel(contentsOf: modelPath, configuration: mlConfig)
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
        config: DownloadConfig = .default,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        try await download(
            repo, to: directory, variant: variant,
            additionalModelNames: additionalModelNames,
            config: config,
            progressHandler: progressHandler,
            configuration: nil)
    }

    /// Internal seam: `configuration` overrides the session used for tree
    /// listing and per-file downloads so tests can drive the full
    /// listing/filtering/download pipeline with a stub `URLProtocol`
    /// (see `DownloadResumeTests`, `HFTreeListerTests`). `nil` (the public
    /// path) uses the shared session.
    static func download(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        additionalModelNames: Set<String> = [],
        config: DownloadConfig = .default,
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

        let include = Self.repoIncludeRule(subPath: subPath, patterns: patterns)

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
                config: config,
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
        config: DownloadConfig = .default,
        progressHandler: ProgressHandler? = nil,
        shouldSkip: (@Sendable (String) -> Bool)? = nil
    ) async throws {
        try await download(
            repo, subdirectory: subdirectory, to: repoDirectory,
            config: config, progressHandler: progressHandler, shouldSkip: shouldSkip,
            configuration: nil)
    }

    /// Internal seam: `configuration` overrides the session used for tree
    /// listing and per-file downloads so tests can drive the pipeline with a
    /// stub `URLProtocol` (see `SubdirectoryDownloadTests`).
    static func download(
        _ repo: Repo,
        subdirectory: String,
        to repoDirectory: URL,
        config: DownloadConfig = .default,
        progressHandler: ProgressHandler? = nil,
        shouldSkip: (@Sendable (String) -> Bool)? = nil,
        configuration: URLSessionConfiguration?
    ) async throws {
        try ensureOnlineAllowed("download(\(repo.folderName)/\(subdirectory))")

        let listingSession = configuration.map { URLSession(configuration: $0) } ?? session
        defer {
            if configuration != nil { listingSession.finishTasksAndInvalidate() }
        }

        // Subdirectory downloads have no compile phase: download spans 0-1.
        let reporter = ProgressReporter(handler: progressHandler, downloadPhaseWeight: 1.0)
        reporter.listing()
        let filesToDownload: [RemoteFile] = try await HFTreeLister.listTree(
            repoRemotePath: repo.remotePath,
            startingAt: subdirectory,
            include: { itemPath, _ in shouldSkip?(itemPath) != true },
            fetch: HFTreeLister.fetch(using: listingSession)
        )
        let totalFiles = filesToDownload.count
        logger.info("Found \(totalFiles) files in \(subdirectory)")

        // Compute total known bytes for byte-weighted progress.
        // Files with unknown sizes (size == -1) are treated as 0 for weighting.
        let totalBytes: Int64 = filesToDownload.reduce(0) { $0 + Int64(max(0, $1.size)) }

        reporter.fileBoundary(
            completedBytes: 0,
            totalBytes: totalBytes,
            completedFiles: 0,
            totalFiles: totalFiles)

        // Fetch files through a bounded task group (#853): the sequential
        // loop paid one network round-trip of latency per file, which
        // dominated wall-clock for many-file packs. The first thrown error
        // cancels the group (in-flight `.partial` files resume by byte range
        // on the next attempt). ConcurrentProgress owns the byte/file
        // counters so emissions stay monotonic across out-of-order finishes.
        let progress = ConcurrentProgress(
            reporter: reporter, totalBytes: totalBytes, totalFiles: totalFiles)
        try await withThrowingTaskGroup(of: Void.self) { group in
            var nextIndex = 0
            func addNextTask() {
                guard nextIndex < filesToDownload.count else { return }
                let index = nextIndex
                let file = filesToDownload[index]
                nextIndex += 1
                group.addTask {
                    try await downloadSubdirectoryFile(
                        file, index: index, repo: repo, subdirectory: subdirectory,
                        repoDirectory: repoDirectory, totalFiles: totalFiles,
                        config: config, configuration: configuration, progress: progress)
                }
            }
            for _ in 0..<min(max(1, config.maxConcurrentFiles), totalFiles) {
                addNextTask()
            }
            while try await group.next() != nil {
                addNextTask()
            }
        }

        logger.info("Downloaded \(subdirectory) from \(repo.folderName)")
    }

    /// One file of a subdirectory download, run inside the bounded task group.
    private static func downloadSubdirectoryFile(
        _ file: RemoteFile,
        index: Int,
        repo: Repo,
        subdirectory: String,
        repoDirectory: URL,
        totalFiles: Int,
        config: DownloadConfig,
        configuration: URLSessionConfiguration?,
        progress: ConcurrentProgress
    ) async throws {
        let destPath = repoDirectory.appendingPathComponent(file.path)

        // Only stream live byte progress for files with a known size: an
        // unknown-size file (-1) carries zero weight in totalBytes, so its
        // real bytesWritten would inflate the fraction mid-file and snap
        // back at the boundary. Boundary emits keep progress monotonic.
        let onBytes = file.size > 0 ? progress.liveBytesCallback(fileIndex: index) : nil

        // Fail loudly on blocked paths: subdirectory downloads land in
        // caller-provided directories, so a regular file where a directory
        // belongs is surfaced, never silently deleted.
        let outcome = try await FileDownloader.ensure(
            file: file,
            from: repo.remotePath,
            at: destPath,
            recoveringBlockedPaths: false,
            config: config,
            configuration: configuration,
            onBytes: onBytes
        )

        let completed = progress.fileCompleted(fileIndex: index, size: file.size)
        if outcome != .alreadyPresent, completed % 5 == 0 || completed == totalFiles {
            logger.info("Downloaded \(completed)/\(totalFiles) \(subdirectory) files")
        }
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

    /// File/directory selection rule for repo downloads: subPath scoping,
    /// required-model patterns, metadata-extension allowances, and
    /// all-or-nothing CoreML bundle matching.
    ///
    /// For subPath repos, paths that are (or are inside) a
    /// `.mlmodelc`/`.mlpackage` are decided at bundle granularity against the
    /// required-model patterns:
    ///
    /// - A required bundle is taken whole. The metadata allowance alone
    ///   sweeps in a bundle's `.json`/`.bin` but drops `model.mil`, leaving a
    ///   bundle that passes the `coremldata.bin` existence check yet fails
    ///   MIL load ("Error in reading the MIL network") — StyleTTS2's
    ///   t64/t128/t256 buckets (#821).
    /// - A non-required bundle is skipped whole. These repos publish the
    ///   uncompiled `.mlpackage` next to each compiled `.mlmodelc`, and the
    ///   `.bin` allowance was pulling every `weight.bin` inside them —
    ///   roughly half of a first-run parakeetEou/kokoroAne download was never
    ///   loaded (#826). Skipping applies to directory traversal too, so the
    ///   tree lister does not recurse into excluded bundles.
    static func repoIncludeRule(
        subPath: String?, patterns: [String]
    ) -> (String, Bool) -> Bool {
        { itemPath, isDirectory in
            let isBundlePath =
                itemPath.hasSuffix(".mlmodelc") || itemPath.hasSuffix(".mlpackage")
                || itemPath.contains(".mlmodelc/") || itemPath.contains(".mlpackage/")
            if isDirectory {
                // For subPath repos, only process paths within the subPath
                if let sub = subPath {
                    if isBundlePath, !patterns.isEmpty {
                        return patterns.contains {
                            (itemPath + "/").hasPrefix($0) || $0.hasPrefix(itemPath + "/")
                        }
                    }
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
                if isBundlePath {
                    return isInSubPath && matchesPattern
                }
                let isMetadata =
                    itemPath.hasSuffix(".json") || itemPath.hasSuffix(".model") || itemPath.hasSuffix(".bin")
                return isInSubPath && (matchesPattern || isMetadata)
            }
            return patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                || itemPath.hasSuffix(".json") || itemPath.hasSuffix(".txt")
        }
    }
}
