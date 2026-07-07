import CoreML
import Foundation

/// HuggingFace model downloader using URLSession
public class DownloadUtils {

    private static let logger = AppLogger(category: "DownloadUtils")

    /// Shared URLSession with registry and proxy configuration
    public static let sharedSession: URLSession = ModelRegistry.configuredSession()

    /// Offline-only mode. When true, every public download surface
    /// (`fetchWithAuth`, `downloadRepo`, `downloadSubdirectory`,
    /// `fetchHuggingFaceFile`) and the `loadModels` retry-with-redownload
    /// fallback throws `DownloadUtils.OfflineError` instead of touching
    /// the network. Applications that bundle their own model assets
    /// should set this once at startup and route loading through manual
    /// APIs (e.g. `MLModel(contentsOf:)`, `VadManager(config:vadModel:)`)
    /// so a corrupt-detected `.mlmodelc` never silently re-downloads at
    /// runtime.
    ///
    /// Defaults to `false`. `nonisolated(unsafe)` is acceptable because
    /// the flag is set once at startup before any FluidAudio loaders
    /// are touched and is read-only thereafter.
    nonisolated(unsafe) public static var enforceOffline: Bool = false

    /// Errors thrown when `enforceOffline` is on and FluidAudio would
    /// otherwise attempt a network fetch or a cache rebuild that
    /// requires network. Sibling to `HuggingFaceDownloadError`.
    public enum OfflineError: LocalizedError {
        /// A code path that would have hit the network was blocked.
        /// `operation` is the short tag of the blocked entry point
        /// (e.g. `"downloadRepo(parakeet-tdt-0.6b-v3-coreml)"`).
        case networkDisabled(operation: String)

        /// `loadModels` was invoked but one or more required files are
        /// missing from the local cache. Caller bundled assets but the
        /// bundle was incomplete; surfacing the missing list lets the
        /// caller decide whether to ship a fix or fail loudly.
        case modelMissing(repo: String, missing: [String])

        public var errorDescription: String? {
            switch self {
            case .networkDisabled(let operation):
                return "FluidAudio offline mode: \(operation) blocked"
            case .modelMissing(let repo, let missing):
                return
                    "FluidAudio offline mode: required models missing for \(repo): \(missing.joined(separator: ", "))"
            }
        }
    }

    /// Throws `OfflineError.networkDisabled` if `enforceOffline` is on.
    /// Call this at the top of any path that would touch the network.
    private static func ensureOnlineAllowed(_ operation: String) throws {
        if enforceOffline {
            throw OfflineError.networkDisabled(operation: operation)
        }
    }

    /// Create a URLRequest with optional auth header and timeout (HFClient owns
    /// token resolution and header shape — #765 Wave 2).
    private static func authorizedRequest(
        url: URL, timeout: TimeInterval = DownloadConfig.default.timeout
    ) -> URLRequest {
        HFClient.authorizedRequest(url: url, timeout: timeout)
    }

    /// Fetch data from a URL with HuggingFace authentication if available
    /// Use this for API calls that need auth tokens for private repos or higher rate limits
    public static func fetchWithAuth(from url: URL) async throws -> (Data, URLResponse) {
        try ensureOnlineAllowed("fetchWithAuth(\(url.absoluteString))")
        let request = authorizedRequest(url: url)
        return try await sharedSession.data(for: request)
    }

    /// Forward — the single HTML sniffer lives in HFClient (#765 Wave 2);
    /// kept here because DownloadArtifactValidationTests pins this symbol.
    static func looksLikeHTML(_ data: Data) -> Bool {
        HFClient.looksLikeHTML(data)
    }

    /// Moved to `FluidAudio.HuggingFaceDownloadError` (Shared/Download/DownloadTypes.swift)
    /// so the extracted download primitives don't depend back on this class
    /// (#765 Wave 2). The nested spelling stays source-compatible.
    public typealias HuggingFaceDownloadError = HFDownload.DownloadError

    /// Phase of a model download operation.
    public enum DownloadPhase: Sendable {
        /// Listing files from the remote repository.
        case listing
        /// Downloading model files. `completedFiles` / `totalFiles` track per-file progress.
        case downloading(completedFiles: Int, totalFiles: Int)
        /// Compiling CoreML models after download.
        case compiling(modelName: String)
    }

    /// Progress snapshot passed to ``ProgressHandler`` closures.
    public struct DownloadProgress: Sendable {
        /// Fraction complete in [0, 1].
        public let fractionCompleted: Double
        /// Current phase of the operation.
        public let phase: DownloadPhase

        public init(fractionCompleted: Double, phase: DownloadPhase) {
            self.fractionCompleted = fractionCompleted
            self.phase = phase
        }
    }

    /// Callback type for download progress reporting.
    ///
    /// Called on an unspecified queue. If you need to update UI, dispatch to
    /// the main actor inside your handler.
    public typealias ProgressHandler = @Sendable (DownloadProgress) -> Void

    /// Moved to `FluidAudio.DownloadConfig` (Shared/Download/DownloadTypes.swift);
    /// nested spelling stays source-compatible.
    public typealias DownloadConfig = HFDownload.Config

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
            if enforceOffline {
                logger.warning(
                    "Offline mode: load failed and re-download blocked. \(error.localizedDescription)"
                )
                throw error
            }

            // Cancellation is not corruption. A cancelled first load (app
            // teardown, user abort) must never wipe a valid cache — deleting
            // here threw away fully-downloaded multi-hundred-MB repos.
            if isCancellationError(error) {
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

    /// Forward to `RetryPolicy.isCancellation` (see there for the chain-walk
    /// semantics); kept because DownloadUtilsCancellationTests pins this symbol.
    static func isCancellationError(_ error: Error) -> Bool {
        RetryPolicy.isCancellation(error)
    }

    public static func clearModelCache(forRepo repo: Repo, directory: URL) {
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try? FileManager.default.removeItem(at: repoPath)
    }

    /// Remove all downloaded models and caches.
    ///
    /// Clears both cache locations:
    /// - `~/Library/Application Support/FluidAudio/Models/` (ASR, VAD, Diarization)
    /// - the shared TTS root: `~/.cache/fluidaudio/` on macOS,
    ///   `Application Support/fluidaudio/` on iOS (matches `TtsCacheDirectory`).
    public static func clearAllModelCaches() {
        let fm = FileManager.default

        // ASR, VAD, Diarization models
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let modelsDir = appSupport.appendingPathComponent("FluidAudio/Models")
            try? fm.removeItem(at: modelsDir)
        }

        // TTS models (Kokoro, PocketTTS, Supertonic3, StyleTTS2).
        // Remove the whole `fluidaudio/` root so every backend subdirectory
        // (Models/, voice packs, etc.) is cleared, not just `Models/`.
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
            if enforceOffline {
                let missing = ModelCache.missingModels(at: repoPath, models: effectiveModels)
                logger.error(
                    "Offline mode: required models missing at \(repoPath.path): \(missing)"
                )
                throw OfflineError.modelMissing(repo: repo.folderName, missing: missing)
            }
            logger.info("Models not found in cache at \(repoPath.path)")
            try await downloadRepo(
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
    public static func downloadRepo(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        additionalModelNames: Set<String> = [],
        progressHandler: ProgressHandler? = nil
    ) async throws {
        try await downloadRepo(
            repo, to: directory, variant: variant,
            additionalModelNames: additionalModelNames,
            progressHandler: progressHandler,
            configuration: nil)
    }

    /// Internal seam: `configuration` overrides the session used for tree
    /// listing and per-file downloads so characterization tests can drive the
    /// full listing/filtering/download pipeline with a stub `URLProtocol`
    /// (#765 Wave 1). `nil` (the public path) uses the shared session.
    static func downloadRepo(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        additionalModelNames: Set<String> = [],
        progressHandler: ProgressHandler? = nil,
        configuration: URLSessionConfiguration?
    ) async throws {
        try ensureOnlineAllowed("downloadRepo(\(repo.folderName))")
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let listingSession = configuration.map { URLSession(configuration: $0) } ?? sharedSession
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

            // Pinned asymmetry vs downloadSubdirectory: cached/empty files
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

    // MARK: - Test-pinned forwards (#765)

    /// Forwards to the Wave 4 primitives — kept because the download-resume,
    /// artifact-validation, and retry characterization suites pin these
    /// symbols on DownloadUtils.
    static func isRetryableDownloadError(_ error: Error) -> Bool {
        RetryPolicy.isRetryable(error)
    }

    static func validateDownloadedArtifact(
        at tempURL: URL,
        response: HTTPURLResponse?,
        path: String,
        expectedSize: Int
    ) throws {
        try FileDownloader.validateDownloadedArtifact(
            at: tempURL, response: response, path: path, expectedSize: expectedSize)
    }

    static func resumeValidatorURL(for partialFileURL: URL) -> URL {
        FileDownloader.resumeValidatorURL(for: partialFileURL)
    }

    static func downloadFileWithRetry(
        request: URLRequest,
        path: String,
        expectedSize: Int,
        partialFileURL: URL? = nil,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0,
        configuration: URLSessionConfiguration? = nil
    ) async throws -> URL {
        try await FileDownloader.download(
            request: request, path: path, expectedSize: expectedSize,
            partialFileURL: partialFileURL, onProgress: onProgress,
            maxAttempts: maxAttempts, minBackoff: minBackoff,
            configuration: configuration)
    }

    /// Forward to ProgressReporter (kept: DownloadUtilsProgressTests pins it).
    static func subdirectoryProgressFraction(
        completedBytes: Int64,
        totalBytes: Int64,
        completedFiles: Int,
        totalFiles: Int
    ) -> Double {
        ProgressReporter.downloadFraction(
            completedBytes: completedBytes, totalBytes: totalBytes,
            completedFiles: completedFiles, totalFiles: totalFiles)
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
    public static func downloadSubdirectory(
        _ repo: Repo,
        subdirectory: String,
        to repoDirectory: URL,
        progressHandler: ProgressHandler? = nil,
        shouldSkip: (@Sendable (String) -> Bool)? = nil
    ) async throws {
        try ensureOnlineAllowed("downloadSubdirectory(\(repo.folderName)/\(subdirectory))")
        // Subdirectory downloads have no compile phase: download spans 0-1.
        let reporter = ProgressReporter(handler: progressHandler, downloadPhaseWeight: 1.0)
        reporter.listing()
        let filesToDownload: [RemoteFile] = try await HFTreeLister.listTree(
            repoRemotePath: repo.remotePath,
            startingAt: subdirectory,
            include: { itemPath, _ in shouldSkip?(itemPath) != true },
            fetch: HFTreeLister.fetch(using: sharedSession)
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

    /// Fetch a single file from HuggingFace with retry
    public static func fetchHuggingFaceFile(
        from url: URL,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0
    ) async throws -> Data {
        try ensureOnlineAllowed("fetchHuggingFaceFile(\(description))")
        var lastError: Error?
        let request = authorizedRequest(url: url)

        for attempt in 1...maxAttempts {
            do {
                let (data, response) = try await sharedSession.data(for: request)

                guard let httpResponse = response as? HTTPURLResponse else {
                    throw HuggingFaceDownloadError.invalidResponse
                }

                try HFClient.checkRateLimit(httpResponse, context: "fetching \(description)")

                guard (200..<300).contains(httpResponse.statusCode) else {
                    throw HuggingFaceDownloadError.invalidResponse
                }

                return data

            } catch {
                lastError = error
                if attempt < maxAttempts {
                    let backoffSeconds = pow(2.0, Double(attempt - 1)) * minBackoff
                    logger.warning(
                        "Download attempt \(attempt) for \(description) failed: \(error.localizedDescription). Retrying in \(String(format: "%.1f", backoffSeconds))s."
                    )
                    try await Task.sleep(nanoseconds: UInt64(backoffSeconds * 1_000_000_000))
                }
            }
        }

        throw lastError ?? HuggingFaceDownloadError.invalidResponse
    }
}
