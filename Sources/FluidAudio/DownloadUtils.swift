import CoreML
import Foundation
import os

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

    /// Get HuggingFace token from environment if available.
    /// Supports multiple env vars for compatibility with different HuggingFace tools:
    /// - HF_TOKEN: Official HuggingFace CLI
    /// - HUGGING_FACE_HUB_TOKEN: Python huggingface_hub library
    /// - HUGGINGFACEHUB_API_TOKEN: LangChain and older integrations
    private static var huggingFaceToken: String? {
        ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGINGFACEHUB_API_TOKEN"]
    }

    /// Create a URLRequest with optional auth header and timeout
    private static func authorizedRequest(
        url: URL, timeout: TimeInterval = DownloadConfig.default.timeout
    ) -> URLRequest {
        var request = URLRequest(url: url, timeoutInterval: timeout)
        if let token = huggingFaceToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    /// Fetch data from a URL with HuggingFace authentication if available
    /// Use this for API calls that need auth tokens for private repos or higher rate limits
    public static func fetchWithAuth(from url: URL) async throws -> (Data, URLResponse) {
        try ensureOnlineAllowed("fetchWithAuth(\(url.absoluteString))")
        let request = authorizedRequest(url: url)
        return try await sharedSession.data(for: request)
    }

    /// Validate that response data is JSON, not HTML error page
    /// HuggingFace sometimes returns 200 OK with HTML error pages during rate limiting/timeouts
    private static func validateJSONResponse(_ data: Data, path: String) throws {
        // Check if response starts with HTML markers
        if let responseString = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) {
            if responseString.hasPrefix("<") || responseString.lowercased().contains("<!doctype html") {
                let snippet = String(responseString.prefix(100))
                throw HuggingFaceDownloadError.htmlErrorResponse(path: path, snippet: snippet)
            }
        }
    }

    static func looksLikeHTML(_ data: Data) -> Bool {
        let prefix = data.prefix(512)
        let text = String(data: prefix, encoding: .utf8) ?? String(decoding: prefix, as: UTF8.self)
        let lowered = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        return lowered.hasPrefix("<!doctype html") || lowered.hasPrefix("<html") || lowered.hasPrefix("<?xml")
    }

    /// `response` is nil when re-validating a fully-downloaded partial file from
    /// a previous run (no live response to check Content-Type against).
    static func validateDownloadedArtifact(
        at tempURL: URL,
        response: HTTPURLResponse?,
        path: String,
        expectedSize: Int
    ) throws {
        if let contentType = response?.value(forHTTPHeaderField: "Content-Type")?.lowercased(),
            contentType.contains("text/html")
        {
            throw HuggingFaceDownloadError.invalidArtifact(
                path: path, reason: "server returned Content-Type: \(contentType)")
        }

        let actualSize =
            ((try? FileManager.default.attributesOfItem(atPath: tempURL.path))?[.size] as? Int) ?? 0
        if actualSize == 0 {
            throw HuggingFaceDownloadError.invalidArtifact(path: path, reason: "empty file")
        }

        if let handle = try? FileHandle(forReadingFrom: tempURL) {
            defer { try? handle.close() }
            if looksLikeHTML(handle.readData(ofLength: 512)) {
                throw HuggingFaceDownloadError.invalidArtifact(
                    path: path, reason: "response body begins with HTML markup")
            }
        }

        // HuggingFace reports the exact (LFS-resolved) object size; a short body is truncation.
        if expectedSize > 0 && actualSize != expectedSize {
            throw HuggingFaceDownloadError.invalidArtifact(
                path: path,
                reason: "size mismatch (expected \(expectedSize) bytes, got \(actualSize))")
        }
    }

    public enum HuggingFaceDownloadError: LocalizedError {
        case invalidResponse
        case rateLimited(statusCode: Int, message: String)
        case downloadFailed(path: String, underlying: Error)
        case modelNotFound(path: String)
        case htmlErrorResponse(path: String, snippet: String)
        case invalidArtifact(path: String, reason: String)

        public var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "Received an invalid response from Hugging Face."
            case .rateLimited(_, let message):
                return "Hugging Face rate limit encountered: \(message)"
            case .downloadFailed(let path, let underlying):
                return "Failed to download \(path): \(underlying.localizedDescription)"
            case .htmlErrorResponse(let path, let snippet):
                return "HuggingFace returned HTML instead of JSON for \(path) (rate limit or server issue): \(snippet)"
            case .modelNotFound(let path):
                return "Model file not found: \(path)"
            case .invalidArtifact(let path, let reason):
                return "Downloaded artifact for \(path) is invalid (\(reason)); refusing to cache it."
            }
        }
    }

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

    public struct DownloadConfig: Sendable {
        public let timeout: TimeInterval

        public init(timeout: TimeInterval = 1800) {  // 30 minutes for large models
            self.timeout = timeout
        }

        public static let `default` = DownloadConfig()
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

            // Try to delete the corrupted cache
            do {
                try FileManager.default.removeItem(at: repoPath)
                logger.info("Successfully deleted corrupted cache at \(repoPath.path)")
            } catch {
                // If deletion fails (excluding "file not found"), log the error but continue
                // Robust directory creation will handle any remaining files
                let nsError = error as NSError
                if nsError.domain == NSCocoaErrorDomain && nsError.code == NSFileNoSuchFileError {
                    // File already doesn't exist - this is fine
                } else {
                    logger.warning("Failed to delete cache: \(error.localizedDescription)")
                    logger.info("Will attempt to overwrite during re-download")
                }
            }

            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        }
    }

    /// `true` when `error` represents cancellation (Swift `CancellationError`,
    /// `NSURLErrorCancelled`, or `NSUserCancelledError`, at any depth of the
    /// underlying-error chain) rather than a corrupted cache.
    ///
    /// The `NSUnderlyingErrorKey` chain is walked to its end. Visited errors
    /// are tracked by identity so a self-referential chain terminates without
    /// an arbitrary depth cap.
    static func isCancellationError(_ error: Error) -> Bool {
        if error is CancellationError { return true }

        var current: NSError? = error as NSError
        var visited: Set<ObjectIdentifier> = []
        while let nsError = current, visited.insert(ObjectIdentifier(nsError)).inserted {
            if nsError.domain == NSURLErrorDomain, nsError.code == NSURLErrorCancelled {
                return true
            }
            if nsError.domain == NSCocoaErrorDomain, nsError.code == NSUserCancelledError {
                return true
            }
            current = nsError.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return false
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
        let allModelsExist = effectiveModels.allSatisfy { model in
            let modelPath = repoPath.appendingPathComponent(model)
            return FileManager.default.fileExists(atPath: modelPath.path)
        }

        if !allModelsExist {
            // In offline mode surface a typed error listing the
            // missing files instead of attempting a HuggingFace fetch.
            if enforceOffline {
                let missing = effectiveModels.filter { name in
                    !FileManager.default.fileExists(atPath: repoPath.appendingPathComponent(name).path)
                }.sorted()
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
            progressHandler?(
                DownloadProgress(fractionCompleted: 0.5, phase: .downloading(completedFiles: 0, totalFiles: 0)))
        }

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.allowLowPrecisionAccumulationOnGPU = true

        var models: [String: MLModel] = [:]
        for (index, name) in modelNames.enumerated() {
            let modelPath = repoPath.appendingPathComponent(name)
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

            progressHandler?(
                DownloadProgress(
                    fractionCompleted: 0.5 + 0.5 * Double(index) / Double(modelNames.count),
                    phase: .compiling(modelName: name)
                ))

            let start = Date()
            let model = try MLModel(contentsOf: modelPath, configuration: config)
            let elapsed = Date().timeIntervalSince(start)

            models[name] = model

            let ms = elapsed * 1000
            let formatted = String(format: "%.2f", ms)
            logger.info("Compiled model \(name) in \(formatted) ms :: \(SystemInfo.summary())")
        }

        progressHandler?(DownloadProgress(fractionCompleted: 1.0, phase: .compiling(modelName: "")))
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

        // Get all files recursively using HuggingFace API
        var filesToDownload: [(path: String, size: Int)] = []

        func listDirectory(path: String) async throws {
            let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
            let dirURL = try ModelRegistry.apiModels(repo.remotePath, apiPath)
            let request = authorizedRequest(url: dirURL)

            let (dirData, response) = try await listingSession.data(for: request)

            if let httpResponse = response as? HTTPURLResponse {
                if httpResponse.statusCode == 429 || httpResponse.statusCode == 503 {
                    throw HuggingFaceDownloadError.rateLimited(
                        statusCode: httpResponse.statusCode, message: "Rate limited while listing files")
                }
            }

            // Validate that response is JSON, not HTML error page
            try validateJSONResponse(dirData, path: path)

            guard let items = try JSONSerialization.jsonObject(with: dirData) as? [[String: Any]] else {
                throw HuggingFaceDownloadError.invalidResponse
            }

            for item in items {
                guard let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String
                else { continue }

                if itemType == "directory" {
                    // For subPath repos, only process paths within the subPath
                    let shouldProcess: Bool
                    if let sub = subPath {
                        shouldProcess =
                            itemPath == sub || itemPath.hasPrefix("\(sub)/")
                            || patterns.contains { itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/") }
                    } else {
                        shouldProcess =
                            patterns.isEmpty
                            || patterns.contains { itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/") }
                    }
                    if shouldProcess {
                        try await listDirectory(path: itemPath)
                    }
                } else if itemType == "file" {
                    // For subPath repos, only include files within the subPath
                    let shouldInclude: Bool
                    if let sub = subPath {
                        let isInSubPath = itemPath.hasPrefix("\(sub)/")
                        let matchesPattern =
                            patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                        let isMetadata =
                            itemPath.hasSuffix(".json") || itemPath.hasSuffix(".model") || itemPath.hasSuffix(".bin")
                        shouldInclude = isInSubPath && (matchesPattern || isMetadata)
                    } else {
                        shouldInclude =
                            patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                            || itemPath.hasSuffix(".json") || itemPath.hasSuffix(".txt")
                    }
                    if shouldInclude {
                        let fileSize = item["size"] as? Int ?? -1
                        filesToDownload.append((path: itemPath, size: fileSize))
                    }
                }
            }
        }

        // Pull root-level files whose basename is in `names`. Some subPath repos
        // keep shared auxiliary files at the repo root rather than inside the
        // precision subdirectory, so a subPath-only traversal misses them.
        func listRootFiles(matching names: Set<String>) async throws {
            let dirURL = try ModelRegistry.apiModels(repo.remotePath, "tree/main")
            let request = authorizedRequest(url: dirURL)
            let (dirData, response) = try await listingSession.data(for: request)

            if let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 429 || httpResponse.statusCode == 503
            {
                throw HuggingFaceDownloadError.rateLimited(
                    statusCode: httpResponse.statusCode, message: "Rate limited while listing root files")
            }

            try validateJSONResponse(dirData, path: "")

            guard let items = try JSONSerialization.jsonObject(with: dirData) as? [[String: Any]] else {
                throw HuggingFaceDownloadError.invalidResponse
            }

            for item in items {
                guard let itemPath = item["path"] as? String,
                    item["type"] as? String == "file",
                    names.contains((itemPath as NSString).lastPathComponent)
                else { continue }
                let fileSize = item["size"] as? Int ?? -1
                filesToDownload.append((path: itemPath, size: fileSize))
            }
        }

        // Start listing from subPath if specified, otherwise from root
        progressHandler?(DownloadProgress(fractionCompleted: 0.0, phase: .listing))
        try await listDirectory(path: subPath ?? "")

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
                try await listRootFiles(matching: Set(missingAux))
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

            // Skip if already exists
            if FileManager.default.fileExists(atPath: destPath.path) {
                completedBytes += Int64(max(0, file.size))
                continue
            }

            // Create parent directory, removing any conflicting files in the path
            let parentDir = destPath.deletingLastPathComponent()
            try createDirectoryRobustly(at: parentDir)

            // HuggingFace returns 500 for 0-byte files — create empty file locally
            if file.size == 0 {
                FileManager.default.createFile(atPath: destPath.path, contents: Data())
                continue
            }

            // Download file (use original path for HuggingFace URL)
            let encodedFilePath =
                file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            let fileURL = try ModelRegistry.resolveModel(repo.remotePath, encodedFilePath)
            let request = authorizedRequest(url: fileURL)

            // Bounded retry on transient failures, with byte-range resume via a
            // persistent `<dest>.partial` file so retries and re-runs continue a
            // partially-downloaded file instead of restarting it (#757).
            let onProgress: (@Sendable (Int64, Int64) -> Void)?
            if let handler = progressHandler {
                let baseBytes = completedBytes
                let fileCount = filesToDownload.count
                let totalBytesSnapshot = totalBytes
                let fileIndex = index
                onProgress = { bytesWritten, _ in
                    guard totalBytesSnapshot > 0 else { return }
                    let current = baseBytes + bytesWritten
                    // Download phase occupies 0.0–0.5 of the overall range.
                    let fraction = 0.5 * Double(current) / Double(totalBytesSnapshot)
                    handler(
                        DownloadProgress(
                            fractionCompleted: min(fraction, 0.5),
                            phase: .downloading(completedFiles: fileIndex, totalFiles: fileCount)
                        ))
                }
            } else {
                onProgress = nil
            }

            let tempFileURL = try await downloadFileWithRetry(
                request: request,
                path: file.path,
                expectedSize: file.size,
                partialFileURL: destPath.appendingPathExtension("partial"),
                onProgress: onProgress,
                configuration: configuration
            )

            // Move downloaded file to destination
            if FileManager.default.fileExists(atPath: destPath.path) {
                try? FileManager.default.removeItem(at: destPath)
            }
            try FileManager.default.moveItem(at: tempFileURL, to: destPath)

            completedBytes += Int64(max(0, file.size))

            if (index + 1) % 10 == 0 || index == filesToDownload.count - 1 {
                logger.info("Downloaded \(index + 1)/\(filesToDownload.count) files")
            }

            progressHandler?(
                DownloadProgress(
                    fractionCompleted: totalBytes > 0
                        ? 0.5 * Double(completedBytes) / Double(totalBytes)
                        : 0.5 * Double(index + 1) / Double(filesToDownload.count),
                    phase: .downloading(completedFiles: index + 1, totalFiles: filesToDownload.count)
                ))
        }

        // Verify required models are present
        for model in requiredModels {
            let modelPath = repoPath.appendingPathComponent(model)
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw HuggingFaceDownloadError.modelNotFound(path: model)
            }
        }

        logger.info("Downloaded all required models for \(repo.folderName)")
    }

    // MARK: - Helper Functions

    /// Robustly create a directory, removing any conflicting files in the path.
    ///
    /// This handles cases where a file exists where a directory should be, which can happen
    /// during corrupted cache recovery when partial deletion leaves files in place of directories.
    ///
    /// - Parameter url: The directory path to create
    /// - Throws: Errors from FileManager if directory creation fails after cleanup
    private static func createDirectoryRobustly(at url: URL) throws {
        let fm = FileManager.default
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

    // MARK: - Streaming download to a persistent file

    /// Stream a download directly into `destination` so received bytes survive
    /// a mid-transfer drop (#757). Pure transport: the caller sets resume
    /// headers and validates the HTTP status. Body bytes are written only for
    /// 2xx — appended after `resumeOffset` on 206, from byte 0 otherwise.
    ///
    /// - Parameters:
    ///   - resumeOffset: Byte count a 206 appends after; also the base for
    ///     progress callbacks so resumed progress never dips.
    ///   - configuration: Session configuration override for tests.
    ///   - onProgress: `(totalBytesWritten, totalBytesExpected)`, both
    ///     including `resumeOffset`. Delegate-driven byte progress (#756).
    ///   - onResponse: Fires before any body byte is written — the resume path
    ///     persists the validator here so it survives a drop mid-body.
    /// Internal (not private) so download-resume tests can drive it directly.
    static func streamDownload(
        request: URLRequest,
        to destination: URL,
        resumeOffset: Int64 = 0,
        configuration: URLSessionConfiguration? = nil,
        onProgress: (@Sendable (Int64, Int64) -> Void)? = nil,
        onResponse: (@Sendable (HTTPURLResponse) -> Void)? = nil
    ) async throws -> HTTPURLResponse {
        let delegate = StreamingDownloadDelegate(
            destination: destination,
            resumeOffset: resumeOffset,
            onProgress: onProgress,
            onResponse: onResponse
        )
        // Dedicated session with delegate — one per download to avoid cross-talk.
        let session = URLSession(
            configuration: configuration ?? sharedSession.configuration,
            delegate: delegate,
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }

        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                let task = session.dataTask(with: request)
                delegate.attach(continuation: continuation, task: task)
                task.resume()
            }
        } onCancel: {
            delegate.cancel()
        }
    }

    // MARK: - Per-file download with bounded retry and byte-range resume

    /// Sidecar storing the HTTP validator (ETag/Last-Modified) a partial file
    /// came from; written at response time so it survives a drop mid-body.
    static func resumeValidatorURL(for partialFileURL: URL) -> URL {
        partialFileURL.appendingPathExtension("etag")
    }

    private static func fileSize(at url: URL) -> Int64 {
        ((try? FileManager.default.attributesOfItem(atPath: url.path))?[.size] as? Int64) ?? 0
    }

    /// Remove a partial download and its validator sidecar.
    private static func clearPartialDownload(_ partialFileURL: URL) {
        try? FileManager.default.removeItem(at: partialFileURL)
        try? FileManager.default.removeItem(at: resumeValidatorURL(for: partialFileURL))
    }

    /// Resume validator for `If-Range`: a strong ETag, else Last-Modified.
    /// Weak ETags are unusable for `If-Range` (RFC 9110 §13.1.5); nil means
    /// resume is unsafe and the file restarts from 0.
    private static func resumeValidator(from response: HTTPURLResponse) -> String? {
        if let etag = response.value(forHTTPHeaderField: "ETag"), !etag.hasPrefix("W/") {
            return etag
        }
        return response.value(forHTTPHeaderField: "Last-Modified")
    }

    /// Download a single repo file with bounded exponential-backoff retry on
    /// transient failures (timeout/TLS/connectivity, HTTP 429/503/5xx; 4xx and
    /// non-network errors fail fast — see `isRetryableDownloadError`).
    ///
    /// Resume (#757): bytes stream into `partialFileURL`, so a retry — or a new
    /// process — continues with `Range`/`If-Range` instead of restarting from
    /// byte 0. A 200 (range ignored / remote changed) restarts the file rather
    /// than splicing mixed-version bytes.
    ///
    /// - Parameters:
    ///   - partialFileURL: Where in-flight bytes live (e.g. `<dest>.partial`).
    ///     Pass nil for a unique temp location (in-run resume only).
    ///   - configuration: Session configuration override for tests.
    /// - Returns: The URL of a validated (2xx) fully-written download; the
    ///   caller moves it into the cache.
    /// Internal (not private) so download-resume tests can drive it directly.
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
        let defaultPartialURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("partial")
        let partialURL = partialFileURL ?? defaultPartialURL
        let validatorURL = resumeValidatorURL(for: partialURL)
        var lastError: Error?

        for attempt in 1...maxAttempts {
            do {
                var attemptRequest = request
                var resumeOffset: Int64 = 0

                let partialSize = fileSize(at: partialURL)
                let validator = (try? String(contentsOf: validatorURL, encoding: .utf8))?
                    .trimmingCharacters(in: .whitespacesAndNewlines)

                if partialSize > 0, let validator, !validator.isEmpty {
                    if expectedSize > 0 && partialSize >= Int64(expectedSize) {
                        // A previous run finished the body but didn't get to move
                        // the file. Validate and reuse it instead of re-fetching.
                        if (try? validateDownloadedArtifact(
                            at: partialURL, response: nil, path: path, expectedSize: expectedSize))
                            != nil
                        {
                            try? FileManager.default.removeItem(at: validatorURL)
                            return partialURL
                        }
                        clearPartialDownload(partialURL)
                    } else {
                        attemptRequest.setValue("bytes=\(partialSize)-", forHTTPHeaderField: "Range")
                        attemptRequest.setValue(validator, forHTTPHeaderField: "If-Range")
                        resumeOffset = partialSize
                        logger.info(
                            "Resuming \(path) from byte \(partialSize) (attempt \(attempt))")
                    }
                } else if partialSize > 0 {
                    // Partial bytes with no validator can't be safely resumed.
                    clearPartialDownload(partialURL)
                }

                let httpResponse = try await streamDownload(
                    request: attemptRequest,
                    to: partialURL,
                    resumeOffset: resumeOffset,
                    configuration: configuration,
                    onProgress: onProgress,
                    onResponse: { response in
                        // Persist the validator as soon as the fresh body starts, so
                        // a drop mid-body still leaves it for the next attempt. A 206
                        // keeps the validator of the response it is extending.
                        guard response.statusCode != 206, (200..<300).contains(response.statusCode)
                        else { return }
                        if let validator = resumeValidator(from: response) {
                            try? validator.write(to: validatorURL, atomically: true, encoding: .utf8)
                        } else {
                            try? FileManager.default.removeItem(at: validatorURL)
                        }
                    }
                )

                if httpResponse.statusCode == 429 || httpResponse.statusCode == 503 {
                    throw HuggingFaceDownloadError.rateLimited(
                        statusCode: httpResponse.statusCode,
                        message: "Rate limited while downloading \(path)")
                }

                if httpResponse.statusCode == 416 {
                    // Range not satisfiable — the partial is bogus (or the remote
                    // shrank). Restart from 0 on the next attempt.
                    clearPartialDownload(partialURL)
                    throw HuggingFaceDownloadError.invalidArtifact(
                        path: path, reason: "server rejected resume range (416)")
                }

                guard (200..<300).contains(httpResponse.statusCode) else {
                    throw HuggingFaceDownloadError.downloadFailed(
                        path: path,
                        underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                    )
                }

                // Validate before the caller moves the file into the cache.
                do {
                    try validateDownloadedArtifact(
                        at: partialURL, response: httpResponse, path: path,
                        expectedSize: expectedSize)
                } catch {
                    // A completed-but-invalid body must not be resumed from.
                    clearPartialDownload(partialURL)
                    throw error
                }

                try? FileManager.default.removeItem(at: validatorURL)
                return partialURL
            } catch {
                lastError = error
                guard attempt < maxAttempts, isRetryableDownloadError(error) else {
                    throw error
                }
                let backoffSeconds = pow(2.0, Double(attempt - 1)) * minBackoff
                logger.warning(
                    "Download attempt \(attempt) for \(path) failed: \(error.localizedDescription). Retrying in \(String(format: "%.1f", backoffSeconds))s."
                )
                try await Task.sleep(nanoseconds: UInt64(backoffSeconds * 1_000_000_000))
            }
        }

        throw lastError ?? HuggingFaceDownloadError.invalidResponse
    }

    /// Classify a per-file download error as transient (worth retrying) or
    /// permanent. Transient: URLSession timeout / TLS / connectivity failures and
    /// HTTP 429/503/5xx. Everything else (404/other 4xx, invalid response,
    /// non-network errors) is permanent.
    ///
    /// Internal (not private) so retry-policy characterization tests can pin this
    /// classification ahead of the #765 refactor.
    static func isRetryableDownloadError(_ error: Error) -> Bool {
        if let urlError = error as? URLError {
            switch urlError.code {
            case .timedOut, .cannotConnectToHost, .cannotFindHost,
                .networkConnectionLost, .notConnectedToInternet,
                .dnsLookupFailed, .secureConnectionFailed,
                .resourceUnavailable:
                return true
            default:
                return false
            }
        }

        switch error {
        case HuggingFaceDownloadError.rateLimited:
            return true
        case HuggingFaceDownloadError.invalidArtifact:
            // Usually a transient unhealthy network path (proxy, mirror 5xx) — retry.
            return true
        case HuggingFaceDownloadError.downloadFailed(_, let underlying):
            let nsError = underlying as NSError
            return nsError.domain == "HTTP" && (500...599).contains(nsError.code)
        default:
            return false
        }
    }

    static func subdirectoryProgressFraction(
        completedBytes: Int64,
        totalBytes: Int64,
        completedFiles: Int,
        totalFiles: Int
    ) -> Double {
        guard totalFiles > 0 else { return 1.0 }
        guard totalBytes > 0 else {
            return min(Double(completedFiles) / Double(totalFiles), 1.0)
        }

        return min(Double(completedBytes) / Double(totalBytes), 1.0)
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
        progressHandler?(DownloadProgress(fractionCompleted: 0.0, phase: .listing))
        var filesToDownload: [(path: String, size: Int)] = []

        func listFiles(at path: String) async throws {
            let dirURL = try ModelRegistry.apiModels(repo.remotePath, "tree/main/\(path)")
            let (dirData, response) = try await fetchWithAuth(from: dirURL)
            if let httpResponse = response as? HTTPURLResponse,
                httpResponse.statusCode == 429 || httpResponse.statusCode == 503
            {
                throw HuggingFaceDownloadError.rateLimited(
                    statusCode: httpResponse.statusCode,
                    message: "Rate limited while listing files in \(path)")
            }

            // Validate that response is JSON, not HTML error page
            try validateJSONResponse(dirData, path: path)

            guard let items = try JSONSerialization.jsonObject(with: dirData) as? [[String: Any]] else {
                throw HuggingFaceDownloadError.invalidResponse
            }
            for item in items {
                guard let itemPath = item["path"] as? String,
                    let itemType = item["type"] as? String
                else { continue }

                if shouldSkip?(itemPath) == true {
                    continue
                }

                if itemType == "directory" {
                    try await listFiles(at: itemPath)
                } else if itemType == "file" {
                    let fileSize = item["size"] as? Int ?? -1
                    filesToDownload.append((path: itemPath, size: fileSize))
                }
            }
        }

        try await listFiles(at: subdirectory)
        let totalFiles = filesToDownload.count
        logger.info("Found \(totalFiles) files in \(subdirectory)")

        // Compute total known bytes for byte-weighted progress.
        // Files with unknown sizes (size == -1) are treated as 0 for weighting.
        let totalBytes: Int64 = filesToDownload.reduce(0) { $0 + Int64(max(0, $1.size)) }
        var completedBytes: Int64 = 0

        progressHandler?(
            DownloadProgress(
                fractionCompleted: subdirectoryProgressFraction(
                    completedBytes: completedBytes,
                    totalBytes: totalBytes,
                    completedFiles: 0,
                    totalFiles: totalFiles
                ),
                phase: .downloading(completedFiles: 0, totalFiles: totalFiles)))

        for (index, file) in filesToDownload.enumerated() {
            let destPath = repoDirectory.appendingPathComponent(file.path)

            if FileManager.default.fileExists(atPath: destPath.path) {
                completedBytes += Int64(max(0, file.size))
                progressHandler?(
                    DownloadProgress(
                        fractionCompleted: subdirectoryProgressFraction(
                            completedBytes: completedBytes,
                            totalBytes: totalBytes,
                            completedFiles: index + 1,
                            totalFiles: totalFiles
                        ),
                        phase: .downloading(
                            completedFiles: index + 1, totalFiles: totalFiles)))
                continue
            }

            try FileManager.default.createDirectory(
                at: destPath.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            if file.size == 0 {
                FileManager.default.createFile(atPath: destPath.path, contents: Data())
                completedBytes += Int64(max(0, file.size))
                progressHandler?(
                    DownloadProgress(
                        fractionCompleted: subdirectoryProgressFraction(
                            completedBytes: completedBytes,
                            totalBytes: totalBytes,
                            completedFiles: index + 1,
                            totalFiles: totalFiles
                        ),
                        phase: .downloading(
                            completedFiles: index + 1, totalFiles: totalFiles)))
                if (index + 1) % 5 == 0 || index == totalFiles - 1 {
                    logger.info("Downloaded \(index + 1)/\(totalFiles) \(subdirectory) files")
                }
                continue
            }

            let encodedPath =
                file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            let fileURL = try ModelRegistry.resolveModel(repo.remotePath, encodedPath)
            let request = authorizedRequest(url: fileURL)

            let onProgress: (@Sendable (Int64, Int64) -> Void)?
            // Only stream live byte progress for files with a known size: an
            // unknown-size file (-1) carries zero weight in totalBytes, so its
            // real bytesWritten would inflate the fraction mid-file and snap
            // back at the boundary. Boundary emits keep progress monotonic.
            if let handler = progressHandler, file.size > 0 {
                let baseBytes = completedBytes
                let totalBytesSnapshot = totalBytes
                let fileIndex = index
                let fileCount = totalFiles
                onProgress = { bytesWritten, _ in
                    guard totalBytesSnapshot > 0 else { return }
                    let currentBytes = baseBytes + bytesWritten
                    let fraction = subdirectoryProgressFraction(
                        completedBytes: currentBytes,
                        totalBytes: totalBytesSnapshot,
                        completedFiles: fileIndex,
                        totalFiles: fileCount
                    )
                    handler(
                        DownloadProgress(
                            fractionCompleted: fraction,
                            phase: .downloading(completedFiles: fileIndex, totalFiles: fileCount)
                        ))
                }
            } else {
                onProgress = nil
            }

            // downloadFileWithRetry validates the artifact (HTML error pages,
            // truncated bodies) internally, and the persistent `.partial` file
            // gives retries and re-runs byte-range resume, same as downloadRepo.
            let tempURL = try await downloadFileWithRetry(
                request: request,
                path: file.path,
                expectedSize: file.size,
                partialFileURL: destPath.appendingPathExtension("partial"),
                onProgress: onProgress
            )

            if FileManager.default.fileExists(atPath: destPath.path) {
                try? FileManager.default.removeItem(at: destPath)
            }
            try FileManager.default.moveItem(at: tempURL, to: destPath)

            completedBytes += Int64(max(0, file.size))

            progressHandler?(
                DownloadProgress(
                    fractionCompleted: subdirectoryProgressFraction(
                        completedBytes: completedBytes,
                        totalBytes: totalBytes,
                        completedFiles: index + 1,
                        totalFiles: totalFiles
                    ),
                    phase: .downloading(
                        completedFiles: index + 1, totalFiles: totalFiles)))

            if (index + 1) % 5 == 0 || index == totalFiles - 1 {
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

                if httpResponse.statusCode == 429 || httpResponse.statusCode == 503 {
                    throw HuggingFaceDownloadError.rateLimited(
                        statusCode: httpResponse.statusCode,
                        message: "HTTP \(httpResponse.statusCode)"
                    )
                }

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

// MARK: - URLSession download delegate for byte-level progress

/// `URLSessionDownloadTask` → async/await bridge that forwards byte progress (#756).
/// Streams response bytes straight into a destination file so partial content
/// survives connection drops (#757). Appends after `resumeOffset` on HTTP 206;
/// truncates and writes from 0 on any other 2xx; discards the body otherwise.
private final class StreamingDownloadDelegate: NSObject, URLSessionDataDelegate, Sendable {
    private struct State {
        var continuation: CheckedContinuation<HTTPURLResponse, Error>?
        var task: URLSessionDataTask?
        var handle: FileHandle?
        var bytesWritten: Int64 = 0
        var expectedTotal: Int64 = -1
        var writeError: Error?
        var finished = false
    }

    private let destination: URL
    private let resumeOffset: Int64
    private let onProgress: (@Sendable (Int64, Int64) -> Void)?
    private let onResponse: (@Sendable (HTTPURLResponse) -> Void)?
    // Holds the non-Sendable continuation and file handle; the lock (not
    // `@unchecked`) makes the delegate Sendable as URLSession requires.
    private let state = OSAllocatedUnfairLock<State>(uncheckedState: State())

    init(
        destination: URL,
        resumeOffset: Int64,
        onProgress: (@Sendable (Int64, Int64) -> Void)?,
        onResponse: (@Sendable (HTTPURLResponse) -> Void)?
    ) {
        self.destination = destination
        self.resumeOffset = resumeOffset
        self.onProgress = onProgress
        self.onResponse = onResponse
    }

    /// Register the continuation and task before the task starts.
    func attach(
        continuation: CheckedContinuation<HTTPURLResponse, Error>,
        task: URLSessionDataTask
    ) {
        state.withLockUnchecked {
            $0.continuation = continuation
            $0.task = task
        }
    }

    func cancel() {
        state.withLockUnchecked { $0.task }?.cancel()
    }

    func urlSession(
        _ session: URLSession,
        dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping (URLSession.ResponseDisposition) -> Void
    ) {
        guard let http = response as? HTTPURLResponse else {
            completionHandler(.cancel)
            return
        }
        onResponse?(http)

        // Only 2xx bodies are file content worth keeping; error bodies are
        // discarded so a 503 page can never overwrite resumable bytes.
        if (200..<300).contains(http.statusCode) {
            let appending = http.statusCode == 206
            do {
                if !FileManager.default.fileExists(atPath: destination.path) {
                    FileManager.default.createFile(atPath: destination.path, contents: nil)
                }
                let handle = try FileHandle(forWritingTo: destination)
                if appending {
                    try handle.seekToEnd()
                } else {
                    try handle.truncate(atOffset: 0)
                }
                let base = appending ? resumeOffset : 0
                // Content-Range carries the full object size on a 206; a plain
                // 200 reports the full size via expectedContentLength.
                let expectedTotal: Int64
                if let contentRange = http.value(forHTTPHeaderField: "Content-Range"),
                    let totalPart = contentRange.split(separator: "/").last,
                    let total = Int64(totalPart)
                {
                    expectedTotal = total
                } else if http.expectedContentLength >= 0 {
                    expectedTotal = base + http.expectedContentLength
                } else {
                    expectedTotal = -1
                }
                state.withLockUnchecked {
                    $0.handle = handle
                    $0.bytesWritten = base
                    $0.expectedTotal = expectedTotal
                }
            } catch {
                state.withLockUnchecked { $0.writeError = error }
                completionHandler(.cancel)
                return
            }
        }
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        let progress: (written: Int64, expected: Int64)? = state.withLockUnchecked { st in
            guard let handle = st.handle, st.writeError == nil else { return nil }
            do {
                try handle.write(contentsOf: data)
                st.bytesWritten += Int64(data.count)
                return (st.bytesWritten, st.expectedTotal)
            } catch {
                st.writeError = error
                return nil
            }
        }
        if let progress {
            onProgress?(progress.written, progress.expected)
        } else if state.withLockUnchecked({ $0.writeError }) != nil {
            cancel()
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        let response = task.response as? HTTPURLResponse
        // Extract the resume action under the lock, run it after releasing.
        let resume = state.withLockUnchecked { st -> (() -> Void)? in
            guard !st.finished, let continuation = st.continuation else { return nil }
            st.finished = true
            st.continuation = nil
            st.task = nil
            try? st.handle?.close()
            st.handle = nil
            let writeError = st.writeError
            return {
                if let writeError {
                    continuation.resume(throwing: writeError)
                } else if let error {
                    continuation.resume(throwing: error)
                } else if let response {
                    continuation.resume(returning: response)
                } else {
                    continuation.resume(throwing: DownloadUtils.HuggingFaceDownloadError.invalidResponse)
                }
            }
        }
        resume?()
    }
}
