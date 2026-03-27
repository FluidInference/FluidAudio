import CoreML
import Foundation
import OSLog

/// HuggingFace model downloader using URLSession
public class DownloadUtils {

    private static let logger = AppLogger(category: "DownloadUtils")

    /// Shared URLSession with registry and proxy configuration
    public static let sharedSession: URLSession = ModelRegistry.configuredSession()

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

    public enum HuggingFaceDownloadError: LocalizedError {
        case invalidResponse
        case rateLimited(statusCode: Int, message: String)
        case downloadFailed(path: String, underlying: Error)
        case modelNotFound(path: String)
        case htmlErrorResponse(path: String, snippet: String)

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
            }
        }
    }

    /// Download configuration

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
            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)
            try? FileManager.default.removeItem(at: repoPath)

            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant,
                progressHandler: progressHandler)
        }
    }

    public static func clearModelCache(forRepo repo: Repo, directory: URL) {
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try? FileManager.default.removeItem(at: repoPath)
    }

    /// Remove all downloaded models and caches.
    ///
    /// Clears both cache locations:
    /// - `~/Library/Application Support/FluidAudio/Models/` (ASR, VAD, Diarization)
    /// - `~/.cache/fluidaudio/Models/` (TTS)
    public static func clearAllModelCaches() {
        let fm = FileManager.default

        // ASR, VAD, Diarization models
        if let appSupport = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            let modelsDir = appSupport.appendingPathComponent("FluidAudio/Models")
            try? fm.removeItem(at: modelsDir)
        }

        // TTS models (Kokoro, PocketTTS)
        #if os(macOS)
        let home = fm.homeDirectoryForCurrentUser
        let ttsCache = home.appendingPathComponent(".cache/fluidaudio/Models")
        try? fm.removeItem(at: ttsCache)
        #else
        if let cacheDir = fm.urls(for: .cachesDirectory, in: .userDomainMask).first {
            let ttsCache = cacheDir.appendingPathComponent("fluidaudio/Models")
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
        let allModelsExist = requiredModels.allSatisfy { model in
            let modelPath = repoPath.appendingPathComponent(model)
            return FileManager.default.fileExists(atPath: modelPath.path)
        }

        if !allModelsExist {
            logger.info("Models not found in cache at \(repoPath.path)")
            try await downloadRepo(repo, to: directory, variant: variant, progressHandler: progressHandler)
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

    /// Download a HuggingFace repository using URLSession (does not load models)
    public static func downloadRepo(
        _ repo: Repo,
        to directory: URL,
        variant: String? = nil,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
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

            let (dirData, response) = try await sharedSession.data(for: request)

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

        // Start listing from subPath if specified, otherwise from root
        progressHandler?(DownloadProgress(fractionCompleted: 0.0, phase: .listing))
        try await listDirectory(path: subPath ?? "")
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

            // Create parent directory
            try FileManager.default.createDirectory(
                at: destPath.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

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

            let tempFileURL: URL
            let httpResponse: HTTPURLResponse

            if let handler = progressHandler {
                let baseBytes = completedBytes
                let fileCount = filesToDownload.count
                let totalBytesSnapshot = totalBytes
                (tempFileURL, httpResponse) = try await downloadWithProgress(
                    request: request,
                    onProgress: { bytesWritten, _ in
                        guard totalBytesSnapshot > 0 else { return }
                        let current = baseBytes + bytesWritten
                        // Download phase occupies 0.0–0.5 of the overall range.
                        let fraction = 0.5 * Double(current) / Double(totalBytesSnapshot)
                        handler(
                            DownloadProgress(
                                fractionCompleted: min(fraction, 0.5),
                                phase: .downloading(completedFiles: index, totalFiles: fileCount)
                            ))
                    }
                )
            } else {
                let (url, response) = try await sharedSession.download(for: request)
                guard let resp = response as? HTTPURLResponse else {
                    throw HuggingFaceDownloadError.invalidResponse
                }
                tempFileURL = url
                httpResponse = resp
            }

            // Validate response
            if httpResponse.statusCode == 429 || httpResponse.statusCode == 503 {
                throw HuggingFaceDownloadError.rateLimited(
                    statusCode: httpResponse.statusCode,
                    message: "Rate limited while downloading \(file.path)")
            }

            guard (200..<300).contains(httpResponse.statusCode) else {
                throw HuggingFaceDownloadError.downloadFailed(
                    path: file.path,
                    underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                )
            }

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

    // MARK: - Delegate-based download with per-byte progress

    /// Download a single file using a delegate to get byte-level progress.
    ///
    /// This is a pure transport helper — the caller is responsible for validating
    /// the HTTP status and moving the temporary file to its final destination.
    ///
    /// - Parameters:
    ///   - request: The URLRequest to download.
    ///   - onProgress: Called with `(totalBytesWritten, totalBytesExpected)` as data arrives.
    /// - Returns: The temporary file URL and HTTP response.
    private static func downloadWithProgress(
        request: URLRequest,
        onProgress: @escaping @Sendable (Int64, Int64) -> Void
    ) async throws -> (URL, HTTPURLResponse) {
        let delegate = DownloadProgressDelegate(onProgress: onProgress)
        // Dedicated session with delegate — one per download to avoid cross-talk.
        let session = URLSession(
            configuration: sharedSession.configuration,
            delegate: delegate,
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }

        let (tempURL, response) = try await session.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw HuggingFaceDownloadError.invalidResponse
        }

        return (tempURL, httpResponse)
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
    public static func downloadSubdirectory(
        _ repo: Repo,
        subdirectory: String,
        to repoDirectory: URL
    ) async throws {
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

                if itemType == "directory" {
                    try await listFiles(at: itemPath)
                } else if itemType == "file" {
                    let fileSize = item["size"] as? Int ?? -1
                    filesToDownload.append((path: itemPath, size: fileSize))
                }
            }
        }

        try await listFiles(at: subdirectory)
        logger.info("Found \(filesToDownload.count) files in \(subdirectory)")

        for (index, file) in filesToDownload.enumerated() {
            let destPath = repoDirectory.appendingPathComponent(file.path)

            if FileManager.default.fileExists(atPath: destPath.path) {
                continue
            }

            try FileManager.default.createDirectory(
                at: destPath.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )

            if file.size == 0 {
                FileManager.default.createFile(atPath: destPath.path, contents: Data())
                continue
            }

            let encodedPath =
                file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
            let fileURL = try ModelRegistry.resolveModel(repo.remotePath, encodedPath)
            let request = authorizedRequest(url: fileURL)

            let (tempURL, response) = try await sharedSession.download(for: request)
            guard let httpResponse = response as? HTTPURLResponse else {
                throw HuggingFaceDownloadError.invalidResponse
            }

            if httpResponse.statusCode == 429 || httpResponse.statusCode == 503 {
                throw HuggingFaceDownloadError.rateLimited(
                    statusCode: httpResponse.statusCode,
                    message: "Rate limited while downloading \(file.path)")
            }

            guard (200..<300).contains(httpResponse.statusCode) else {
                throw HuggingFaceDownloadError.downloadFailed(
                    path: file.path,
                    underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                )
            }

            if FileManager.default.fileExists(atPath: destPath.path) {
                try? FileManager.default.removeItem(at: destPath)
            }
            try FileManager.default.moveItem(at: tempURL, to: destPath)

            if (index + 1) % 5 == 0 || index == filesToDownload.count - 1 {
                logger.info("Downloaded \(index + 1)/\(filesToDownload.count) \(subdirectory) files")
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

/// Lightweight delegate that forwards `didWriteData` callbacks to a closure.
private final class DownloadProgressDelegate: NSObject, URLSessionDownloadDelegate, Sendable {
    private let onProgress: @Sendable (Int64, Int64) -> Void

    init(onProgress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Required by protocol — the async download(for:) API handles the file.
    }
}
