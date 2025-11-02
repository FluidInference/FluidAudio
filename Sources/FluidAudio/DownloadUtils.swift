import CoreML
import Foundation
import OSLog

/// HuggingFace model downloader based on swift-transformers implementation
public class DownloadUtils {

    private static let logger = AppLogger(category: "DownloadUtils")

    /// Shared URLSession with registry and proxy configuration
    public static let sharedSession: URLSession = ModelRegistry.configuredSession()

    private static let huggingFaceUserAgent = "FluidAudio/1.0 (HuggingFaceDownloader)"

    public enum HuggingFaceDownloadError: LocalizedError {
        case invalidResponse
        case rateLimited(statusCode: Int, message: String)
        case unexpectedContent(statusCode: Int, mimeType: String?, snippet: String)

        public var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "Received an invalid response from Hugging Face."
            case .rateLimited(_, let message):
                return "Hugging Face rate limit encountered: \(message)"
            case .unexpectedContent(_, let mimeType, let snippet):
                let mimeInfo = mimeType ?? "unknown MIME type"
                return "Unexpected Hugging Face content (\(mimeInfo)): \(snippet)"
            }
        }
    }

    private static func huggingFaceToken() -> String? {
        let env = ProcessInfo.processInfo.environment
        return env["HF_TOKEN"]
            ?? env["HUGGINGFACEHUB_API_TOKEN"]
            ?? env["HUGGING_FACE_HUB_TOKEN"]
    }

    private static func isLikelyHtml(_ data: Data) -> Bool {
        guard !data.isEmpty,
            let prefix = String(data: data.prefix(128), encoding: .utf8)?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
        else {
            return false
        }

        return prefix.hasPrefix("<!doctype html") || prefix.hasPrefix("<html")
    }

    private static func makeHuggingFaceRequest(for url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue(huggingFaceUserAgent, forHTTPHeaderField: "User-Agent")
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        request.timeoutInterval = DownloadConfig.default.timeout

        if let token = huggingFaceToken() {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        return request
    }

    public static func fetchHuggingFaceFile(
        from url: URL,
        description: String,
        maxAttempts: Int = 4,
        minBackoff: TimeInterval = 1.0
    ) async throws -> Data {
        var lastError: Error?

        for attempt in 1...maxAttempts {
            do {
                let request = makeHuggingFaceRequest(for: url)
                let (data, response) = try await sharedSession.data(for: request)

                guard let httpResponse = response as? HTTPURLResponse else {
                    throw HuggingFaceDownloadError.invalidResponse
                }

                if httpResponse.statusCode == 429 || httpResponse.statusCode == 503 {
                    let message = "HTTP \(httpResponse.statusCode)"
                    throw HuggingFaceDownloadError.rateLimited(
                        statusCode: httpResponse.statusCode, message: message)
                }

                if let mimeType = httpResponse.mimeType?.lowercased(),
                    mimeType == "text/html"
                {
                    let snippet = String(data: data.prefix(256), encoding: .utf8) ?? ""
                    throw HuggingFaceDownloadError.unexpectedContent(
                        statusCode: httpResponse.statusCode,
                        mimeType: mimeType,
                        snippet: snippet
                    )
                }

                if isLikelyHtml(data) {
                    let snippet = String(data: data.prefix(256), encoding: .utf8) ?? ""
                    throw HuggingFaceDownloadError.unexpectedContent(
                        statusCode: httpResponse.statusCode,
                        mimeType: httpResponse.mimeType,
                        snippet: snippet
                    )
                }

                return data

            } catch let error as HuggingFaceDownloadError {
                lastError = error

                if attempt == maxAttempts {
                    break
                }

                let backoffSeconds = pow(2.0, Double(attempt - 1)) * minBackoff
                let backoffNanoseconds = UInt64(backoffSeconds * 1_000_000_000)
                let formattedBackoff = String(format: "%.1f", backoffSeconds)

                switch error {
                case .rateLimited(let statusCode, _):
                    if huggingFaceToken() == nil {
                        logger.warning(
                            "Rate limit (HTTP \(statusCode)) while downloading \(description). "
                                + "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN for higher limits. "
                                + "Retrying in \(formattedBackoff)s."
                        )
                    } else {
                        logger.warning(
                            "Rate limit (HTTP \(statusCode)) while downloading \(description). "
                                + "Retrying in \(formattedBackoff)s."
                        )
                    }
                case .unexpectedContent(_, _, let snippet):
                    logger.warning(
                        "Unexpected content while downloading \(description). "
                            + "Snippet: \(snippet.prefix(100)). "
                            + "Retrying in \(formattedBackoff)s."
                    )
                case .invalidResponse:
                    logger.warning(
                        "Invalid response while downloading \(description). "
                            + "Retrying in \(formattedBackoff)s."
                    )
                }

                try await Task.sleep(nanoseconds: backoffNanoseconds)

            } catch {
                lastError = error

                if attempt == maxAttempts {
                    break
                }

                let backoffSeconds = pow(2.0, Double(attempt - 1)) * minBackoff
                let backoffNanoseconds = UInt64(backoffSeconds * 1_000_000_000)
                let formattedBackoff = String(format: "%.1f", backoffSeconds)

                logger.warning(
                    "Download attempt \(attempt) for \(description) failed: "
                        + "\(error.localizedDescription). "
                        + "Retrying in \(formattedBackoff)s."
                )

                try await Task.sleep(nanoseconds: backoffNanoseconds)
            }
        }

        throw lastError ?? HuggingFaceDownloadError.invalidResponse
    }

    /// Download progress callback
    public typealias ProgressHandler = (Double) -> Void

    /// Download configuration
    public struct DownloadConfig {
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
        variant: String? = nil
    ) async throws -> [String: MLModel] {
        // Ensure host environment is logged for debugging (once per process)
        await SystemInfo.logOnce(using: logger)
        do {
            // 1st attempt: normal load
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant)
        } catch {
            // 1st attempt failed → wipe cache to signal redownload
            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)
            try? FileManager.default.removeItem(at: repoPath)

            // 2nd attempt after fresh download
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits, variant: variant)
        }
    }

    /// Internal helper to download repo (if needed) and load CoreML models
    /// - Parameters:
    ///   - repo: The HuggingFace repository to download
    ///   - modelNames: Array of model file names to load (e.g., ["model.mlmodelc"])
    ///   - directory: Base directory to store repos (e.g., ~/Library/Application Support/FluidAudio)
    ///   - computeUnits: CoreML compute units to use (default: CPU and Neural Engine)
    /// - Returns: Dictionary mapping model names to loaded MLModel instances
    private static func loadModelsOnce(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        variant: String? = nil
    ) async throws -> [String: MLModel] {
        // Ensure host environment is logged for debugging (once per process)
        await SystemInfo.logOnce(using: logger)
        // Ensure base directory exists
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Download repo if needed
        let repoPath = directory.appendingPathComponent(repo.folderName)
        if !FileManager.default.fileExists(atPath: repoPath.path) {
            logger.info("Models not found in cache at \(repoPath.path)")
            try await downloadRepo(repo, to: directory, variant: variant)
        } else {
            logger.info("Found \(repo.folderName) locally, no download needed")
        }

        // Configure CoreML
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.allowLowPrecisionAccumulationOnGPU = true

        // Load each model
        var models: [String: MLModel] = [:]
        for name in modelNames {
            let modelPath = repoPath.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw CocoaError(
                    .fileNoSuchFile,
                    userInfo: [
                        NSFilePathErrorKey: modelPath.path,
                        NSLocalizedDescriptionKey: "Model file not found: \(name)",
                    ])
            }

            do {
                // Validate model directory structure before loading (.mlmodelc bundle)
                var isDirectory: ObjCBool = false
                guard
                    FileManager.default.fileExists(
                        atPath: modelPath.path, isDirectory: &isDirectory),
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

                // Measure Core ML model initialization time (aka local compilation/open)
                let start = Date()
                let model = try MLModel(contentsOf: modelPath, configuration: config)
                let elapsed = Date().timeIntervalSince(start)

                models[name] = model

                let ms = elapsed * 1000
                let formatted = String(format: "%.2f", ms)
                logger.info("Compiled model \(name) in \(formatted) ms :: \(SystemInfo.summary())")
            } catch {
                logger.error("Failed to load model \(name): \(error)")

                if let contents = try? FileManager.default.contentsOfDirectory(
                    atPath: modelPath.deletingLastPathComponent().path)
                {
                    logger.error("Model directory contents: \(contents)")
                }

                throw error
            }
        }

        return models
    }

    /// Get required model names for a given repository
    /// Uses centralized ModelNames where available to avoid cross‑type coupling
    private static func getRequiredModelNames(for repo: Repo) -> Set<String> {
        switch repo {
        case .vad:
            return ModelNames.VAD.requiredModels
        case .parakeet, .parakeetV2:
            return ModelNames.ASR.requiredModels
        case .diarizer:
            return ModelNames.Diarizer.requiredModels
        case .kokoro:
            return ModelNames.TTS.requiredModels
        }
    }

    /// Download a HuggingFace repository
    private static func downloadRepo(_ repo: Repo, to directory: URL, variant: String? = nil) async throws {
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        // Get the required model names for this repo from the appropriate manager
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)

        // Download all repository contents
        let files = try await listRepoFiles(repo)

        for file in files {
            switch file.type {
            case "directory" where file.path.hasSuffix(".mlmodelc"):
                // Check if this model is required (with or without subfolder prefix)
                let isRequired =
                    requiredModels.contains(file.path) || requiredModels.contains { $0.hasSuffix("/" + file.path) }

                if isRequired {
                    logger.info("Downloading required model: \(file.path)")

                    // Find if this should go in a subfolder
                    if let fullPath = requiredModels.first(where: { $0.hasSuffix("/" + file.path) }),
                        fullPath.contains("/")
                    {
                        // Extract subfolder (e.g., "speaker-diarization-offline/Segmentation.mlmodelc" -> "speaker-diarization-offline")
                        let subfolder = String(fullPath.split(separator: "/").first!)
                        let subfolderPath = repoPath.appendingPathComponent(subfolder)
                        try FileManager.default.createDirectory(at: subfolderPath, withIntermediateDirectories: true)

                        // Download to subfolder
                        try await downloadModelDirectory(repo: repo, dirPath: file.path, to: subfolderPath)
                    } else {
                        // Download to root of repo
                        try await downloadModelDirectory(repo: repo, dirPath: file.path, to: repoPath)
                    }
                } else {
                    logger.info("Skipping unrequired model: \(file.path)")
                }

            case "file" where isEssentialFile(file.path):
                logger.info("Downloading \(file.path)")
                try await downloadFile(
                    from: repo,
                    path: file.path,
                    to: repoPath.appendingPathComponent(file.path),
                    expectedSize: file.size,
                    config: .default
                )

            default:
                break
            }
        }

        logger.info("Downloaded all required models for \(repo.folderName)")
    }

    /// Check if a file is essential for model operation
    private static func isEssentialFile(_ path: String) -> Bool {
        path.hasSuffix(".json") || path.hasSuffix(".txt") || path == "config.json"
    }

    /// List files in a HuggingFace repository
    private static func listRepoFiles(_ repo: Repo, path: String = "") async throws -> [RepoFile] {
        let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
        let apiURL = try ModelRegistry.apiModels(repo.remotePath, apiPath)

        var request = URLRequest(url: apiURL)
        request.timeoutInterval = 30

        let (data, response) = try await sharedSession.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }

        return try JSONDecoder().decode([RepoFile].self, from: data)
    }

    /// Download a CoreML model directory and all its contents
    private static func downloadModelDirectory(
        repo: Repo, dirPath: String, to destination: URL
    )
        async throws
    {
        let modelDir = destination.appendingPathComponent(dirPath)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)

        let files = try await listRepoFiles(repo, path: dirPath)

        for item in files {
            switch item.type {
            case "directory":
                try await downloadModelDirectory(repo: repo, dirPath: item.path, to: destination)

            case "file":
                let expectedSize = item.lfs?.size ?? item.size

                // Only log large files (>10MB) to reduce noise
                if expectedSize > 10_000_000 {
                    logger.info("Downloading \(item.path) (\(formatBytes(expectedSize)))")
                } else {
                    logger.debug("Downloading \(item.path) (\(formatBytes(expectedSize)))")
                }

                try await downloadFile(
                    from: repo,
                    path: item.path,
                    to: destination.appendingPathComponent(item.path),
                    expectedSize: expectedSize,
                    config: .default,
                    progressHandler: createProgressHandler(for: item.path, size: expectedSize)
                )

            default:
                break
            }
        }
    }

    /// Create a progress handler for large files
    private static func createProgressHandler(for path: String, size: Int) -> ProgressHandler? {
        // Only show progress for files over 100MB (most files are under this)
        guard size > 100_000_000 else { return nil }

        let fileName = path.split(separator: "/").last ?? ""
        var lastReportedPercentage = 0

        return { progress in
            let percentage = Int(progress * 100)
            if percentage >= lastReportedPercentage + 10 {
                lastReportedPercentage = percentage
                logger.info("Progress: \(percentage)% of \(fileName)")
            }
        }
    }

    /// Download a single file with chunked transfer and resume support
    private static func downloadFile(
        from repo: Repo,
        path: String,
        to destination: URL,
        expectedSize: Int,
        config: DownloadConfig,
        progressHandler: ProgressHandler? = nil
    ) async throws {
        // Create parent directories
        let parentDir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        if let attrs = try? FileManager.default.attributesOfItem(atPath: destination.path),
            let fileSize = attrs[.size] as? Int64,
            fileSize == expectedSize
        {
            logger.info("File already downloaded: \(path)")
            progressHandler?(1.0)
            return
        }

        // Temporary file for downloading
        let tempURL = destination.appendingPathExtension("download")

        // Check if we can resume a partial download
        var startByte: Int64 = 0
        if let attrs = try? FileManager.default.attributesOfItem(atPath: tempURL.path),
            let fileSize = attrs[.size] as? Int64
        {
            startByte = fileSize
            logger.info("⏸️ Resuming download from \(formatBytes(Int(startByte)))")
        }

        // Download URL
        let downloadURL = try ModelRegistry.resolveModel(repo.remotePath, path)

        // Download the file (no retries)
        do {
            try await performChunkedDownload(
                from: downloadURL,
                to: tempURL,
                startByte: startByte,
                expectedSize: Int64(expectedSize),
                config: config,
                progressHandler: progressHandler
            )

            // Verify file size before moving
            if let attrs = try? FileManager.default.attributesOfItem(atPath: tempURL.path),
                let fileSize = attrs[.size] as? Int64
            {
                if fileSize != expectedSize {
                    logger.warning(
                        "⚠️ Downloaded file size mismatch for \(path): got \(fileSize), expected \(expectedSize)"
                    )
                }
            }

            // Move completed file with better error handling
            do {
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: tempURL, to: destination)
            } catch {
                // In CI, file operations might fail due to sandbox restrictions
                // Try copying instead of moving as a fallback
                logger.warning("Move failed for \(path), attempting copy: \(error)")
                try FileManager.default.copyItem(at: tempURL, to: destination)
                try? FileManager.default.removeItem(at: tempURL)
            }
            logger.info("Downloaded \(path)")

        } catch {
            logger.error("Download failed: \(error)")
            throw error
        }
    }

    /// Perform chunked download with progress tracking
    private static func performChunkedDownload(
        from url: URL,
        to destination: URL,
        startByte: Int64,
        expectedSize: Int64,
        config: DownloadConfig,
        progressHandler: ProgressHandler?
    ) async throws {
        var request = URLRequest(url: url)
        request.timeoutInterval = config.timeout

        // Use URLSession download task with progress
        // Always use URLSession.download for reliability (proven to work in PR #32)
        let (tempFile, response) = try await sharedSession.download(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
            httpResponse.statusCode == 200
        else {
            throw URLError(.badServerResponse)
        }

        // Ensure parent directory exists before moving
        let parentDir = destination.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parentDir, withIntermediateDirectories: true)

        // Move to destination with better error handling for CI
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: tempFile, to: destination)
        } catch {
            // In CI, URLSession might download to a different temp location
            // Try copying instead of moving as a fallback
            logger.warning("Move failed, attempting copy: \(error)")
            try FileManager.default.copyItem(at: tempFile, to: destination)
            try? FileManager.default.removeItem(at: tempFile)
        }

        // Report complete
        progressHandler?(1.0)
    }

    /// Format bytes for display
    private static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }

    /// Repository file information
    private struct RepoFile: Codable {
        let type: String
        let path: String
        let size: Int
        let lfs: LFSInfo?

        struct LFSInfo: Codable {
            let size: Int
            let sha256: String?  // Some repos might have this
            let oid: String?  // Most use this instead
            let pointerSize: Int?

            enum CodingKeys: String, CodingKey {
                case size
                case sha256
                case oid
                case pointerSize = "pointer_size"
            }
        }
    }
}
