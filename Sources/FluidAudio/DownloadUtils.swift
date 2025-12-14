import CoreML
import Foundation
import OSLog

/// HuggingFace model downloader using URLSession
public class DownloadUtils {

    private static let logger = AppLogger(category: "DownloadUtils")

    /// Shared URLSession with registry and proxy configuration
    public static let sharedSession: URLSession = ModelRegistry.configuredSession()

    /// Get HuggingFace token from environment if available
    /// Checks HF_TOKEN and HUGGING_FACE_HUB_TOKEN environment variables
    private static var huggingFaceToken: String? {
        ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? ProcessInfo.processInfo.environment["HUGGING_FACE_HUB_TOKEN"]
    }

    /// Create a URLRequest with optional auth header
    private static func authorizedRequest(url: URL) -> URLRequest {
        var request = URLRequest(url: url)
        if let token = huggingFaceToken {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    public enum HuggingFaceDownloadError: LocalizedError {
        case invalidResponse
        case rateLimited(statusCode: Int, message: String)
        case unexpectedContent(statusCode: Int, mimeType: String?, snippet: String)
        case downloadFailed(path: String, underlying: Error)
        case modelNotFound(path: String)

        public var errorDescription: String? {
            switch self {
            case .invalidResponse:
                return "Received an invalid response from Hugging Face."
            case .rateLimited(_, let message):
                return "Hugging Face rate limit encountered: \(message)"
            case .unexpectedContent(_, let mimeType, let snippet):
                let mimeInfo = mimeType ?? "unknown MIME type"
                return "Unexpected Hugging Face content (\(mimeInfo)): \(snippet)"
            case .downloadFailed(let path, let underlying):
                return "Failed to download \(path): \(underlying.localizedDescription)"
            case .modelNotFound(let path):
                return "Model file not found: \(path)"
            }
        }
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

        // Download repo if needed - check that all required models exist, not just repo folder
        let repoPath = directory.appendingPathComponent(repo.folderName)
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)
        let allModelsExist = requiredModels.allSatisfy { model in
            let modelPath = repoPath.appendingPathComponent(model)
            return FileManager.default.fileExists(atPath: modelPath.path)
        }

        if !allModelsExist {
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

    /// Download a HuggingFace repository using URLSession
    private static func downloadRepo(_ repo: Repo, to directory: URL, variant: String? = nil) async throws {
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let finalPath = directory.appendingPathComponent(repo.folderName)

        // Download to a temporary directory first to ensure atomic operation
        let tempPath = directory.appendingPathComponent(".\(repo.folderName).downloading.\(UUID().uuidString)")

        // Clean up any existing temp directory
        try? FileManager.default.removeItem(at: tempPath)
        try FileManager.default.createDirectory(at: tempPath, withIntermediateDirectories: true)

        // Get the required model names for this repo
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant)

        do {
            // Build patterns for filtering
            var patterns: [String] = []
            for model in requiredModels {
                patterns.append("\(model)/")
            }

            // Get all files recursively using HuggingFace API
            var filesToDownload: [(path: String, isLFS: Bool)] = []

            func processDirectory(path: String) async throws {
                let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
                let dirURL = try ModelRegistry.apiModels(repo.remotePath, apiPath)
                let request = authorizedRequest(url: dirURL)

                let (dirData, response) = try await sharedSession.data(for: request)

                if let httpResponse = response as? HTTPURLResponse,
                    httpResponse.statusCode == 429
                {
                    throw HuggingFaceDownloadError.rateLimited(
                        statusCode: 429, message: "Rate limited while listing files")
                }

                guard let items = try JSONSerialization.jsonObject(with: dirData) as? [[String: Any]] else {
                    return
                }

                for item in items {
                    guard let itemPath = item["path"] as? String,
                        let itemType = item["type"] as? String
                    else { continue }

                    if itemType == "directory" {
                        // Check if this directory matches our patterns
                        let shouldProcess =
                            patterns.isEmpty
                            || patterns.contains { itemPath.hasPrefix($0) || $0.hasPrefix(itemPath + "/") }
                        if shouldProcess {
                            try await processDirectory(path: itemPath)
                        }
                    } else if itemType == "file" {
                        // Check if file matches patterns
                        let shouldInclude =
                            patterns.isEmpty || patterns.contains { itemPath.hasPrefix($0) }
                            || itemPath.hasSuffix(".json") || itemPath.hasSuffix(".txt")
                        if shouldInclude {
                            let isLFS = (item["lfs"] as? [String: Any]) != nil
                            filesToDownload.append((itemPath, isLFS))
                        }
                    }
                }
            }

            try await processDirectory(path: "")
            logger.info("Found \(filesToDownload.count) files to download")

            // Download each file
            for (index, file) in filesToDownload.enumerated() {
                let destPath = tempPath.appendingPathComponent(file.path)

                // Create parent directory
                try FileManager.default.createDirectory(
                    at: destPath.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )

                // Skip if destination already exists
                if FileManager.default.fileExists(atPath: destPath.path) {
                    continue
                }

                // Download file using registry-aware URL
                let encodedFilePath =
                    file.path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.path
                let fileURL = try ModelRegistry.resolveModel(repo.remotePath, encodedFilePath)
                let request = authorizedRequest(url: fileURL)

                // Use streaming download for LFS files (typically larger)
                if file.isLFS {
                    let (tempFileURL, response) = try await sharedSession.download(for: request)

                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw HuggingFaceDownloadError.invalidResponse
                    }

                    if httpResponse.statusCode == 429 {
                        throw HuggingFaceDownloadError.rateLimited(
                            statusCode: 429, message: "Rate limited while downloading \(file.path)")
                    }

                    guard (200..<300).contains(httpResponse.statusCode) else {
                        throw HuggingFaceDownloadError.downloadFailed(
                            path: file.path,
                            underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                        )
                    }

                    try FileManager.default.moveItem(at: tempFileURL, to: destPath)
                } else {
                    // Use in-memory download for small files
                    let (fileData, response) = try await sharedSession.data(for: request)

                    guard let httpResponse = response as? HTTPURLResponse else {
                        throw HuggingFaceDownloadError.invalidResponse
                    }

                    if httpResponse.statusCode == 429 {
                        throw HuggingFaceDownloadError.rateLimited(
                            statusCode: 429, message: "Rate limited while downloading \(file.path)")
                    }

                    guard (200..<300).contains(httpResponse.statusCode) else {
                        throw HuggingFaceDownloadError.downloadFailed(
                            path: file.path,
                            underlying: NSError(domain: "HTTP", code: httpResponse.statusCode)
                        )
                    }

                    try fileData.write(to: destPath)
                }

                if (index + 1) % 10 == 0 || index == filesToDownload.count - 1 {
                    logger.info("Downloaded \(index + 1)/\(filesToDownload.count) files")
                }
            }

            // Verify required models are present
            for model in requiredModels {
                let modelPath = tempPath.appendingPathComponent(model)
                guard FileManager.default.fileExists(atPath: modelPath.path) else {
                    try? FileManager.default.removeItem(at: tempPath)
                    throw HuggingFaceDownloadError.modelNotFound(path: model)
                }
            }

            // Atomically move from temp to final location
            if FileManager.default.fileExists(atPath: finalPath.path) {
                logger.info("Removing existing directory at \(finalPath.path)")
                try FileManager.default.removeItem(at: finalPath)
            }

            try FileManager.default.moveItem(at: tempPath, to: finalPath)
            logger.info("Downloaded all required models for \(repo.folderName)")

        } catch {
            try? FileManager.default.removeItem(at: tempPath)
            logger.error("Failed to download repo \(repo.folderName): \(error)")
            throw HuggingFaceDownloadError.downloadFailed(path: repo.remotePath, underlying: error)
        }
    }

    /// Fetch a single file from HuggingFace
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

    /// Format bytes for display
    private static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
}
