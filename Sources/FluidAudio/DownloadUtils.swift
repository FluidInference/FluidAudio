import CoreML
import Foundation
import OSLog

/// HuggingFace model downloader based on swift-transformers implementation
public class DownloadUtils {

    private static let logger = AppLogger(category: "DownloadUtils")

    public static let sharedSession: URLSession = {
        let configuration = URLSessionConfiguration.default

        // Configure proxy settings if environment variables are set
        if let proxyConfig = configureProxySettings() {
            configuration.connectionProxyDictionary = proxyConfig
        }

        return URLSession(configuration: configuration)
    }()

    private static func configureProxySettings() -> [String: Any]? {
        #if os(macOS)
        var proxyConfig: [String: Any] = [:]
        var hasProxyConfig = false

        // Configure HTTPS proxy
        if let httpsProxy = ProcessInfo.processInfo.environment["https_proxy"],
            let proxySettings = parseProxyURL(httpsProxy, type: "HTTPS")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
            hasProxyConfig = true
        }

        // Configure HTTP proxy
        if let httpProxy = ProcessInfo.processInfo.environment["http_proxy"],
            let proxySettings = parseProxyURL(httpProxy, type: "HTTP")
        {
            proxyConfig.merge(proxySettings) { _, new in new }
            hasProxyConfig = true
        }

        return hasProxyConfig ? proxyConfig : nil
        #else
        // Proxy configuration not available on iOS
        return nil
        #endif
    }

    private static func parseProxyURL(_ proxyURLString: String, type: String) -> [String: Any]? {
        #if os(macOS)
        guard let proxyURL = URL(string: proxyURLString),
            let host = proxyURL.host,
            let port = proxyURL.port
        else {
            logger.warning("Invalid \(type) proxy URL: \(proxyURLString)")
            return nil
        }

        let config: [String: Any]
        switch type {
        case "HTTPS":
            config = [
                kCFNetworkProxiesHTTPSEnable as String: true,
                kCFNetworkProxiesHTTPSProxy as String: host,
                kCFNetworkProxiesHTTPSPort as String: port,
            ]
        case "HTTP":
            config = [
                kCFNetworkProxiesHTTPEnable as String: true,
                kCFNetworkProxiesHTTPProxy as String: host,
                kCFNetworkProxiesHTTPPort as String: port,
            ]
        default:
            return nil
        }

        logger.info("Configured \(type) proxy: \(host):\(port)")
        return config
        #else
        // Proxy configuration not available on iOS
        return nil
        #endif
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
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> [String: MLModel] {
        // Ensure host environment is logged for debugging (once per process)
        await SystemInfo.logOnce(using: logger)
        do {
            // 1st attempt: normal load
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits)
        } catch {
            // 1st attempt failed → wipe cache to signal redownload
            logger.warning("First load failed: \(error.localizedDescription)")
            logger.info("Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)
            try? FileManager.default.removeItem(at: repoPath)

            // 2nd attempt after fresh download
            return try await loadModelsOnce(
                repo, modelNames: modelNames,
                directory: directory, computeUnits: computeUnits)
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
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> [String: MLModel] {
        // Ensure host environment is logged for debugging (once per process)
        await SystemInfo.logOnce(using: logger)
        // Ensure base directory exists
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Download repo if needed
        let repoPath = directory.appendingPathComponent(repo.folderName)
        if !FileManager.default.fileExists(atPath: repoPath.path) {
            logger.info("Models not found in cache at \(repoPath.path)")
            try await downloadRepo(repo, to: directory)
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
                // Validate model directory structure before loading
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

                // Check for essential model files
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

                // List directory contents for debugging
                if let contents = try? FileManager.default.contentsOfDirectory(
                    at: modelPath, includingPropertiesForKeys: nil)
                {
                    logger.error("Model directory contents: \(contents.map { $0.lastPathComponent })")
                }

                throw error
            }
        }

        return models
    }

    /// Download a HuggingFace repository
    private static func downloadRepo(_ repo: Repo, to directory: URL) async throws {
        logger.info("Downloading \(repo.folderName) from HuggingFace...")

        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)

        // Get the required model names for this repo from the appropriate manager
        let requiredModels = ModelNames.getRequiredModelNames(for: repo)

        // Download all repository contents
        let files = try await listRepoFiles(repo)

        for file in files {
            switch file.type {
            case "directory" where file.path.hasSuffix(".mlmodelc"):
                // Only download if this model is in our required list
                if requiredModels.contains(file.path) {
                    logger.info("Downloading required model: \(file.path)")
                    try await downloadModelDirectory(repo: repo, dirPath: file.path, to: repoPath)
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
        let apiURL = URL(string: "https://huggingface.co/api/models/\(repo.rawValue)/\(apiPath)")!

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

        // Check if file already exists and is complete
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
        let downloadURL = URL(
            string: "https://huggingface.co/\(repo.rawValue)/resolve/main/\(path)")!

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
