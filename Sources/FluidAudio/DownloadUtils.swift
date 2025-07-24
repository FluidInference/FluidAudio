import Foundation
import CoreML
import OSLog

/// HuggingFace model downloader based on swift-transformers implementation
public class DownloadUtils {

    private static let logger = Logger(subsystem: "com.fluidaudio", category: "DownloadUtils")
    
    /// Download progress callback
    public typealias ProgressHandler = (Double) -> Void
    
    /// Download configuration
    public struct DownloadConfig {
        public let maxRetries: Int
        public let timeout: TimeInterval
        public let chunkSize: Int
        public let allowsCellular: Bool
        
        public init(
            maxRetries: Int = 3,
            timeout: TimeInterval = 300,
            chunkSize: Int = 10 * 1024 * 1024, // 10MB chunks
            allowsCellular: Bool = true
        ) {
            self.maxRetries = maxRetries
            self.timeout = timeout
            self.chunkSize = chunkSize
            self.allowsCellular = allowsCellular
        }
        
        public static let `default` = DownloadConfig()
    }

    /// Model repositories on HuggingFace
    public enum Repo: String, CaseIterable {
        case vad = "FluidInference/silero-vad-coreml"
        case parakeet = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
        case diarizer = "FluidInference/speaker-diarization-coreml"

        var url: String { "https://huggingface.co/\(rawValue)" }
        var folderName: String { rawValue.split(separator: "/").last!.description }
    }

    public static func loadModels(
        _ repo: Repo,
        modelNames: [String],
        directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> [String: MLModel] {
        do {
            // 1st attempt: normal load
            return try await loadModelsOnce(repo, modelNames: modelNames,
                                            directory: directory, computeUnits: computeUnits)
        } catch {
            // 1st attempt failed → wipe cache to signal redownload
            logger.warning("⚠️ First load failed: \(error.localizedDescription)")
            logger.info("🔄 Deleting cache and re-downloading…")
            let repoPath = directory.appendingPathComponent(repo.folderName)
            try? FileManager.default.removeItem(at: repoPath)

            // 2nd attempt after fresh download
            return try await loadModelsOnce(repo, modelNames: modelNames,
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
        // Ensure base directory exists
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        // Download repo if needed
        let repoPath = directory.appendingPathComponent(repo.folderName)
        if !FileManager.default.fileExists(atPath: repoPath.path) {
            try await downloadRepo(repo, to: directory)
        }

        // Configure CoreML
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits

        // Load each model
        var models: [String: MLModel] = [:]
        for name in modelNames {
            let modelPath = repoPath.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: modelPath.path) else {
                throw CocoaError(.fileNoSuchFile, userInfo: [
                    NSFilePathErrorKey: modelPath.path,
                    NSLocalizedDescriptionKey: "Model file not found: \(name)"
                ])
            }

            do {
                models[name] = try MLModel(contentsOf: modelPath, configuration: config)
                logger.info("✅ Loaded model: \(name)")
            } catch {
                logger.error("❌ Failed to load model \(name): \(error)")
                throw error
            }
        }

        return models
    }

    /// Download a HuggingFace repository using git clone
    private static func downloadRepo(_ repo: Repo, to directory: URL) async throws {
        logger.info("📥 Downloading \(repo.folderName) from HuggingFace...")
        print("📥 Downloading \(repo.folderName)...")

        // Always use URLSession for better compatibility
        // Git is not assumed to be available on user machines
        try await downloadRepoWithURLSession(repo, to: directory)
    }
    
    // DEPRECATED: Git method removed - we don't assume git is installed
    // All downloads now use URLSession for better compatibility
    /*
    /// Download using git (original method)
    private static func downloadRepoWithGit(_ repo: Repo, to directory: URL, gitPath: String) async throws {
        // Use a temporary directory for cloning
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        // Download to temp directory first using git clone
        let process = Process()
        process.executableURL = URL(fileURLWithPath: gitPath)
        process.arguments = ["clone", "--depth", "1", repo.url]
        process.currentDirectoryURL = tempDir

        let pipe = Pipe()
        process.standardError = pipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let errorData = pipe.fileHandleForReading.readDataToEndOfFile()
            let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            logger.error("❌ Git clone failed: \(errorMessage)")
            throw URLError(.cannotConnectToHost, userInfo: [
                NSLocalizedDescriptionKey: "Failed to download repository: \(errorMessage)"
            ])
        }

        // Check if Git LFS files need to be pulled
        let downloadedPath = tempDir.appendingPathComponent(repo.folderName)
        let gitAttributesPath = downloadedPath.appendingPathComponent(".gitattributes")

        if FileManager.default.fileExists(atPath: gitAttributesPath.path) {
            logger.info("📦 Pulling Git LFS files...")

            let lfsProcess = Process()
            lfsProcess.executableURL = URL(fileURLWithPath: gitPath)
            lfsProcess.arguments = ["lfs", "pull"]
            lfsProcess.currentDirectoryURL = downloadedPath

            do {
                try lfsProcess.run()
                lfsProcess.waitUntilExit()
            } catch {
                logger.warning("⚠️ Git LFS pull failed, models might already be included")
            }
        }

        // Move from temp to final location
        let finalPath = directory.appendingPathComponent(repo.folderName)

        // This should never happen in normal flow, but handle edge cases
        // where deletion failed or was partial during recovery
        if FileManager.default.fileExists(atPath: finalPath.path) {
            logger.warning("⚠️ Unexpected: directory exists at \(finalPath.lastPathComponent) during download")
            try FileManager.default.removeItem(at: finalPath)
        }

        try FileManager.default.moveItem(at: downloadedPath, to: finalPath)

        logger.info("✅ Downloaded \(repo.folderName)")
        print("✅ Download complete")
    }
    */
    
    /// Download a HuggingFace repository
    private static func downloadRepoWithURLSession(_ repo: Repo, to directory: URL) async throws {
        logger.info("📥 Downloading \(repo.folderName) from HuggingFace...")
        
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
        
        // Get list of files in the repository
        let files = try await listRepoFiles(repo)
        
        // Filter to only model files we need
        let modelFiles = files.filter { file in
            file.path.hasSuffix(".mlmodelc") && !file.path.hasPrefix(".")
        }
        
        // Download each model file
        for file in modelFiles {
            let fileURL = repoPath.appendingPathComponent(file.path)
            let expectedSize = file.lfs?.size ?? file.size
            
            logger.info("📥 Downloading \(file.path) (\(formatBytes(expectedSize)))")
            
            try await downloadFile(
                from: repo,
                path: file.path,
                to: fileURL,
                expectedSize: expectedSize,
                config: .default
            )
        }
        
        logger.info("✅ Downloaded all models for \(repo.folderName)")
    }
    
    /// List files in a HuggingFace repository
    private static func listRepoFiles(_ repo: Repo) async throws -> [RepoFile] {
        let apiURL = URL(string: "https://huggingface.co/api/models/\(repo.rawValue)/tree/main")!
        
        var request = URLRequest(url: apiURL)
        request.timeoutInterval = 30
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        return try JSONDecoder().decode([RepoFile].self, from: data)
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
           fileSize == expectedSize {
            logger.info("✅ File already downloaded: \(path)")
            progressHandler?(1.0)
            return
        }
        
        // Temporary file for downloading
        let tempURL = destination.appendingPathExtension("download")
        
        // Check if we can resume a partial download
        var startByte: Int64 = 0
        if let attrs = try? FileManager.default.attributesOfItem(atPath: tempURL.path),
           let fileSize = attrs[.size] as? Int64 {
            startByte = fileSize
            logger.info("⏸️ Resuming download from \(formatBytes(Int(startByte)))")
        }
        
        // Download URL
        let downloadURL = URL(string: "https://huggingface.co/\(repo.rawValue)/resolve/main/\(path)")!
        
        // Try download with retries
        var lastError: Error?
        for attempt in 0..<config.maxRetries {
            do {
                if attempt > 0 {
                    logger.info("🔄 Retry attempt \(attempt + 1) of \(config.maxRetries)")
                }
                
                // Update start byte for resume
                if attempt > 0, let attrs = try? FileManager.default.attributesOfItem(atPath: tempURL.path),
                   let fileSize = attrs[.size] as? Int64 {
                    startByte = fileSize
                }
                
                try await performChunkedDownload(
                    from: downloadURL,
                    to: tempURL,
                    startByte: startByte,
                    expectedSize: Int64(expectedSize),
                    config: config,
                    progressHandler: progressHandler
                )
                
                // Success - move completed file
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: tempURL, to: destination)
                
                logger.info("✅ Downloaded \(path)")
                return
                
            } catch {
                lastError = error
                logger.error("❌ Download attempt \(attempt + 1) failed: \(error)")
                
                // Don't retry on certain errors
                if (error as? URLError)?.code == .fileDoesNotExist {
                    throw error
                }
                
                // Wait before retry
                if attempt < config.maxRetries - 1 {
                    try await Task.sleep(nanoseconds: UInt64(pow(2.0, Double(attempt)) * 1_000_000_000))
                }
            }
        }
        
        throw lastError ?? URLError(.unknown)
    }
    
    /// Perform chunked download with resume support
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
        
        // Add range header for resume
        if startByte > 0 {
            request.setValue("bytes=\(startByte)-", forHTTPHeaderField: "Range")
        }
        
        // Create or append to file
        if !FileManager.default.fileExists(atPath: destination.path) {
            FileManager.default.createFile(atPath: destination.path, contents: nil)
        }
        
        guard let fileHandle = FileHandle(forWritingAtPath: destination.path) else {
            throw URLError(.cannotCreateFile)
        }
        
        defer { try? fileHandle.close() }
        
        // Seek to end for appending
        if startByte > 0 {
            try fileHandle.seek(toOffset: UInt64(startByte))
        }
        
        // Download in chunks
        var bytesReceived = startByte
        let session = URLSession.shared
        
        let (bytes, response) = try await session.bytes(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse,
              httpResponse.statusCode == 200 || httpResponse.statusCode == 206 else {
            throw URLError(.badServerResponse)
        }
        
        // Download with proper chunking
        let chunkSize = config.chunkSize
        var buffer = Data()
        buffer.reserveCapacity(chunkSize)
        
        for try await byte in bytes {
            buffer.append(byte)
            
            // Write chunk when buffer is full
            if buffer.count >= chunkSize {
                try fileHandle.write(contentsOf: buffer)
                bytesReceived += Int64(buffer.count)
                buffer.removeAll(keepingCapacity: true)
                
                // Report progress
                if let progressHandler = progressHandler {
                    let progress = Double(bytesReceived) / Double(expectedSize)
                    progressHandler(min(progress, 1.0))
                }
            }
        }
        
        // Write remaining data
        if !buffer.isEmpty {
            try fileHandle.write(contentsOf: buffer)
            bytesReceived += Int64(buffer.count)
            
            // Final progress update
            if let progressHandler = progressHandler {
                let progress = Double(bytesReceived) / Double(expectedSize)
                progressHandler(min(progress, 1.0))
            }
        }
        
        // Verify final size
        let finalSize = try FileManager.default.attributesOfItem(atPath: destination.path)[.size] as? Int64 ?? 0
        if finalSize != expectedSize {
            logger.warning("⚠️ File size mismatch: expected \(expectedSize), got \(finalSize)")
        }
    }
    
    /// Format bytes for display
    private static func formatBytes(_ bytes: Int) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: Int64(bytes))
    }
    
    /// Repository file information
    private struct RepoFile: Codable {
        let path: String
        let size: Int
        let lfs: LFSInfo?
        
        struct LFSInfo: Codable {
            let size: Int
            let sha256: String
            let pointerSize: Int
            
            enum CodingKeys: String, CodingKey {
                case size
                case sha256
                case pointerSize = "pointer_size"
            }
        }
    }

}
