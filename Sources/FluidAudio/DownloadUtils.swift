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
        
        public init(
            maxRetries: Int = 3,
            timeout: TimeInterval = 600, // 10 minutes for large models
            chunkSize: Int = 10 * 1024 * 1024 // 10MB chunks
        ) {
            self.maxRetries = maxRetries
            self.timeout = timeout
            self.chunkSize = chunkSize
        }
        
        public static let `default` = DownloadConfig()
    }

    /// Model repositories on HuggingFace
    public enum Repo: String, CaseIterable {
        case vad = "FluidInference/silero-vad-coreml"
        case parakeet = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
        case diarizer = "FluidInference/speaker-diarization-coreml"

        var folderName: String { 
            rawValue.split(separator: "/").last?.description ?? rawValue 
        }
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

    /// Download a HuggingFace repository
    private static func downloadRepo(_ repo: Repo, to directory: URL) async throws {
        logger.info("📥 Downloading \(repo.folderName) from HuggingFace...")
        print("📥 Downloading \(repo.folderName)...")
        
        let repoPath = directory.appendingPathComponent(repo.folderName)
        
        // Ensure clean directory
        try cleanupIncompleteDownloads(at: repoPath)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
        
        // Download all repository contents
        let files = try await listRepoFiles(repo)
        
        for file in files {
            switch file.type {
            case "directory" where file.path.hasSuffix(".mlmodelc"):
                logger.info("📥 Downloading model: \(file.path)")
                try await downloadModelDirectory(repo: repo, dirPath: file.path, to: repoPath)
                
            case "file" where isEssentialFile(file.path):
                logger.info("📥 Downloading \(file.path)")
                try await downloadFile(
                    from: repo,
                    path: file.path,
                    to: repoPath.appendingPathComponent(file.path),
                    expectedSize: file.size,
                    config: .default
                )
                
            default:
                break // Skip other files/directories
            }
        }
        
        logger.info("✅ Downloaded all models for \(repo.folderName)")
    }
    
    /// Check if a file is essential for model operation
    private static func isEssentialFile(_ path: String) -> Bool {
        path.hasSuffix(".json") || path.hasSuffix(".txt") || path == "config.json"
    }
    
    /// Clean up incomplete downloads in a directory
    private static func cleanupIncompleteDownloads(at directory: URL) throws {
        guard FileManager.default.fileExists(atPath: directory.path) else { return }
        
        let contents = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
        contents?.filter { $0.pathExtension == "download" }.forEach { file in
            logger.info("🧹 Cleaning up: \(file.lastPathComponent)")
            try? FileManager.default.removeItem(at: file)
        }
    }
    
    /// List files in a HuggingFace repository
    private static func listRepoFiles(_ repo: Repo, path: String = "") async throws -> [RepoFile] {
        let apiPath = path.isEmpty ? "tree/main" : "tree/main/\(path)"
        let apiURL = URL(string: "https://huggingface.co/api/models/\(repo.rawValue)/\(apiPath)")!
        
        var request = URLRequest(url: apiURL)
        request.timeoutInterval = 30
        
        let (data, response) = try await URLSession.shared.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        return try JSONDecoder().decode([RepoFile].self, from: data)
    }
    
    /// Download a CoreML model directory and all its contents
    private static func downloadModelDirectory(repo: Repo, dirPath: String, to destination: URL) async throws {
        let modelDir = destination.appendingPathComponent(dirPath)
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        
        let files = try await listRepoFiles(repo, path: dirPath)
        
        for item in files {
            switch item.type {
            case "directory":
                try await downloadModelDirectory(repo: repo, dirPath: item.path, to: destination)
                
            case "file":
                let expectedSize = item.lfs?.size ?? item.size
                guard shouldDownloadFile(path: item.path, size: expectedSize) else { continue }
                
                logger.info("📥 Downloading \(item.path) (\(formatBytes(expectedSize)))")
                
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
    
    /// Determine if a file should be downloaded based on its path and size
    private static func shouldDownloadFile(path: String, size: Int) -> Bool {
        let essentialExtensions = [".bin", ".mlmodel", ".mil", ".json", ".weights"]
        let isEssential = essentialExtensions.contains { path.hasSuffix($0) }
        
        if size < 50 && !isEssential {
            logger.debug("Skipping small file: \(path) (\(size) bytes)")
            return false
        }
        return true
    }
    
    /// Create a progress handler for large files
    private static func createProgressHandler(for path: String, size: Int) -> ProgressHandler? {
        guard size > 50_000_000 else { return nil }
        
        let fileName = path.split(separator: "/").last ?? ""
        var lastReportedPercentage = 0
        
        return { progress in
            let percentage = Int(progress * 100)
            if percentage >= lastReportedPercentage + 10 {
                lastReportedPercentage = percentage
                logger.info("   Progress: \(percentage)% of \(fileName)")
                print("   ⏳ \(percentage)% downloaded of \(fileName)")
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
        
        // Download with retries
        for attempt in 0..<config.maxRetries {
            do {
                if attempt > 0 {
                    logger.info("🔄 Retry \(attempt + 1)/\(config.maxRetries)")
                    // Get current file size for resume
                    startByte = (try? FileManager.default.attributesOfItem(atPath: tempURL.path)[.size] as? Int64) ?? 0
                }
                
                try await performChunkedDownload(
                    from: downloadURL,
                    to: tempURL,
                    startByte: startByte,
                    expectedSize: Int64(expectedSize),
                    config: config,
                    progressHandler: progressHandler
                )
                
                // Move completed file
                try? FileManager.default.removeItem(at: destination)
                try FileManager.default.moveItem(at: tempURL, to: destination)
                logger.info("✅ Downloaded \(path)")
                return
                
            } catch let error as URLError where error.code == .fileDoesNotExist {
                throw error // Don't retry 404s
                
            } catch {
                logger.error("❌ Attempt \(attempt + 1) failed: \(error)")
                if attempt < config.maxRetries - 1 {
                    try await Task.sleep(nanoseconds: UInt64(attempt + 1) * 1_000_000_000)
                } else {
                    throw error
                }
            }
        }
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
        let type: String
        let path: String
        let size: Int
        let lfs: LFSInfo?
        
        struct LFSInfo: Codable {
            let size: Int
            let sha256: String?  // Some repos might have this
            let oid: String?     // Most use this instead
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
