import Foundation
import CoreML
import OSLog

/// Simple git-based model downloader for HuggingFace repos
public class DownloadUtils {

    private static let logger = Logger(subsystem: "com.fluidaudio", category: "DownloadUtils")

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

            // 2nd attempt after fresh clone
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

        // Check if git is available
        let gitPath = "/usr/bin/git"
        if FileManager.default.fileExists(atPath: gitPath) {
            // Use git if available (faster for large repos with many files)
            logger.info("Using git for faster download")
            try await downloadRepoWithGit(repo, to: directory, gitPath: gitPath)
        } else {
            // Fallback to URLSession for users without git
            logger.info("Git not found, using URLSession download method")
            try await downloadRepoWithURLSession(repo, to: directory)
        }
    }
    
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
    
    /// Download a HuggingFace repository using URLSession (for users without git)
    private static func downloadRepoWithURLSession(_ repo: Repo, to directory: URL) async throws {
        logger.info("📥 Downloading \(repo.folderName) using URLSession...")
        
        let repoPath = directory.appendingPathComponent(repo.folderName)
        try FileManager.default.createDirectory(at: repoPath, withIntermediateDirectories: true)
        
        // HuggingFace API to get file list
        let apiURL = URL(string: "https://huggingface.co/api/models/\(repo.rawValue)/tree/main")!
        
        let (data, response) = try await URLSession.shared.data(from: apiURL)
        
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        
        // Parse JSON response
        struct HFFile: Codable {
            let path: String
            let lfs: LFSInfo?
            
            struct LFSInfo: Codable {
                let size: Int
                let sha256: String
                let pointer_size: Int
            }
        }
        
        let files = try JSONDecoder().decode([HFFile].self, from: data)
        
        // Download each file
        for file in files {
            // Skip non-essential files
            if file.path.hasPrefix(".") || file.path == "README.md" || file.path == "config.json" {
                continue
            }
            
            // Only download .mlmodelc files (the actual models we need)
            guard file.path.hasSuffix(".mlmodelc") else {
                continue
            }
            
            logger.info("📥 Downloading \(file.path) (\(file.lfs?.size ?? 0) bytes)...")
            
            let fileURL = repoPath.appendingPathComponent(file.path)
            
            // Create subdirectories if needed
            let fileDir = fileURL.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: fileDir, withIntermediateDirectories: true)
            
            // Download URL - use resolve endpoint for LFS files
            let downloadURL = URL(string: "https://huggingface.co/\(repo.rawValue)/resolve/main/\(file.path)")!
            
            do {
                let (fileData, fileResponse) = try await URLSession.shared.data(from: downloadURL)
                
                guard let httpFileResponse = fileResponse as? HTTPURLResponse, 
                      httpFileResponse.statusCode == 200 else {
                    throw URLError(.badServerResponse, userInfo: [
                        NSLocalizedDescriptionKey: "Failed to download \(file.path): HTTP \(httpFileResponse.statusCode)"
                    ])
                }
                
                try fileData.write(to: fileURL)
                logger.info("✅ Downloaded \(file.path) (\(fileData.count) bytes)")
            } catch {
                logger.error("❌ Failed to download \(file.path): \(error)")
                throw error
            }
        }
        
        logger.info("✅ Downloaded all files for \(repo.folderName)")
    }

}
