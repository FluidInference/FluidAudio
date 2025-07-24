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
            
            // Force delete with better error handling
            do {
                try FileManager.default.removeItem(at: repoPath)
                logger.info("✅ Successfully deleted \(repo.folderName)")
            } catch {
                logger.error("❌ Failed to delete \(repo.folderName): \(error)")
                // Continue anyway - download might overwrite
            }

            // 2nd attempt after fresh clone
            do {
                return try await loadModelsOnce(repo, modelNames: modelNames,
                                                directory: directory, computeUnits: computeUnits)
            } catch {
                logger.error("❌ Second load attempt also failed: \(error)")
                // If it's a CoreML compilation error, add more context
                if let nsError = error as NSError?, nsError.domain == "com.apple.CoreML" {
                    throw URLError(.cannotDecodeContentData, userInfo: [
                        NSLocalizedDescriptionKey: "CoreML model compilation failed. The models may be corrupted or incompatible with this macOS version.",
                        NSUnderlyingErrorKey: error
                    ])
                }
                throw error
            }
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
            
            // Verify model size (CoreML models should be > 1KB at minimum)
            if let attrs = try? FileManager.default.attributesOfItem(atPath: modelPath.path),
               let fileSize = attrs[.size] as? Int64,
               fileSize < 1024 {
                logger.warning("⚠️ Model \(name) seems too small: \(fileSize) bytes")
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
        guard FileManager.default.fileExists(atPath: gitPath) else {
            logger.error("❌ Git not found at \(gitPath)")
            throw URLError(.cannotFindHost, userInfo: [
                NSLocalizedDescriptionKey: "Git is not installed. Please install Xcode Command Line Tools by running 'xcode-select --install' in Terminal."
            ])
        }
        
        // TODO: Consider implementing URLSession-based download as fallback for users without git
        // This would require downloading individual files from HuggingFace's web API

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

}
