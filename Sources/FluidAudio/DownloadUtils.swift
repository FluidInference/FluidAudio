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
            try? FileManager.default.removeItem(at: repoPath)

            // 2nd attempt after fresh clone
            return try await loadModelsOnce(repo, modelNames: modelNames,
                                            directory: directory, computeUnits: computeUnits)
        }
    }


    /// Download repo (if needed) and load CoreML models
    /// - Parameters:
    ///   - repo: The HuggingFace repository to download
    ///   - modelNames: Array of model file names to load (e.g., ["model.mlmodelc"])
    ///   - directory: Base directory to store repos (e.g., ~/Library/Application Support/FluidAudio)
    ///   - computeUnits: CoreML compute units to use (default: CPU and Neural Engine)
    /// - Returns: Dictionary mapping model names to loaded MLModel instances
    public static func loadModelsOnce(
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
            try await cloneRepo(repo, to: directory)
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

    /// Clone a HuggingFace repository using git
    private static func cloneRepo(_ repo: Repo, to directory: URL) async throws {
        logger.info("📥 Downloading \(repo.folderName) from HuggingFace...")
        print("📥 Downloading \(repo.folderName)...")

        // Use a temporary directory for cloning
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)

        defer {
            try? FileManager.default.removeItem(at: tempDir)
        }

        // Clone to temp directory first
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
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
                NSLocalizedDescriptionKey: "Failed to clone repository: \(errorMessage)"
            ])
        }

        // Check if Git LFS files need to be pulled
        let clonedPath = tempDir.appendingPathComponent(repo.folderName)
        let gitAttributesPath = clonedPath.appendingPathComponent(".gitattributes")

        if FileManager.default.fileExists(atPath: gitAttributesPath.path) {
            logger.info("📦 Pulling Git LFS files...")

            let lfsProcess = Process()
            lfsProcess.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            lfsProcess.arguments = ["lfs", "pull"]
            lfsProcess.currentDirectoryURL = clonedPath

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
            logger.warning("⚠️ Unexpected: directory exists at \(finalPath.lastPathComponent) during clone")
            try FileManager.default.removeItem(at: finalPath)
        }
        
        try FileManager.default.moveItem(at: clonedPath, to: finalPath)

        logger.info("✅ Downloaded \(repo.folderName)")
        print("✅ Download complete")
    }

}
