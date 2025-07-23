import Foundation
import CoreML
import OSLog

/// Dead simple model downloader
public class DownloadUtils {
    
    private static let logger = Logger(subsystem: "com.fluidaudio", category: "DownloadUtils")
    
    /// Model repos
    public enum Repo: String {
        case vad = "FluidInference/silero-vad-coreml"
        case parakeet = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
        case diarizer = "FluidInference/speaker-diarization-coreml"
        
        var url: String { "https://huggingface.co/\(rawValue)" }
        var folderName: String { rawValue.split(separator: "/").last!.description }
    }
    
    /// Download repo and load models
    public static func loadModels(
        _ repo: Repo,
        modelNames: [String],
        directory: URL
    ) async throws -> [String: MLModel] {
        // Download repo if needed
        let repoPath = directory.appendingPathComponent(repo.folderName)
        if !FileManager.default.fileExists(atPath: repoPath.path) {
            try await cloneRepo(repo, to: directory)
        }
        
        // Load models
        var models: [String: MLModel] = [:]
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        
        for name in modelNames {
            let path = repoPath.appendingPathComponent(name)
            models[name] = try MLModel(contentsOf: path, configuration: config)
        }
        
        return models
    }
    
    /// Clone repo using git
    private static func cloneRepo(_ repo: Repo, to directory: URL) async throws {
        logger.info("📥 Downloading \(repo.folderName)...")
        print("📥 Downloading \(repo.folderName)...")
        
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["clone", "--depth", "1", repo.url, repo.folderName]
        process.currentDirectoryURL = directory
        
        try process.run()
        process.waitUntilExit()
        
        guard process.terminationStatus == 0 else {
            throw URLError(.cannotConnectToHost)
        }
        
        logger.info("✅ Downloaded \(repo.folderName)")
    }
    
    // MARK: - Legacy support (keeping existing code working)
    
    public struct ModelRepo {
        let name: String
        let url: String
        static let vad = ModelRepo(name: "silero-vad-coreml", url: "https://huggingface.co/FluidInference/silero-vad-coreml")
        static let parakeet = ModelRepo(name: "parakeet-tdt-0.6b-v2-coreml", url: "https://huggingface.co/FluidInference/parakeet-tdt-0.6b-v2-coreml")
        static let diarizer = ModelRepo(name: "speaker-diarization-coreml", url: "https://huggingface.co/FluidInference/speaker-diarization-coreml")
    }
    
    public struct ModelConfig {
        let repoPath: String
        let modelName: String
        let requiredFiles: [String]
        
        static let vadModels = [
            ModelConfig(repoPath: "", modelName: "silero_stft.mlmodelc", requiredFiles: []),
            ModelConfig(repoPath: "", modelName: "silero_encoder.mlmodelc", requiredFiles: []),
            ModelConfig(repoPath: "", modelName: "silero_rnn_decoder.mlmodelc", requiredFiles: [])
        ]
        
        static let parakeetModels = [
            ModelConfig(repoPath: "", modelName: "Melspectogram", requiredFiles: []),
            ModelConfig(repoPath: "", modelName: "ParakeetEncoder", requiredFiles: []),
            ModelConfig(repoPath: "", modelName: "ParakeetDecoder", requiredFiles: []),
            ModelConfig(repoPath: "", modelName: "RNNTJoint", requiredFiles: [])
        ]
        
        static let diarizerModels = [
            ModelConfig(repoPath: "", modelName: "pyannote_segmentation.mlmodelc", requiredFiles: []),
            ModelConfig(repoPath: "", modelName: "wespeaker.mlmodelc", requiredFiles: [])
        ]
    }
    
    public static func downloadRepoIfNeeded(_ repo: ModelRepo, to baseDirectory: URL) async throws -> URL {
        let repoPath = baseDirectory.appendingPathComponent(repo.name)
        if !FileManager.default.fileExists(atPath: repoPath.path) {
            let r = repo.name == "silero-vad-coreml" ? Repo.vad : 
                    repo.name == "parakeet-tdt-0.6b-v2-coreml" ? Repo.parakeet : Repo.diarizer
            try await cloneRepo(r, to: baseDirectory)
        }
        return repoPath
    }
    
    public static func loadModels(modelNames: [String], from repoDirectory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws -> [String: MLModel] {
        var models: [String: MLModel] = [:]
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        
        for name in modelNames {
            let path = repoDirectory.appendingPathComponent(name)
            models[name] = try MLModel(contentsOf: path, configuration: config)
        }
        return models
    }
    
    public static func loadModelsWithRecovery(repo: ModelRepo, modelNames: [String], baseDirectory: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, maxRetries: Int = 2) async throws -> [String: MLModel] {
        let r = repo.name == "silero-vad-coreml" ? Repo.vad : 
                repo.name == "parakeet-tdt-0.6b-v2-coreml" ? Repo.parakeet : Repo.diarizer
        
        for attempt in 0...maxRetries {
            do {
                return try await loadModels(r, modelNames: modelNames, directory: baseDirectory)
            } catch {
                if attempt == maxRetries { throw error }
                try? FileManager.default.removeItem(at: baseDirectory.appendingPathComponent(r.folderName))
            }
        }
        throw URLError(.unknown)
    }
    
    public static func loadModelsWithRecovery(at directory: URL, configs: [ModelConfig], computeUnits: MLComputeUnits = .cpuAndNeuralEngine, maxRetries: Int = 2) async throws -> [String: MLModel] {
        let repo: ModelRepo = configs.first?.modelName.contains("silero") == true ? .vad :
                              configs.first?.modelName.contains("Parakeet") == true || configs.first?.modelName == "Melspectogram" ? .parakeet : .diarizer
        
        return try await loadModelsWithRecovery(repo: repo, modelNames: configs.map { $0.modelName }, baseDirectory: directory.deletingLastPathComponent(), computeUnits: computeUnits, maxRetries: maxRetries)
    }
    
    public static func checkModelFiles(in directory: URL, modelNames: [String]) throws -> [String] {
        modelNames.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
    
    public static func isModelCompiled(at url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }
    
    public static func downloadVadModelFolder(folderName: String, to folderPath: URL) async throws {
        logger.warning("downloadVadModelFolder is deprecated")
    }
    
    public static func performModelRecovery(modelPaths: [URL], downloadAction: @Sendable () async throws -> Void) async throws {
        try await downloadAction()
    }
    
    public static func downloadMLModelBundle(repoPath: String, modelName: String, outputPath: URL) async throws {
        logger.warning("downloadMLModelBundle is deprecated")
    }
    
    public static func loadModelsWithAutoRecovery(modelPaths: [(url: URL, name: String)], config: MLModelConfiguration, maxRetries: Int = 2, recoveryAction: @Sendable () async throws -> Void) async throws -> [MLModel] {
        for attempt in 0...maxRetries {
            do {
                return try modelPaths.map { try MLModel(contentsOf: $0.url, configuration: config) }
            } catch {
                if attempt == maxRetries { throw error }
                try await recoveryAction()
            }
        }
        throw URLError(.unknown)
    }
    
    public static func downloadParakeetModelsIfNeeded(to modelsDirectory: URL) async throws {
        let repoDir = try await downloadRepoIfNeeded(.parakeet, to: modelsDirectory.deletingLastPathComponent())
        
        for modelName in ["Melspectogram", "ParakeetEncoder", "ParakeetDecoder", "RNNTJoint"] {
            let source = repoDir.appendingPathComponent(modelName)
            let dest = modelsDirectory.appendingPathComponent(modelName)
            
            if !FileManager.default.fileExists(atPath: dest.path) && FileManager.default.fileExists(atPath: source.path) {
                try FileManager.default.copyItem(at: source, to: dest)
            }
        }
    }
    
    public static func downloadVocabularySync(from urlString: String, to destinationPath: URL) throws {
        let data = try Data(contentsOf: URL(string: urlString)!)
        try data.write(to: destinationPath)
    }
}