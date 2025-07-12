import Foundation
import CoreML

/// Utility class for downloading CoreML models from Hugging Face
public class DownloadUtils {

    /// Download a complete .mlmodelc bundle from Hugging Face
    public static func downloadMLModelBundle(
        repoPath: String,
        modelName: String,
        outputPath: URL
    ) async throws {
        try FileManager.default.createDirectory(at: outputPath, withIntermediateDirectories: true)

        let bundleFiles = [
            "model.mil",
            "coremldata.bin",
            "metadata.json"
        ]

        let weightFiles = [
            "weights/weight.bin"
        ]

        for fileName in bundleFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(fileName)")!

            do {
                let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

                if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try? FileManager.default.removeItem(at: destinationPath)
                    try FileManager.default.moveItem(at: tempFile, to: destinationPath)
                } else {
                    // Create minimal versions for optional files
                    if fileName == "metadata.json" {
                        let destinationPath = outputPath.appendingPathComponent(fileName)
                        try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                    }
                }
            } catch {
                // For critical files, create minimal versions
                if fileName == "coremldata.bin" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try Data().write(to: destinationPath)
                } else if fileName == "metadata.json" {
                    let destinationPath = outputPath.appendingPathComponent(fileName)
                    try "{}".write(to: destinationPath, atomically: true, encoding: .utf8)
                }
            }
        }

        // Download weight files
        for weightFile in weightFiles {
            let fileURL = URL(string: "https://huggingface.co/\(repoPath)/resolve/main/\(modelName)/\(weightFile)")!

            let (tempFile, response) = try await URLSession.shared.download(from: fileURL)

            if let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 {
                let destinationPath = outputPath.appendingPathComponent(weightFile)
                let weightsDir = destinationPath.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: weightsDir, withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: destinationPath)
                try FileManager.default.moveItem(at: tempFile, to: destinationPath)
            }
        }
    }

    /// Download VAD model folder structure from Hugging Face
    public static func downloadVadModelFolder(
        folderName: String,
        to folderPath: URL
    ) async throws {
        // Create the folder
        try FileManager.default.createDirectory(at: folderPath, withIntermediateDirectories: true)

        // Download the main files inside the .mlmodelc folder
        let modelFiles = [
            "coremldata.bin",
            "model.espresso.net",
            "model.espresso.shape",
            "model.espresso.weights"
        ]

        let baseURL = "https://huggingface.co/alexwengg/coreml-silero-vad/resolve/main/\(folderName)"

        for fileName in modelFiles {
            let fileURL = "\(baseURL)/\(fileName)"
            let destinationPath = folderPath.appendingPathComponent(fileName)
            try await downloadFile(from: fileURL, to: destinationPath)
        }

        // Download subdirectory files required by CoreML
        let subDirs = ["analytics", "model", "neural_network_optionals"]
        for subDir in subDirs {
            let subDirPath = folderPath.appendingPathComponent(subDir)
            try FileManager.default.createDirectory(at: subDirPath, withIntermediateDirectories: true)

            let subFileURL = "\(baseURL)/\(subDir)/coremldata.bin"
            let subDestinationPath = subDirPath.appendingPathComponent("coremldata.bin")

            do {
                try await downloadFile(from: subFileURL, to: subDestinationPath)
            } catch {
                // Some subdirectory files might be optional
            }
        }
    }

    /// Download a single file from URL
    public static func downloadFile(from urlString: String, to destinationPath: URL) async throws {
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }

        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw URLError(.badServerResponse)
        }

        guard httpResponse.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }

        try data.write(to: destinationPath)
    }

    /// Perform model recovery by deleting corrupted models and re-downloading
    public static func performModelRecovery(
        modelPaths: [URL],
        downloadAction: @Sendable () async throws -> Void
    ) async throws {
        // Remove potentially corrupted model files
        for modelPath in modelPaths {
            if FileManager.default.fileExists(atPath: modelPath.path) {
                try FileManager.default.removeItem(at: modelPath)
            }
        }

        // Re-download the models
        try await downloadAction()
    }

    /// Load models with automatic recovery on compilation failures
    public static func loadModelsWithAutoRecovery(
        modelPaths: [(url: URL, name: String)],
        config: MLModelConfiguration,
        maxRetries: Int = 2,
        recoveryAction: @Sendable () async throws -> Void
    ) async throws -> [MLModel] {
        var attempt = 0

        while attempt <= maxRetries {
            do {
                var models: [MLModel] = []

                for (modelURL, _) in modelPaths {
                    let model = try MLModel(contentsOf: modelURL, configuration: config)
                    models.append(model)
                }

                return models

            } catch {
                if attempt >= maxRetries {
                    throw error
                }

                // Auto-recovery: Delete corrupted models and re-download
                try await performModelRecovery(
                    modelPaths: modelPaths.map { $0.url },
                    downloadAction: recoveryAction
                )

                attempt += 1
            }
        }

        // This should never be reached, but Swift requires it
        throw URLError(.unknown)
    }

    /// Check if a model is properly compiled
    public static func isModelCompiled(at url: URL) -> Bool {
        let coreMLDataPath = url.appendingPathComponent("coremldata.bin")
        return FileManager.default.fileExists(atPath: coreMLDataPath.path)
    }

    /// Check for missing or corrupted models
    public static func checkModelFiles(in directory: URL, modelNames: [String]) throws -> [String] {
        var missingModels: [String] = []

        for modelName in modelNames {
            let modelPath = directory.appendingPathComponent(modelName)

            if !FileManager.default.fileExists(atPath: modelPath.path) {
                missingModels.append(modelName)
            } else {
                // Check for corrupted or incomplete downloads
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: modelPath.path)
                    if let fileSize = attributes[.size] as? Int64, fileSize < 1000 {
                        missingModels.append(modelName)
                        try? FileManager.default.removeItem(at: modelPath)
                    }
                } catch {
                    missingModels.append(modelName)
                }
            }
        }

        return missingModels
    }
}
