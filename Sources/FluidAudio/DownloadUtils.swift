import CoreML
import Foundation
import Hub

/// Utility class for downloading CoreML models from Hugging Face
public class DownloadUtils {

    /// Download a complete .mlmodelc bundle from Hugging Face using Hub.snapshot
    public static func downloadMLModelBundle(
        repoPath: String,
        modelName: String,
        outputPath: URL
    ) async throws {
        let repo = Hub.Repo(id: repoPath)

        // The models are stored as .mlmodelc folders directly at the root level
        let modelFolderName = "\(modelName).mlmodelc"

        do {
            // Download the entire repository to get all models
            print("🔄 Downloading entire repository snapshot...")
            let snapshotURL = try await Hub.snapshot(from: repo)
            print("✅ Downloaded snapshot to: \(snapshotURL.path)")

            let sourceModelPath = snapshotURL.appendingPathComponent(modelFolderName)
            print("📁 Looking for model at: \(sourceModelPath.path)")

            // Check if the source model path exists
            if FileManager.default.fileExists(atPath: sourceModelPath.path) {
                print("✅ Source model found at: \(sourceModelPath.path)")
            } else {
                print("❌ Source model NOT found at: \(sourceModelPath.path)")
                // List what's actually in the snapshot directory
                do {
                    let contents = try FileManager.default.contentsOfDirectory(
                        atPath: snapshotURL.path)
                    print("📂 Snapshot directory contents: \(contents)")
                } catch {
                    print("❌ Could not list snapshot directory contents: \(error)")
                }

                throw NSError(
                    domain: "FluidAudioDownloadError",
                    code: 1003,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Model \(modelFolderName) not found in downloaded repository"
                    ]
                )
            }

            // Create output directory with proper error handling for iOS sandboxing
            do {
                try FileManager.default.createDirectory(
                    at: outputPath.deletingLastPathComponent(), withIntermediateDirectories: true)
            } catch {
                #if os(iOS)
                    // On iOS, provide more context for debugging sandboxing issues
                    throw NSError(
                        domain: "FluidAudioDownloadError",
                        code: 1001,
                        userInfo: [
                            NSLocalizedDescriptionKey:
                                "Failed to create model directory on iOS. Path: \(outputPath.path)",
                            NSUnderlyingErrorKey: error,
                        ]
                    )
                #else
                    throw error
                #endif
            }

            // Remove existing model if it exists
            try? FileManager.default.removeItem(at: outputPath)

            // Copy (not move) the downloaded model to the target location to preserve the original
            try FileManager.default.copyItem(at: sourceModelPath, to: outputPath)
            print("✅ Model copied to: \(outputPath.path)")

        } catch {
            throw NSError(
                domain: "FluidAudioDownloadError",
                code: 1002,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Failed to download model bundle \(modelName) from \(repoPath)",
                    NSUnderlyingErrorKey: error,
                ]
            )
        }
    }

    /// Download VAD model folder structure from Hugging Face using Hub.snapshot
    public static func downloadVadModelFolder(
        folderName: String,
        to folderPath: URL
    ) async throws {
        let repo = Hub.Repo(id: "FluidInference/silero-vad-coreml")

        do {
            // Download the entire repository
            print("🔄 Downloading VAD repository snapshot...")
            let snapshotURL = try await Hub.snapshot(from: repo)
            print("✅ Downloaded VAD snapshot to: \(snapshotURL.path)")

            let sourceModelPath = snapshotURL.appendingPathComponent(folderName)

            // Create output directory
            try FileManager.default.createDirectory(
                at: folderPath.deletingLastPathComponent(), withIntermediateDirectories: true)

            // Remove existing model if it exists
            try? FileManager.default.removeItem(at: folderPath)

            // Copy the downloaded model to the target location
            try FileManager.default.copyItem(at: sourceModelPath, to: folderPath)
            print("✅ VAD model copied to: \(folderPath.path)")

        } catch {
            throw NSError(
                domain: "FluidAudioDownloadError",
                code: 1004,
                userInfo: [
                    NSLocalizedDescriptionKey: "Failed to download VAD model folder \(folderName)",
                    NSUnderlyingErrorKey: error,
                ]
            )
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

    /// Load models with automatic recovery on compilation failures
    public static func withAutoRecovery<T>(
        maxRetries: Int = 2,
        makeAttempt: () async throws -> T,
        recovery: @Sendable () async throws -> Void
    ) async throws -> T {

        var attempt = 0
        while attempt <= maxRetries {
            do {
                return try await makeAttempt()
            } catch {
                if attempt >= maxRetries {
                    throw error
                }
                try await recovery()
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
                    let attributes = try FileManager.default.attributesOfItem(
                        atPath: modelPath.path)
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

    public static func downloadVocabularySync(from urlString: String, to destinationPath: URL)
        throws
    {
        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }

        let data = try Data(contentsOf: url)
        try data.write(to: destinationPath)
    }

    public static func downloadParakeetModelsIfNeeded(to modelsDirectory: URL) async throws {
        let models = [
            ("Melspectogram", modelsDirectory.appendingPathComponent("Melspectogram.mlmodelc")),
            ("ParakeetEncoder", modelsDirectory.appendingPathComponent("ParakeetEncoder.mlmodelc")),
            ("ParakeetDecoder", modelsDirectory.appendingPathComponent("ParakeetDecoder.mlmodelc")),
            ("RNNTJoint", modelsDirectory.appendingPathComponent("RNNTJoint.mlmodelc")),
        ]

        var missingModels: [String] = []
        for (name, path) in models {
            if !FileManager.default.fileExists(atPath: path.path) {
                missingModels.append(name)
                print("Model \(name) not found at \(path.path)")
            }
        }

        if !missingModels.isEmpty {
            print("Downloading \(missingModels.count) missing Parakeet models...")

            try FileManager.default.createDirectory(
                at: modelsDirectory, withIntermediateDirectories: true)

            let repoPath = "FluidInference/parakeet-tdt-0.6b-v2-coreml"

            for modelName in missingModels {
                print("Downloading \(modelName)...")

                do {
                    let modelPath = modelsDirectory.appendingPathComponent("\(modelName).mlmodelc")

                    // Download the compiled model bundle
                    try await downloadMLModelBundle(
                        repoPath: repoPath,
                        modelName: modelName,
                        outputPath: modelPath
                    )

                    print("✅ Downloaded \(modelName).mlmodelc")
                } catch {
                    print("Failed to download \(modelName): \(error)")
                    throw error
                }
            }
        } else {
            print("All Parakeet models already present")
        }
    }
}
