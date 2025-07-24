@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct AsrModels {
    public let melspectrogram: MLModel
    public let encoder: MLModel
    public let decoder: MLModel
    public let joint: MLModel
    public let configuration: MLModelConfiguration

    private static let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "AsrModels")

    public init(
        melspectrogram: MLModel,
        encoder: MLModel,
        decoder: MLModel,
        joint: MLModel,
        configuration: MLModelConfiguration
    ) {
        self.melspectrogram = melspectrogram
        self.encoder = encoder
        self.decoder = decoder
        self.joint = joint
        self.configuration = configuration
    }
}

@available(macOS 13.0, iOS 16.0, *)
extension AsrModels {

    public enum ModelNames {
        public static let melspectrogram = "Melspectogram.mlmodelc"
        public static let encoder = "ParakeetEncoder.mlmodelc"
        public static let decoder = "ParakeetDecoder.mlmodelc"
        public static let joint = "RNNTJoint.mlmodelc"
        public static let vocabulary = "parakeet_vocab.json"
    }

    /// Load ASR models from a directory
    /// - Parameters:
    ///   - directory: Directory containing the model files
    ///   - configuration: MLModel configuration to use (defaults to optimized settings)
    /// - Returns: Loaded ASR models
    public static func load(
        from directory: URL,
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        logger.info("Loading ASR models from: \(directory.path)")

        let config = configuration ?? defaultConfiguration()
        let melPath = directory.appendingPathComponent(ModelNames.melspectrogram)
        let encoderPath = directory.appendingPathComponent(ModelNames.encoder)
        let decoderPath = directory.appendingPathComponent(ModelNames.decoder)
        let jointPath = directory.appendingPathComponent(ModelNames.joint)
        let fileManager = FileManager.default
        let requiredPaths = [
            (melPath, "Mel-spectrogram"),
            (encoderPath, "Encoder"),
            (decoderPath, "Decoder"),
            (jointPath, "Joint"),
        ]

        for (path, name) in requiredPaths {
            guard fileManager.fileExists(atPath: path.path) else {
                throw AsrModelsError.modelNotFound(name, path)
            }
        }

        // Validate model files before loading
        try await validateModelFiles(at: directory)
        // Load models with auto-recovery
        let models = try await DownloadUtils.withAutoRecovery(
            maxRetries: 2,
            makeAttempt: {
                async let melModel = MLModel.load(contentsOf: melPath, configuration: config)
                async let encoderModel = MLModel.load(
                    contentsOf: encoderPath, configuration: config)
                async let decoderModel = MLModel.load(
                    contentsOf: decoderPath, configuration: config)
                async let jointModel = MLModel.load(contentsOf: jointPath, configuration: config)

                let models = try await AsrModels(
                    melspectrogram: melModel,
                    encoder: encoderModel,
                    decoder: decoderModel,
                    joint: jointModel,
                    configuration: config
                )

                // Log model descriptions for debugging
                logModelDescription("Melspectrogram", models.melspectrogram)
                logModelDescription("Encoder", models.encoder)
                logModelDescription("Decoder", models.decoder)
                logModelDescription("Joint", models.joint)

                return models
            },
            recovery: {
                logger.warning("ASR models appear corrupted, attempting recovery...")

                // Delete corrupted models
                let modelsToDelete = [melPath, encoderPath, decoderPath, jointPath]
                for modelPath in modelsToDelete {
                    if FileManager.default.fileExists(atPath: modelPath.path) {
                        try FileManager.default.removeItem(at: modelPath)
                    }
                }

                // Re-download models
                try await downloadParakeetModelsIfNeeded(to: directory)
            }
        )

        logger.info("Successfully loaded all ASR models")
        return models
    }

    public static func loadFromCache(
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        let cacheDir = defaultCacheDirectory()

        // Check if we're running in a sandboxed environment
        if isSandboxed() {
            logger.info("Running in sandboxed environment, performing additional validation")

            // Try to load and validate models
            do {
                return try await load(from: cacheDir, configuration: configuration)
            } catch AsrModelsError.modelCorrupted(_, _), AsrModelsError.modelValidationFailed(_, _)
            {
                logger.warning("Corrupted models detected in sandbox, forcing re-download")
                // Force re-download if models are corrupted
                try await download(to: cacheDir, force: true)
                return try await load(from: cacheDir, configuration: configuration)
            }
        }

        return try await load(from: cacheDir, configuration: configuration)
    }

    private static func isSandboxed() -> Bool {
        // Check if we're running in a sandboxed environment by looking at the path
        let cacheDir = defaultCacheDirectory()
        return cacheDir.path.contains("/Library/Containers/")
    }

    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true

        // In sandboxed environments, try CPU-only to avoid E5RT validation issues
        if isSandboxed() {
            logger.info("Using CPU-only configuration for sandboxed environment")
            config.computeUnits = .cpuOnly
        } else {
            config.computeUnits = .cpuAndNeuralEngine
        }

        return config
    }
}

@available(macOS 13.0, iOS 16.0, *)
extension AsrModels {

    @discardableResult
    public static func download(
        to directory: URL? = nil,
        force: Bool = false
    ) async throws -> URL {
        let targetDir = directory ?? defaultCacheDirectory()
        logger.info("Downloading ASR models to: \(targetDir.path)")
        if !force && modelsExist(at: targetDir) {
            logger.info("ASR models already present at: \(targetDir.path)")
            return targetDir
        }

        if force {
            let fileManager = FileManager.default
            if fileManager.fileExists(atPath: targetDir.path) {
                try fileManager.removeItem(at: targetDir)
            }
        }

        try await downloadParakeetModelsIfNeeded(to: targetDir)

        logger.info("Successfully downloaded ASR models")
        return targetDir
    }

    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        let targetDir = try await download(to: directory)
        return try await load(from: targetDir, configuration: configuration)
    }

    public static func modelsExist(at directory: URL) -> Bool {
        let fileManager = FileManager.default
        let modelFiles = [
            ModelNames.melspectrogram,
            ModelNames.encoder,
            ModelNames.decoder,
            ModelNames.joint,
        ]
        let modelsPresent = modelFiles.allSatisfy { fileName in
            let path = directory.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }
        let vocabPath = directory.appendingPathComponent(ModelNames.vocabulary)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    public static func defaultCacheDirectory() -> URL {
        return DownloadUtils.getModelDirectory(for: "models/parakeet")
    }
}

// MARK: - Private Download Utilities

@available(macOS 13.0, iOS 16.0, *)
extension AsrModels {

    fileprivate static func downloadParakeetModelsIfNeeded(
        to modelsDirectory: URL,
        from repoPath: String = "FluidInference/parakeet-tdt-0.6b-v2-coreml"
    ) async throws {

        let models = [
            ("Melspectogram", modelsDirectory.appendingPathComponent(ModelNames.melspectrogram)),
            ("ParakeetEncoder", modelsDirectory.appendingPathComponent(ModelNames.encoder)),
            ("ParakeetDecoder", modelsDirectory.appendingPathComponent(ModelNames.decoder)),
            ("RNNTJoint", modelsDirectory.appendingPathComponent(ModelNames.joint)),
        ]

        var missingModels: [String] = []
        for (name, path) in models {
            if !FileManager.default.fileExists(atPath: path.path) {
                missingModels.append(name)
                logger.info("Model \(name) not found at \(path.path)")
            }
        }

        await DownloadUtils.checkIfConfigExists(repoPath: repoPath)

        if !missingModels.isEmpty {
            logger.info("Downloading \(missingModels.count) missing Parakeet models...")

            try FileManager.default.createDirectory(
                at: modelsDirectory, withIntermediateDirectories: true)

            for modelName in missingModels {
                logger.info("Downloading \(modelName)...")

                do {
                    let modelPath = modelsDirectory.appendingPathComponent("\(modelName).mlmodelc")

                    // Download the compiled model bundle
                    try await DownloadUtils.downloadMLModelBundle(
                        repoPath: repoPath,
                        modelName: modelName,
                        outputPath: modelPath
                    )

                    logger.info("✅ Downloaded \(modelName).mlmodelc")
                } catch {
                    logger.error("Failed to download \(modelName): \(error)")
                    throw error
                }
            }
        }

        let vocabPath = modelsDirectory.appendingPathComponent(ModelNames.vocabulary)
        let vocabURL = "https://huggingface.co/\(repoPath)/resolve/main/\(ModelNames.vocabulary)"
        try await DownloadUtils.downloadFile(from: vocabURL, to: vocabPath)
    }

    /// Validate model files to ensure they're properly formatted and not corrupted
    private static func validateModelFiles(at directory: URL) async throws {
        let models = [
            (ModelNames.melspectrogram, "Melspectrogram"),
            (ModelNames.encoder, "ParakeetEncoder"),
            (ModelNames.decoder, "ParakeetDecoder"),
            (ModelNames.joint, "RNNTJoint"),
        ]

        for (fileName, modelName) in models {
            let modelPath = directory.appendingPathComponent(fileName)

            // Check if the model bundle exists and has required files
            let requiredFiles = ["model.mil", "coremldata.bin"]
            for file in requiredFiles {
                let filePath = modelPath.appendingPathComponent(file)
                guard FileManager.default.fileExists(atPath: filePath.path) else {
                    logger.error("Model \(modelName) missing required file: \(file)")
                    throw AsrModelsError.modelCorrupted(modelName, modelPath)
                }

                // Check file size to ensure it's not empty
                do {
                    let attributes = try FileManager.default.attributesOfItem(atPath: filePath.path)
                    if let fileSize = attributes[.size] as? Int64, fileSize < 100 {
                        logger.error(
                            "Model \(modelName) file \(file) is suspiciously small: \(fileSize) bytes"
                        )
                        throw AsrModelsError.modelCorrupted(modelName, modelPath)
                    }
                } catch {
                    logger.error("Failed to check model file size: \(error)")
                    throw AsrModelsError.modelValidationFailed(modelName, modelPath)
                }
            }
        }

        logger.info("All ASR models passed validation")
    }

    /// Compare model files between two directories (useful for debugging)
    public static func compareModels(cliPath: URL, sandboxPath: URL) {
        logger.info("Comparing models between CLI and sandbox environments...")
        logger.info("CLI path: \(cliPath.path)")
        logger.info("Sandbox path: \(sandboxPath.path)")

        let models = [
            ModelNames.melspectrogram,
            ModelNames.encoder,
            ModelNames.decoder,
            ModelNames.joint,
        ]

        for modelName in models {
            let cliModel = cliPath.appendingPathComponent(modelName)
            let sandboxModel = sandboxPath.appendingPathComponent(modelName)

            do {
                // Compare directory sizes
                let cliSize = try directorySize(at: cliModel)
                let sandboxSize = try directorySize(at: sandboxModel)

                if abs(cliSize - sandboxSize) > 100 {
                    logger.warning(
                        "\(modelName) size mismatch - CLI: \(cliSize) bytes, Sandbox: \(sandboxSize) bytes"
                    )
                } else {
                    logger.info("\(modelName) sizes match: \(cliSize) bytes")
                }
            } catch {
                logger.error("Failed to compare \(modelName): \(error)")
            }
        }
    }

    private static func directorySize(at url: URL) throws -> Int64 {
        var totalSize: Int64 = 0
        let fileManager = FileManager.default

        if let enumerator = fileManager.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey])
        {
            for case let fileURL as URL in enumerator {
                let attributes = try fileURL.resourceValues(forKeys: [.fileSizeKey])
                totalSize += Int64(attributes.fileSize ?? 0)
            }
        }

        return totalSize
    }

    /// Log model input/output descriptions for debugging
    private static func logModelDescription(_ name: String, _ model: MLModel) {
        logger.info("=== \(name) Model Description ===")

        let description = model.modelDescription

        // Log inputs
        logger.info("Inputs:")
        for (inputName, inputDesc) in description.inputDescriptionsByName {
            if let multiArrayConstraint = inputDesc.multiArrayConstraint {
                let shape = multiArrayConstraint.shape.map { $0.intValue }
                logger.info(
                    "  \(inputName): shape=\(shape), dataType=\(multiArrayConstraint.dataType.rawValue)"
                )
            } else {
                logger.info("  \(inputName): type=\(String(describing: inputDesc.type))")
            }
        }

        // Log outputs
        logger.info("Outputs:")
        for (outputName, outputDesc) in description.outputDescriptionsByName {
            if let multiArrayConstraint = outputDesc.multiArrayConstraint {
                let shape = multiArrayConstraint.shape.map { $0.intValue }
                logger.info(
                    "  \(outputName): shape=\(shape), dataType=\(multiArrayConstraint.dataType.rawValue)"
                )
            } else {
                logger.info("  \(outputName): type=\(String(describing: outputDesc.type))")
            }
        }
    }
}

public enum AsrModelsError: LocalizedError {
    case modelNotFound(String, URL)
    case downloadFailed(String)
    case loadingFailed(String)
    case modelValidationFailed(String, URL)
    case modelCorrupted(String, URL)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name, let path):
            return "ASR model '\(name)' not found at: \(path.path)"
        case .downloadFailed(let reason):
            return "Failed to download ASR models: \(reason)"
        case .loadingFailed(let reason):
            return "Failed to load ASR models: \(reason)"
        case .modelValidationFailed(let name, let path):
            return "ASR model '\(name)' failed validation at: \(path.path)"
        case .modelCorrupted(let name, let path):
            return "ASR model '\(name)' appears corrupted at: \(path.path)"
        }
    }
}
