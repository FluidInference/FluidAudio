//
//  AsrModels.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct AsrModels: @unchecked Sendable {
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

    /// Get the base FluidAudio directory
    private static func baseDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("FluidAudio")
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
            (jointPath, "Joint")
        ]

        for (path, name) in requiredPaths {
            guard fileManager.fileExists(atPath: path.path) else {
                throw AsrModelsError.modelNotFound(name, path)
            }
        }
        async let melModel = MLModel.load(contentsOf: melPath, configuration: config)
        async let encoderModel = MLModel.load(contentsOf: encoderPath, configuration: config)
        async let decoderModel = MLModel.load(contentsOf: decoderPath, configuration: config)
        async let jointModel = MLModel.load(contentsOf: jointPath, configuration: config)

        let models = try await AsrModels(
            melspectrogram: melModel,
            encoder: encoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: config
        )

        logger.info("Successfully loaded all ASR models")
        return models
    }


    public static func defaultConfiguration() -> MLModelConfiguration {
        let config = MLModelConfiguration()
        config.allowLowPrecisionAccumulationOnGPU = true
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        config.computeUnits = isCI ? .cpuAndNeuralEngine : .all

        return config
    }
}


@available(macOS 13.0, iOS 16.0, *)
extension AsrModels {

    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        let baseDir = directory ?? baseDirectory()

        // Download/load models
        let models = try await DownloadUtils.loadModels(
            .parakeet,
            modelNames: [ModelNames.melspectrogram, ModelNames.encoder, ModelNames.decoder, ModelNames.joint],
            directory: baseDir,
            computeUnits: configuration?.computeUnits ?? defaultConfiguration().computeUnits
        )

        // Create AsrModels from loaded models
        guard let melModel = models[ModelNames.melspectrogram],
              let encoderModel = models[ModelNames.encoder],
              let decoderModel = models[ModelNames.decoder],
              let jointModel = models[ModelNames.joint] else {
            throw AsrModelsError.loadingFailed("Failed to load all ASR models")
        }

        return AsrModels(
            melspectrogram: melModel,
            encoder: encoderModel,
            decoder: decoderModel,
            joint: jointModel,
            configuration: configuration ?? defaultConfiguration()
        )
    }

    public static func modelsExist(at directory: URL) -> Bool {
        // If directory is already the repo, use it; otherwise append repo name
        let repoPath = directory.lastPathComponent == "parakeet-tdt-0.6b-v2-coreml"
            ? directory
            : directory.appendingPathComponent("parakeet-tdt-0.6b-v2-coreml")

        guard FileManager.default.fileExists(atPath: repoPath.path) else {
            return false
        }

        // Check all required files exist
        let requiredFiles = [
            ModelNames.melspectrogram,
            ModelNames.encoder,
            ModelNames.decoder,
            ModelNames.joint,
            ModelNames.vocabulary
        ]

        return requiredFiles.allSatisfy { fileName in
            FileManager.default.fileExists(
                atPath: repoPath.appendingPathComponent(fileName).path
            )
        }
    }

    public static func defaultCacheDirectory() -> URL {
        // Return the repo directory directly
        return baseDirectory().appendingPathComponent("parakeet-tdt-0.6b-v2-coreml", isDirectory: true)
    }
}


public enum AsrModelsError: LocalizedError {
    case modelNotFound(String, URL)
    case downloadFailed(String)
    case loadingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name, let path):
            return "ASR model '\(name)' not found at: \(path.path)"
        case .downloadFailed(let reason):
            return "Failed to download ASR models: \(reason)"
        case .loadingFailed(let reason):
            return "Failed to load ASR models: \(reason)"
        }
    }
}
