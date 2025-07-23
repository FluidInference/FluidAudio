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

    public static func loadFromCache(
        configuration: MLModelConfiguration? = nil
    ) async throws -> AsrModels {
        let cacheDir = defaultCacheDirectory()
        return try await load(from: cacheDir, configuration: configuration)
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

        try await DownloadUtils.downloadParakeetModelsIfNeeded(to: targetDir)

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
            ModelNames.joint
        ]
        // Simplified: just check if model folders exist
        let modelsPresent = modelFiles.allSatisfy { fileName in
            let path = directory.appendingPathComponent(fileName)
            return fileManager.fileExists(atPath: path.path)
        }
        // Note: vocab file is checked separately as it's a text file, not a model
        let vocabPath = directory.deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent(ModelNames.vocabulary)
        let vocabPresent = fileManager.fileExists(atPath: vocabPath.path)

        return modelsPresent && vocabPresent
    }

    public static func defaultCacheDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
            .appendingPathComponent("Parakeet", isDirectory: true)
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
