//
//  VadModels.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct VadModels: @unchecked Sendable {
    public let stft: MLModel
    public let encoder: MLModel
    public let rnn: MLModel
    public let configuration: MLModelConfiguration
    
    private static let logger = Logger(subsystem: "com.fluidinfluence.vad", category: "VadModels")
    
    public init(
        stft: MLModel,
        encoder: MLModel,
        rnn: MLModel,
        configuration: MLModelConfiguration
    ) {
        self.stft = stft
        self.encoder = encoder
        self.rnn = rnn
        self.configuration = configuration
    }
}

@available(macOS 13.0, iOS 16.0, *)
extension VadModels {
    
    public enum ModelNames {
        public static let stft = "silero_stft.mlmodelc"
        public static let encoder = "silero_encoder.mlmodelc"
        public static let rnn = "silero_rnn_decoder.mlmodelc"
    }
    
    /// Get the base FluidAudio directory
    private static func baseDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("FluidAudio")
    }
    
    /// Load VAD models, downloading them if necessary
    /// - Parameters:
    ///   - directory: Base directory for model storage (defaults to Application Support)
    ///   - configuration: MLModel configuration to use (defaults to optimized settings)
    /// - Returns: Loaded VAD models
    public static func load(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> VadModels {
        let baseDir = directory ?? baseDirectory()
        let config = configuration ?? defaultConfiguration()
        
        // Download/load models using DownloadUtils (handles caching automatically)
        let models = try await DownloadUtils.loadModels(
            .vad,
            modelNames: [ModelNames.stft, ModelNames.encoder, ModelNames.rnn],
            directory: baseDir,
            computeUnits: config.computeUnits
        )
        
        // Create VadModels from loaded models
        guard let stftModel = models[ModelNames.stft],
              let encoderModel = models[ModelNames.encoder],
              let rnnModel = models[ModelNames.rnn] else {
            throw VadModelsError.loadingFailed("Failed to load all VAD models")
        }
        
        return VadModels(
            stft: stftModel,
            encoder: encoderModel,
            rnn: rnnModel,
            configuration: config
        )
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
extension VadModels {
    
    /// Alias for load() - kept for backward compatibility
    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> VadModels {
        return try await load(from: directory, configuration: configuration)
    }
    
    public static func modelsExist(at directory: URL) -> Bool {
        // If directory is already the repo, use it; otherwise append repo name
        let repoPath = directory.lastPathComponent == "silero-vad-coreml"
            ? directory
            : directory.appendingPathComponent("silero-vad-coreml")
        
        guard FileManager.default.fileExists(atPath: repoPath.path) else {
            return false
        }
        
        // Check all required files exist
        let requiredFiles = [
            ModelNames.stft,
            ModelNames.encoder,
            ModelNames.rnn
        ]
        
        return requiredFiles.allSatisfy { fileName in
            FileManager.default.fileExists(
                atPath: repoPath.appendingPathComponent(fileName).path
            )
        }
    }
    
    public static func defaultCacheDirectory() -> URL {
        // Return the repo directory directly
        return baseDirectory().appendingPathComponent("silero-vad-coreml", isDirectory: true)
    }
}

public enum VadModelsError: LocalizedError {
    case modelNotFound(String, URL)
    case downloadFailed(String)
    case loadingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name, let path):
            return "VAD model '\(name)' not found at: \(path.path)"
        case .downloadFailed(let reason):
            return "Failed to download VAD models: \(reason)"
        case .loadingFailed(let reason):
            return "Failed to load VAD models: \(reason)"
        }
    }
}