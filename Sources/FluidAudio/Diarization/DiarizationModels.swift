//
//  DiarizationModels.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

@preconcurrency import CoreML
import Foundation
import OSLog

@available(macOS 13.0, iOS 16.0, *)
public struct DiarizationModels: @unchecked Sendable {
    public let segmentation: MLModel
    public let embedding: MLModel
    public let configuration: MLModelConfiguration
    
    private static let logger = Logger(subsystem: "com.fluidinfluence.diarizer", category: "DiarizationModels")
    
    public init(
        segmentation: MLModel,
        embedding: MLModel,
        configuration: MLModelConfiguration = MLModelConfiguration()
    ) {
        self.segmentation = segmentation
        self.embedding = embedding
        self.configuration = configuration
    }
}

@available(macOS 13.0, iOS 16.0, *)
extension DiarizationModels {
    
    public enum ModelNames {
        public static let segmentation = "pyannote_segmentation.mlmodelc"
        public static let embedding = "wespeaker.mlmodelc"
    }
    
    /// Get the base FluidAudio directory
    private static func baseDirectory() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
            .first!
            .appendingPathComponent("FluidAudio")
    }
    
    /// Load diarization models, downloading them if necessary
    /// - Parameters:
    ///   - directory: Base directory for model storage (defaults to Application Support)
    ///   - configuration: MLModel configuration to use (defaults to optimized settings)
    /// - Returns: Loaded diarization models
    public static func load(
        from directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizationModels {
        let baseDir = directory ?? baseDirectory()
        let config = configuration ?? defaultConfiguration()
        
        // Download/load models using DownloadUtils (handles caching automatically)
        let models = try await DownloadUtils.loadModels(
            .diarizer,
            modelNames: [ModelNames.segmentation, ModelNames.embedding],
            directory: baseDir,
            computeUnits: config.computeUnits
        )
        
        // Create DiarizationModels from loaded models
        guard let segmentationModel = models[ModelNames.segmentation],
              let embeddingModel = models[ModelNames.embedding] else {
            throw DiarizationModelsError.loadingFailed("Failed to load all diarization models")
        }
        
        return DiarizationModels(
            segmentation: segmentationModel,
            embedding: embeddingModel,
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
extension DiarizationModels {
    
    /// Alias for load() - kept for backward compatibility
    public static func downloadAndLoad(
        to directory: URL? = nil,
        configuration: MLModelConfiguration? = nil
    ) async throws -> DiarizationModels {
        return try await load(from: directory, configuration: configuration)
    }
    
    public static func modelsExist(at directory: URL) -> Bool {
        // If directory is already the repo, use it; otherwise append repo name
        let repoPath = directory.lastPathComponent == "speaker-diarization-coreml"
            ? directory
            : directory.appendingPathComponent("speaker-diarization-coreml")
        
        guard FileManager.default.fileExists(atPath: repoPath.path) else {
            return false
        }
        
        // Check all required files exist
        let requiredFiles = [
            ModelNames.segmentation,
            ModelNames.embedding
        ]
        
        return requiredFiles.allSatisfy { fileName in
            FileManager.default.fileExists(
                atPath: repoPath.appendingPathComponent(fileName).path
            )
        }
    }
    
    public static func defaultCacheDirectory() -> URL {
        // Return the repo directory directly
        return baseDirectory().appendingPathComponent("speaker-diarization-coreml", isDirectory: true)
    }
}

public enum DiarizationModelsError: LocalizedError {
    case modelNotFound(String, URL)
    case downloadFailed(String)
    case loadingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .modelNotFound(let name, let path):
            return "Diarization model '\(name)' not found at: \(path.path)"
        case .downloadFailed(let reason):
            return "Failed to download diarization models: \(reason)"
        case .loadingFailed(let reason):
            return "Failed to load diarization models: \(reason)"
        }
    }
}