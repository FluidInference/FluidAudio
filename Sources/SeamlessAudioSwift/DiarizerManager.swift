//
//  DiarizerManager.swift
//  SeamlessAudioSwift
//
//  Created by Kiko Wei on 2025-06-24.
//

import Foundation
import OSLog

// MARK: - Backend Configuration

/// Supported diarization backends
public enum DiarizerBackend: String, CaseIterable, Sendable {
    case sherpaOnnx = "sherpa-onnx"
    case coreML = "coreml"
}

/// Configuration for speaker diarization
public struct DiarizerConfig: Sendable {
    public var backend: DiarizerBackend = .sherpaOnnx
    public var clusteringThreshold: Float = 0.7
    public var minDurationOn: Float = 1.0
    public var minDurationOff: Float = 0.5
    public var numClusters: Int = -1  // -1 = auto
    public var debugMode: Bool = false
    public var modelCacheDirectory: URL?

    public static let `default` = DiarizerConfig()

    public init(
        backend: DiarizerBackend = .sherpaOnnx,
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        debugMode: Bool = false,
        modelCacheDirectory: URL? = nil
    ) {
        self.backend = backend
        self.clusteringThreshold = clusteringThreshold
        self.minDurationOn = minDurationOn
        self.minDurationOff = minDurationOff
        self.numClusters = numClusters
        self.debugMode = debugMode
        self.modelCacheDirectory = modelCacheDirectory
    }
}

// MARK: - Public Data Types

/// Represents a speaker segment with timing and speaker information
public struct SpeakerSegment: Sendable, Identifiable {
    public let id = UUID()
    public let speakerClusterId: Int
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let confidenceScore: Float

    public var durationSeconds: Float {
        endTimeSeconds - startTimeSeconds
    }

    public init(speakerClusterId: Int, startTimeSeconds: Float, endTimeSeconds: Float, confidenceScore: Float = 1.0) {
        self.speakerClusterId = speakerClusterId
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.confidenceScore = confidenceScore
    }
}

/// Speaker embedding with quality metrics
public struct SpeakerEmbedding: Sendable {
    public let embedding: [Float]
    public let qualityScore: Float
    public let durationSeconds: Float

    public init(embedding: [Float], qualityScore: Float, durationSeconds: Float) {
        self.embedding = embedding
        self.qualityScore = qualityScore
        self.durationSeconds = durationSeconds
    }
}

/// Model file paths for diarization
public struct ModelPaths: Sendable {
    public let segmentationPath: String
    public let embeddingPath: String

    public init(segmentationPath: String, embeddingPath: String) {
        self.segmentationPath = segmentationPath
        self.embeddingPath = embeddingPath
    }
}

/// Audio validation result
public struct AudioValidationResult: Sendable {
    public let isValid: Bool
    public let durationSeconds: Float
    public let issues: [String]

    public init(isValid: Bool, durationSeconds: Float, issues: [String] = []) {
        self.isValid = isValid
        self.durationSeconds = durationSeconds
        self.issues = issues
    }
}

// MARK: - Protocol for Diarization Managers

/// Protocol that all diarization managers must implement
@available(macOS 13.0, iOS 16.0, *)
public protocol DiarizerManager: Sendable {
    /// The backend type this manager uses
    var backend: DiarizerBackend { get }

    /// Check if the diarization system is ready for use
    var isAvailable: Bool { get }

    /// Initialize the diarization system
    func initialize() async throws

    /// Perform speaker segmentation on audio samples
    func performSegmentation(_ samples: [Float], sampleRate: Int) async throws -> [SpeakerSegment]

    /// Extract speaker embedding from audio samples
    func extractEmbedding(from samples: [Float]) async throws -> SpeakerEmbedding?

    /// Compare similarity between two audio samples
    func compareSpeakers(audio1: [Float], audio2: [Float]) async throws -> Float

    /// Validate if an embedding is valid
    func validateEmbedding(_ embedding: [Float]) -> Bool

    /// Validate audio quality and characteristics
    func validateAudio(_ samples: [Float]) -> AudioValidationResult

    /// Calculate cosine distance between two embeddings
    func cosineDistance(_ a: [Float], _ b: [Float]) -> Float

    /// Clean up resources
    func cleanup() async
}

// MARK: - Factory for Creating Diarization Managers

/// Factory class for creating diarization managers
@available(macOS 13.0, iOS 16.0, *)
public final class DiarizerFactory {
    /// Create a diarization manager based on the configuration
    public static func createManager(config: DiarizerConfig = .default) -> any DiarizerManager {
        switch config.backend {
        case .sherpaOnnx:
            return SherpaOnnxDiarizerManager(config: config)
        case .coreML:
            return CoreMLDiarizerManager(config: config)
        }
    }
}

// MARK: - Error Types

public enum DiarizerError: Error, LocalizedError {
    case notInitialized
    case modelDownloadFailed
    case embeddingExtractionFailed
    case invalidAudioData
    case processingFailed(String)
    case unsupportedBackend(DiarizerBackend)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization system not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download required models."
        case .embeddingExtractionFailed:
            return "Failed to extract speaker embedding from audio."
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .unsupportedBackend(let backend):
            return "Unsupported backend: \(backend.rawValue)"
        }
    }
}

// MARK: - Convenience Extensions

@available(macOS 13.0, iOS 16.0, *)
extension DiarizerFactory {
    /// Create a SherpaOnnx diarization manager with custom configuration
    public static func createSherpaOnnxManager(
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        debugMode: Bool = false,
        modelCacheDirectory: URL? = nil
    ) -> SherpaOnnxDiarizerManager {
        let config = DiarizerConfig(
            backend: .sherpaOnnx,
            clusteringThreshold: clusteringThreshold,
            minDurationOn: minDurationOn,
            minDurationOff: minDurationOff,
            numClusters: numClusters,
            debugMode: debugMode,
            modelCacheDirectory: modelCacheDirectory
        )
        return SherpaOnnxDiarizerManager(config: config)
    }

    /// Create a CoreML diarization manager with custom configuration
    public static func createCoreMLManager(
        clusteringThreshold: Float = 0.7,
        minDurationOn: Float = 1.0,
        minDurationOff: Float = 0.5,
        numClusters: Int = -1,
        debugMode: Bool = false,
        modelCacheDirectory: URL? = nil
    ) -> CoreMLDiarizerManager {
        let config = DiarizerConfig(
            backend: .coreML,
            clusteringThreshold: clusteringThreshold,
            minDurationOn: minDurationOn,
            minDurationOff: minDurationOff,
            numClusters: numClusters,
            debugMode: debugMode,
            modelCacheDirectory: modelCacheDirectory
        )
        return CoreMLDiarizerManager(config: config)
    }
}
