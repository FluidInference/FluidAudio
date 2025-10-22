import Foundation

// MARK: - Diarization Pipeline Types
// This file contains types used in the diarization processing pipeline
// For speaker profile types, see SpeakerTypes.swift

public struct DiarizerConfig: Sendable {
    /// Threshold for clustering speaker embeddings (0.5-0.9). Lower = more speakers.
    public var clusteringThreshold: Float = 0.7

    /// Minimum speech segment duration in seconds. Shorter segments are discarded.
    public var minSpeechDuration: Float = 1.0

    /// Minimum segment duration for updating speaker embeddings (seconds).
    public var minEmbeddingUpdateDuration: Float = 2.0

    /// Minimum silence gap (seconds) before splitting same speaker's segments.
    public var minSilenceGap: Float = 0.5

    /// Expected number of speakers (-1 for automatic).
    public var numClusters: Int = -1

    /// Minimum active frames for valid speech detection.
    public var minActiveFramesCount: Float = 10.0

    /// Enable debug logging.
    public var debugMode: Bool = false

    /// Duration of audio chunks for processing (seconds).
    public var chunkDuration: Float = 10.0

    /// Overlap between chunks (seconds).
    public var chunkOverlap: Float = 0.0

    public static let `default` = DiarizerConfig()

    public init(
        clusteringThreshold: Float = 0.7,
        minSpeechDuration: Float = 1.0,
        minEmbeddingUpdateDuration: Float = 2.0,
        minSilenceGap: Float = 0.5,
        numClusters: Int = -1,
        minActiveFramesCount: Float = 10.0,
        debugMode: Bool = false,
        chunkDuration: Float = 10.0,
        chunkOverlap: Float = 0.0
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.minSpeechDuration = minSpeechDuration
        self.minEmbeddingUpdateDuration = minEmbeddingUpdateDuration
        self.minSilenceGap = minSilenceGap
        self.numClusters = numClusters
        self.minActiveFramesCount = minActiveFramesCount
        self.debugMode = debugMode
        self.chunkDuration = chunkDuration
        self.chunkOverlap = chunkOverlap
    }
}

public struct PipelineTimings: Sendable, Codable {
    public let modelCompilationSeconds: TimeInterval
    public let audioLoadingSeconds: TimeInterval
    public let segmentationSeconds: TimeInterval
    public let embeddingExtractionSeconds: TimeInterval
    public let speakerClusteringSeconds: TimeInterval
    public let postProcessingSeconds: TimeInterval
    public let totalInferenceSeconds: TimeInterval
    public let totalProcessingSeconds: TimeInterval

    public init(
        modelCompilationSeconds: TimeInterval = 0,
        audioLoadingSeconds: TimeInterval = 0,
        segmentationSeconds: TimeInterval = 0,
        embeddingExtractionSeconds: TimeInterval = 0,
        speakerClusteringSeconds: TimeInterval = 0,
        postProcessingSeconds: TimeInterval = 0
    ) {
        self.modelCompilationSeconds = modelCompilationSeconds
        self.audioLoadingSeconds = audioLoadingSeconds
        self.segmentationSeconds = segmentationSeconds
        self.embeddingExtractionSeconds = embeddingExtractionSeconds
        self.speakerClusteringSeconds = speakerClusteringSeconds
        self.postProcessingSeconds = postProcessingSeconds
        self.totalInferenceSeconds =
            segmentationSeconds + embeddingExtractionSeconds + speakerClusteringSeconds
        self.totalProcessingSeconds =
            modelCompilationSeconds + audioLoadingSeconds
            + segmentationSeconds + embeddingExtractionSeconds + speakerClusteringSeconds
            + postProcessingSeconds
    }

    public var stagePercentages: [String: Double] {
        guard totalProcessingSeconds > 0 else {
            return [:]
        }

        return [
            "Model Compilation": (modelCompilationSeconds / totalProcessingSeconds) * 100,
            "Audio Loading": (audioLoadingSeconds / totalProcessingSeconds) * 100,
            "Segmentation": (segmentationSeconds / totalProcessingSeconds) * 100,
            "Embedding Extraction": (embeddingExtractionSeconds / totalProcessingSeconds) * 100,
            "Speaker Clustering": (speakerClusteringSeconds / totalProcessingSeconds) * 100,
            "Post Processing": (postProcessingSeconds / totalProcessingSeconds) * 100,
        ]
    }

    public var bottleneckStage: String {
        let stages = [
            ("Model Compilation", modelCompilationSeconds),
            ("Audio Loading", audioLoadingSeconds),
            ("Segmentation", segmentationSeconds),
            ("Embedding Extraction", embeddingExtractionSeconds),
            ("Speaker Clustering", speakerClusteringSeconds),
            ("Post Processing", postProcessingSeconds),
        ]

        return stages.max(by: { $0.1 < $1.1 })?.0 ?? "Unknown"
    }
}

public struct DiarizationResult: Sendable {
    public let segments: [TimedSpeakerSegment]

    /// Speaker database with embeddings (populated by offline pipelines for downstream use)
    public let speakerDatabase: [String: [Float]]?

    /// Performance timings collected during diarization
    public let timings: PipelineTimings?

    public init(
        segments: [TimedSpeakerSegment],
        speakerDatabase: [String: [Float]]? = nil,
        timings: PipelineTimings? = nil
    ) {
        self.segments = segments
        self.speakerDatabase = speakerDatabase
        self.timings = timings
    }
}

/// Represents a segment of speech from a specific speaker with timing information
/// This is used for diarization results - "who spoke when"
/// For speaker profiles, see Speaker struct in SpeakerTypes.swift
public struct TimedSpeakerSegment: Sendable, Identifiable {
    public let id = UUID()
    public let speakerId: String
    public let embedding: [Float]
    public let startTimeSeconds: Float
    public let endTimeSeconds: Float
    public let qualityScore: Float

    public var durationSeconds: Float {
        endTimeSeconds - startTimeSeconds
    }

    public init(
        speakerId: String, embedding: [Float], startTimeSeconds: Float, endTimeSeconds: Float,
        qualityScore: Float
    ) {
        self.speakerId = speakerId
        self.embedding = embedding
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.qualityScore = qualityScore
    }
}

public struct ModelPaths: Sendable {
    public let segmentationPath: URL
    public let embeddingPath: URL

    public init(segmentationPath: URL, embeddingPath: URL) {
        self.segmentationPath = segmentationPath
        self.embeddingPath = embeddingPath
    }
}

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

public enum DiarizerError: Error, LocalizedError {
    case notInitialized
    case modelDownloadFailed
    case modelCompilationFailed
    case embeddingExtractionFailed
    case invalidAudioData
    case processingFailed(String)
    case memoryAllocationFailed
    case invalidArrayBounds

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization system not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download required models."
        case .modelCompilationFailed:
            return "Failed to compile CoreML models."
        case .embeddingExtractionFailed:
            return "Failed to extract speaker embedding from audio."
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .memoryAllocationFailed:
            return "Failed to allocate ANE-aligned memory."
        case .invalidArrayBounds:
            return "Array bounds exceeded for zero-copy view."
        }
    }
}
