#if Diarizer
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

    /// When true, `performCompleteDiarization` populates
    /// `DiarizationResult.chunkEmbeddings` with one entry per (chunk, local speaker)
    /// that received a speaker ID. Off by default, matching
    /// `OfflineDiarizerConfig.exposeChunkEmbeddings`.
    public var exposeChunkEmbeddings: Bool = false

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
        chunkOverlap: Float = 0.0,
        exposeChunkEmbeddings: Bool = false
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
        self.exposeChunkEmbeddings = exposeChunkEmbeddings
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

/// Per-chunk speaker embedding produced during diarization, surfaced for
/// downstream consumers that need finer-grained data than `TimedSpeakerSegment`
/// (e.g. cluster-level contamination correction, or re-clustering the online
/// diarizer's labels over the whole file). One entry per (segmentation chunk,
/// local speaker slot) emitted by the embedding extraction step.
///
/// The `embedding256` field carries the speaker embedding as the extractor
/// emitted it. For `DiarizerManager` that is the raw WeSpeaker output — the same
/// vector its segments carry — which is not unit-norm. `rho128` carries the
/// PLDA-whitened representation when a PLDA model is loaded (un-normalized;
/// magnitude carries confidence). `rho128` is empty when no PLDA model is
/// available, which is always the case for `DiarizerManager`.
public struct ChunkEmbedding: Sendable, Codable {
    /// Cluster identifier matching `DiarizationResult.segments[*].speakerId`
    /// ("S1", "S2", ... for the offline pipeline; `SpeakerManager` IDs for
    /// `DiarizerManager`). Use this to align chunks back to their assigned cluster.
    public let speakerId: String
    public let chunkIndex: Int
    public let speakerIndex: Int
    public let startTimeSeconds: Double
    public let endTimeSeconds: Double
    public let embedding256: [Float]
    public let rho128: [Double]

    public init(
        speakerId: String,
        chunkIndex: Int,
        speakerIndex: Int,
        startTimeSeconds: Double,
        endTimeSeconds: Double,
        embedding256: [Float],
        rho128: [Double] = []
    ) {
        self.speakerId = speakerId
        self.chunkIndex = chunkIndex
        self.speakerIndex = speakerIndex
        self.startTimeSeconds = startTimeSeconds
        self.endTimeSeconds = endTimeSeconds
        self.embedding256 = embedding256
        self.rho128 = rho128
    }
}

public struct DiarizationResult: Sendable {
    public let segments: [TimedSpeakerSegment]

    /// Speaker database with embeddings (populated by offline pipelines for downstream use)
    public let speakerDatabase: [String: [Float]]?

    /// Per-chunk speaker embeddings with cluster assignments. Populated by the
    /// offline pipeline when `OfflineDiarizerConfig.exposeChunkEmbeddings` is
    /// enabled, and by `DiarizerManager.performCompleteDiarization` when
    /// `DiarizerConfig.exposeChunkEmbeddings` is enabled; nil otherwise.
    /// See `ChunkEmbedding` for field semantics.
    public let chunkEmbeddings: [ChunkEmbedding]?

    /// Performance timings collected during diarization
    public let timings: PipelineTimings?

    public init(
        segments: [TimedSpeakerSegment],
        speakerDatabase: [String: [Float]]? = nil,
        chunkEmbeddings: [ChunkEmbedding]? = nil,
        timings: PipelineTimings? = nil
    ) {
        self.segments = segments
        self.speakerDatabase = speakerDatabase
        self.chunkEmbeddings = chunkEmbeddings
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
#endif
