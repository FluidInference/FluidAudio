import CoreMedia
import Foundation

// MARK: - Diarized ASR Configuration

/// Configuration for the unified streaming ASR + diarization pipeline
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingDiarizedAsrConfig: Sendable {
    /// ASR configuration for transcription
    public let asrConfig: StreamingAsrConfig
    /// Diarization configuration for speaker identification
    public let diarizerConfig: DiarizerConfig
    /// Enable debug logging for both systems
    public let enableDebug: Bool
    /// Alignment tolerance for matching ASR segments to speaker segments (seconds)
    public let alignmentTolerance: TimeInterval

    /// Default configuration optimized for real-time streaming
    public static let `default` = StreamingDiarizedAsrConfig(
        asrConfig: .realtime,
        diarizerConfig: .default,
        enableDebug: false,
        alignmentTolerance: 0.5
    )

    /// Configuration optimized for maximum accuracy (higher latency)
    public static let accurate = StreamingDiarizedAsrConfig(
        asrConfig: .default,
        diarizerConfig: DiarizerConfig(
            clusteringThreshold: 0.7,
            minSpeechDuration: 0.5,
            minEmbeddingUpdateDuration: 1.0,
            debugMode: false,
            chunkDuration: 10.0,
            chunkOverlap: 1.0
        ),
        enableDebug: false,
        alignmentTolerance: 0.3
    )

    public init(
        asrConfig: StreamingAsrConfig = .realtime,
        diarizerConfig: DiarizerConfig = .default,
        enableDebug: Bool = false,
        alignmentTolerance: TimeInterval = 0.5
    ) {
        self.asrConfig = asrConfig
        self.diarizerConfig = diarizerConfig
        self.enableDebug = enableDebug
        self.alignmentTolerance = alignmentTolerance
    }
}

// MARK: - Result Types

/// A transcription result with speaker attribution
@available(macOS 13.0, iOS 16.0, *)
public struct DiarizedTranscriptionResult: Sendable {
    /// Unique identifier for this segment
    public let segmentID: UUID
    /// Revision number for this segment (volatile updates increment this)
    public let revision: Int
    /// The speaker ID who spoke this text
    public let speakerId: String
    /// The transcribed text with formatting
    public let attributedText: AttributedString
    /// Audio time range this text covers
    public let audioTimeRange: CMTimeRange
    /// Whether this is a final result or volatile update
    public let isFinal: Bool
    /// Transcription confidence (0.0-1.0)
    public let transcriptionConfidence: Float
    /// Speaker identification confidence (0.0-1.0)
    public let speakerConfidence: Float
    /// When this result was generated
    public let timestamp: Date

    /// Overall confidence combining both transcription and speaker identification
    public var combinedConfidence: Float {
        (transcriptionConfidence + speakerConfidence) / 2.0
    }

    public init(
        segmentID: UUID,
        revision: Int,
        speakerId: String,
        attributedText: AttributedString,
        audioTimeRange: CMTimeRange,
        isFinal: Bool,
        transcriptionConfidence: Float,
        speakerConfidence: Float,
        timestamp: Date
    ) {
        self.segmentID = segmentID
        self.revision = revision
        self.speakerId = speakerId
        self.attributedText = attributedText
        self.audioTimeRange = audioTimeRange
        self.isFinal = isFinal
        self.transcriptionConfidence = transcriptionConfidence
        self.speakerConfidence = speakerConfidence
        self.timestamp = timestamp
    }
}

/// Complete snapshot of all speaker transcripts
@available(macOS 13.0, iOS 16.0, *)
public struct DiarizedTranscriptSnapshot: Sendable {
    /// Per-speaker finalized transcripts
    public let finalizedBySpeaker: [String: AttributedString]
    /// Per-speaker volatile transcripts (current updates)
    public let volatileBySpeaker: [String: AttributedString]
    /// Combined timeline transcript (all speakers in chronological order)
    public let combinedTranscript: AttributedString
    /// Active speakers in the current segment
    public let activeSpeakers: Set<String>
    /// When this snapshot was created
    public let lastUpdated: Date

    public init(
        finalizedBySpeaker: [String: AttributedString],
        volatileBySpeaker: [String: AttributedString],
        combinedTranscript: AttributedString,
        activeSpeakers: Set<String>,
        lastUpdated: Date
    ) {
        self.finalizedBySpeaker = finalizedBySpeaker
        self.volatileBySpeaker = volatileBySpeaker
        self.combinedTranscript = combinedTranscript
        self.activeSpeakers = activeSpeakers
        self.lastUpdated = lastUpdated
    }
}

/// Speaker statistics for the current session
@available(macOS 13.0, iOS 16.0, *)
public struct SpeakerSessionStats: Sendable {
    /// Speaker ID
    public let speakerId: String
    /// Total speaking time in seconds
    public let totalSpeakingTime: TimeInterval
    /// Number of speaking segments
    public let segmentCount: Int
    /// Average segment duration
    public let averageSegmentDuration: TimeInterval
    /// Average transcription confidence for this speaker
    public let averageConfidence: Float
    /// Most recent activity timestamp
    public let lastActivity: Date

    public init(
        speakerId: String,
        totalSpeakingTime: TimeInterval,
        segmentCount: Int,
        averageSegmentDuration: TimeInterval,
        averageConfidence: Float,
        lastActivity: Date
    ) {
        self.speakerId = speakerId
        self.totalSpeakingTime = totalSpeakingTime
        self.segmentCount = segmentCount
        self.averageSegmentDuration = averageSegmentDuration
        self.averageConfidence = averageConfidence
        self.lastActivity = lastActivity
    }
}

// MARK: - Internal Processing Types

/// Internal structure for matching ASR results with speaker segments
@available(macOS 13.0, iOS 16.0, *)
internal struct AsrSpeakerAlignment: Sendable {
    let asrResult: StreamingTranscriptionResult
    let speakerSegment: TimedSpeakerSegment?
    let alignmentConfidence: Float

    init(
        asrResult: StreamingTranscriptionResult,
        speakerSegment: TimedSpeakerSegment?,
        alignmentConfidence: Float
    ) {
        self.asrResult = asrResult
        self.speakerSegment = speakerSegment
        self.alignmentConfidence = alignmentConfidence
    }
}

/// Processing chunk that contains both ASR and diarization results
@available(macOS 13.0, iOS 16.0, *)
internal struct ProcessingChunk: Sendable {
    let chunkId: UUID
    let audioSamples: [Float]
    let startTimeSeconds: TimeInterval
    let asrResults: [StreamingTranscriptionResult]
    let speakerSegments: [TimedSpeakerSegment]
    let processedAt: Date

    init(
        chunkId: UUID,
        audioSamples: [Float],
        startTimeSeconds: TimeInterval,
        asrResults: [StreamingTranscriptionResult] = [],
        speakerSegments: [TimedSpeakerSegment] = [],
        processedAt: Date = Date()
    ) {
        self.chunkId = chunkId
        self.audioSamples = audioSamples
        self.startTimeSeconds = startTimeSeconds
        self.asrResults = asrResults
        self.speakerSegments = speakerSegments
        self.processedAt = processedAt
    }
}

// MARK: - Errors

/// Errors specific to the diarized ASR pipeline
@available(macOS 13.0, iOS 16.0, *)
public enum DiarizedAsrError: Error, LocalizedError {
    case asrNotInitialized
    case diarizerNotInitialized
    case alignmentFailed(String)
    case processingFailed(String)
    case configurationInvalid(String)

    public var errorDescription: String? {
        switch self {
        case .asrNotInitialized:
            return "ASR system not initialized. Call initialize() first."
        case .diarizerNotInitialized:
            return "Diarizer system not initialized. Call initialize() first."
        case .alignmentFailed(let message):
            return "Failed to align ASR results with speaker segments: \(message)"
        case .processingFailed(let message):
            return "Diarized ASR processing failed: \(message)"
        case .configurationInvalid(let message):
            return "Invalid configuration: \(message)"
        }
    }
}
