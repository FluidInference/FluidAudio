import AVFoundation
import Foundation

/// Segment-based transcription result (volatile iterations and final for a range)
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptionResult: Sendable {
    public let segmentID: UUID
    public let revision: Int
    public let attributedText: AttributedString
    public let audioTimeRange: CMTimeRange
    public let isFinal: Bool
    public let confidence: Float
    public let timestamp: Date

    public init(
        segmentID: UUID,
        revision: Int,
        attributedText: AttributedString,
        audioTimeRange: CMTimeRange,
        isFinal: Bool,
        confidence: Float,
        timestamp: Date
    ) {
        self.segmentID = segmentID
        self.revision = revision
        self.attributedText = attributedText
        self.audioTimeRange = audioTimeRange
        self.isFinal = isFinal
        self.confidence = confidence
        self.timestamp = timestamp
    }
}

/// Snapshot of the full transcript state suitable for simple UIs
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptSnapshot: Sendable {
    public let finalized: AttributedString
    public let volatile: AttributedString?
    public let lastUpdated: Date

    public init(finalized: AttributedString, volatile: AttributedString?, lastUpdated: Date) {
        self.finalized = finalized
        self.volatile = volatile
        self.lastUpdated = lastUpdated
    }
}
