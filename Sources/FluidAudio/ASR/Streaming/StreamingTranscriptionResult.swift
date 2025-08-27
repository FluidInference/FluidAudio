import AVFoundation
import CoreMedia
import Foundation

/// A finalized transcript segment with timing information
@available(macOS 13.0, iOS 16.0, *)
public struct TimestampedSegment: Sendable {
    public let id: UUID
    public let text: String
    public let startTime: TimeInterval  // seconds from stream start
    public let endTime: TimeInterval  // seconds from stream start

    public init(id: UUID, text: String, startTime: TimeInterval, endTime: TimeInterval) {
        self.id = id
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
    }

    /// Duration of this segment in seconds
    public var duration: TimeInterval {
        return endTime - startTime
    }

    /// Formatted timestamp string (HH:MM:SS.mmm format)
    public var formattedTimeRange: String {
        let startFormatted = formatTime(startTime)
        let endFormatted = formatTime(endTime)
        return "\(startFormatted) --> \(endFormatted)"
    }

    private func formatTime(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = Int(seconds) % 3600 / 60
        let secs = Int(seconds) % 60
        let milliseconds = Int((seconds.truncatingRemainder(dividingBy: 1)) * 1000)
        return String(format: "%02d:%02d:%02d.%03d", hours, minutes, secs, milliseconds)
    }
}

/// Enhanced snapshot with timing information for finalized segments
@available(macOS 13.0, iOS 16.0, *)
public struct StreamingTranscriptSnapshot: Sendable {
    public let finalized: AttributedString
    public let volatile: AttributedString?
    public let lastUpdated: Date
    public let timestampedSegments: [TimestampedSegment]  // NEW: segments with timing

    public init(
        finalized: AttributedString, volatile: AttributedString?, lastUpdated: Date,
        timestampedSegments: [TimestampedSegment] = []
    ) {
        self.finalized = finalized
        self.volatile = volatile
        self.lastUpdated = lastUpdated
        self.timestampedSegments = timestampedSegments
    }

    /// Get the full transcript with embedded timestamps
    public var timestampedText: String {
        guard !timestampedSegments.isEmpty else {
            return finalized.description
        }

        return timestampedSegments.map { segment in
            "[\(segment.formattedTimeRange)] \(segment.text)"
        }.joined(separator: "\n")
    }

    /// Export in SRT subtitle format
    public var srtFormat: String {
        guard !timestampedSegments.isEmpty else { return "" }

        return timestampedSegments.enumerated().map { index, segment in
            let srtStart = formatSRTTime(segment.startTime)
            let srtEnd = formatSRTTime(segment.endTime)
            return "\(index + 1)\n\(srtStart) --> \(srtEnd)\n\(segment.text)\n"
        }.joined(separator: "\n")
    }

    private func formatSRTTime(_ seconds: TimeInterval) -> String {
        let hours = Int(seconds) / 3600
        let minutes = Int(seconds) % 3600 / 60
        let secs = Int(seconds) % 60
        let milliseconds = Int((seconds.truncatingRemainder(dividingBy: 1)) * 1000)
        return String(format: "%02d:%02d:%02d,%03d", hours, minutes, secs, milliseconds)
    }
}
