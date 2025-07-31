#if os(macOS)
import Foundation

/// Simple types for CLI display purposes only
/// These are not part of the main FluidAudio API

public struct TranscriptionUpdate {
    public let streamId: UUID
    public let text: String
    public let type: UpdateType
    public let timestamp: Date
    public let timeRange: ClosedRange<TimeInterval>
    
    public enum UpdateType {
        case pending
        case partial
        case confirmed
        case finalized
    }
    
    public init(
        streamId: UUID,
        text: String,
        type: UpdateType,
        timestamp: Date,
        timeRange: ClosedRange<TimeInterval> = 0...0
    ) {
        self.streamId = streamId
        self.text = text
        self.type = type
        self.timestamp = timestamp
        self.timeRange = timeRange
    }
}

public struct StreamMetrics {
    public let streamId: UUID
    public let totalChunks: Int
    public let chunkCount: Int
    public let averageProcessingTime: TimeInterval
    public let averageConfidence: Float
    public let rtfx: Double
    public let averageLatency: TimeInterval
    public let timeToFirstWord: TimeInterval?
    public let totalAudioDuration: TimeInterval
    public let totalProcessingTime: TimeInterval
    
    public init(
        streamId: UUID = UUID(),
        totalChunks: Int = 0,
        chunkCount: Int = 0,
        averageProcessingTime: TimeInterval = 0,
        averageConfidence: Float = 0,
        rtfx: Double = 0,
        averageLatency: TimeInterval = 0,
        timeToFirstWord: TimeInterval? = nil,
        totalAudioDuration: TimeInterval = 0,
        totalProcessingTime: TimeInterval = 0
    ) {
        self.streamId = streamId
        self.totalChunks = totalChunks
        self.chunkCount = chunkCount
        self.averageProcessingTime = averageProcessingTime
        self.averageConfidence = averageConfidence
        self.rtfx = rtfx
        self.averageLatency = averageLatency
        self.timeToFirstWord = timeToFirstWord
        self.totalAudioDuration = totalAudioDuration
        self.totalProcessingTime = totalProcessingTime
    }
}
#endif