import Foundation

// MARK: - Configuration

@available(macOS 13.0, iOS 16.0, *)
public struct RealtimeAsrConfig: Sendable {
    public let asrConfig: ASRConfig
    public let chunkDuration: TimeInterval  // Duration of each audio chunk in seconds
    public let bufferCapacity: Int          // Maximum buffer size in samples
    public let stabilizationDelay: Int      // Number of chunks before confirming text
    
    public static let `default` = RealtimeAsrConfig(
        asrConfig: ASRConfig(
            sampleRate: 16000,
            maxSymbolsPerFrame: 3,
            enableDebug: false,
            realtimeMode: true,
            chunkSizeMs: 2500,  // Increased to 2.5s for better accuracy
            tdtConfig: TdtConfig(
                durations: [0, 1, 2, 3, 4],
                includeTokenDuration: true,
                maxSymbolsPerStep: 3
            )
        ),
        chunkDuration: 2.5,  // Increased from 1.5s
        bufferCapacity: 160_000, // 10 seconds at 16kHz
        stabilizationDelay: 3
    )
    
    public static let lowLatency = RealtimeAsrConfig(
        asrConfig: ASRConfig(
            sampleRate: 16000,
            maxSymbolsPerFrame: 3,
            enableDebug: false,
            realtimeMode: true,
            chunkSizeMs: 2000,  // 2.0s for better accuracy while maintaining low latency
            tdtConfig: TdtConfig(
                durations: [0, 1, 2, 3],
                includeTokenDuration: true,
                maxSymbolsPerStep: 3
            )
        ),
        chunkDuration: 2.0,  // Balanced for accuracy and latency
        bufferCapacity: 96_000, // 6 seconds at 16kHz
        stabilizationDelay: 2
    )
    
    /// Ultra-low latency configuration with reduced accuracy
    /// WARNING: This configuration sacrifices accuracy for minimal latency
    public static let ultraLowLatency = RealtimeAsrConfig(
        asrConfig: ASRConfig(
            sampleRate: 16000,
            maxSymbolsPerFrame: 2,
            enableDebug: false,
            realtimeMode: true,
            chunkSizeMs: 1500,  // 1.5s chunks - expect reduced accuracy
            tdtConfig: TdtConfig(
                durations: [0, 1, 2],
                includeTokenDuration: true,
                maxSymbolsPerStep: 2
            )
        ),
        chunkDuration: 1.5,  // Minimum viable for basic transcription
        bufferCapacity: 80_000, // 5 seconds at 16kHz
        stabilizationDelay: 1  // Faster confirmation for lower latency
    )
    
    public init(
        asrConfig: ASRConfig,
        chunkDuration: TimeInterval,
        bufferCapacity: Int,
        stabilizationDelay: Int
    ) {
        self.asrConfig = asrConfig
        self.chunkDuration = chunkDuration
        self.bufferCapacity = bufferCapacity
        self.stabilizationDelay = stabilizationDelay
    }
    
    /// Number of samples in a chunk
    public var chunkSizeInSamples: Int {
        Int(chunkDuration * Double(asrConfig.sampleRate))
    }
}

// MARK: - Transcription Updates

@available(macOS 13.0, iOS 16.0, *)
public struct TranscriptionUpdate: Sendable {
    public let streamId: UUID
    public let timestamp: Date
    public let timeRange: ClosedRange<TimeInterval>
    public let text: String
    public let type: UpdateType
    public let confidence: Float
    public let processingTime: TimeInterval
    
    public enum UpdateType: Sendable {
        case pending    // Most recent chunk, may change
        case partial    // Stabilizing, likely accurate
        case confirmed  // Final, won't change
    }
    
    public init(
        streamId: UUID,
        timestamp: Date,
        timeRange: ClosedRange<TimeInterval>,
        text: String,
        type: UpdateType,
        confidence: Float,
        processingTime: TimeInterval
    ) {
        self.streamId = streamId
        self.timestamp = timestamp
        self.timeRange = timeRange
        self.text = text
        self.type = type
        self.confidence = confidence
        self.processingTime = processingTime
    }
}

// MARK: - Stream State

@available(macOS 13.0, iOS 16.0, *)
public struct StreamMetrics: Sendable {
    public var timeToFirstWord: TimeInterval?
    public var totalProcessingTime: TimeInterval = 0
    public var totalAudioDuration: TimeInterval = 0
    public var chunkCount: Int = 0
    
    public init(
        timeToFirstWord: TimeInterval? = nil,
        totalProcessingTime: TimeInterval = 0,
        totalAudioDuration: TimeInterval = 0,
        chunkCount: Int = 0
    ) {
        self.timeToFirstWord = timeToFirstWord
        self.totalProcessingTime = totalProcessingTime
        self.totalAudioDuration = totalAudioDuration
        self.chunkCount = chunkCount
    }
    
    public var rtfx: Float {
        guard totalProcessingTime > 0 else { return 0 }
        return Float(totalAudioDuration / totalProcessingTime)
    }
    
    public var averageLatency: TimeInterval {
        guard chunkCount > 0 else { return 0 }
        return totalProcessingTime / TimeInterval(chunkCount)
    }
}

// MARK: - Internal Types

@available(macOS 13.0, iOS 16.0, *)
struct TranscriptionSegment: Sendable {
    let id: UUID = UUID()
    let chunkIndex: Int
    let timeRange: ClosedRange<TimeInterval>
    let text: String
    let confidence: Float
    let timestamp: Date
    var updateType: TranscriptionUpdate.UpdateType
    
    init(
        chunkIndex: Int,
        timeRange: ClosedRange<TimeInterval>,
        text: String,
        confidence: Float,
        updateType: TranscriptionUpdate.UpdateType = .pending
    ) {
        self.chunkIndex = chunkIndex
        self.timeRange = timeRange
        self.text = text
        self.confidence = confidence
        self.timestamp = Date()
        self.updateType = updateType
    }
}

// MARK: - Errors

@available(macOS 13.0, iOS 16.0, *)
public enum RealtimeAsrError: LocalizedError, Sendable {
    case streamNotFound
    case bufferOverflow
    case invalidConfiguration
    case processingFailed(String)
    
    public var errorDescription: String? {
        switch self {
        case .streamNotFound:
            return "Stream not found. It may have been removed or never created."
        case .bufferOverflow:
            return "Audio buffer overflow. Processing cannot keep up with input rate."
        case .invalidConfiguration:
            return "Invalid realtime ASR configuration."
        case .processingFailed(let message):
            return "Realtime processing failed: \(message)"
        }
    }
}

// MARK: - Text Stabilization

@available(macOS 13.0, iOS 16.0, *)
struct StabilizationResult: Sendable {
    let confirmed: String
    let partial: String
    let pending: String
    let fullText: String
}