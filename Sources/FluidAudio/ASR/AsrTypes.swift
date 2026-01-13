import Foundation

// MARK: - Configuration

public struct ASRConfig: Sendable {
    public let sampleRate: Int
    public let tdtConfig: TdtConfig

    /// Enable streaming mode for large files to reduce memory usage.
    /// When enabled, files larger than `streamingThreshold` samples will be processed
    /// using streaming to maintain constant memory usage.
    public let streamingEnabled: Bool

    /// File size threshold in samples for enabling streaming.
    /// Files with more samples than this threshold will use streaming mode.
    /// Default: 480,000 samples (~30 seconds at 16kHz)
    public let streamingThreshold: Int

    public static let `default` = ASRConfig()

    public init(
        sampleRate: Int = 16000,
        tdtConfig: TdtConfig = .default,
        streamingEnabled: Bool = true,
        streamingThreshold: Int = 480_000
    ) {
        self.sampleRate = sampleRate
        self.tdtConfig = tdtConfig
        self.streamingEnabled = streamingEnabled
        self.streamingThreshold = streamingThreshold
    }
}

// MARK: - Results

public struct ASRResult: Codable, Sendable {
    public let text: String
    public let confidence: Float
    public let duration: TimeInterval
    public let processingTime: TimeInterval
    public let tokenTimings: [TokenTiming]?
    public let performanceMetrics: ASRPerformanceMetrics?

    public init(
        text: String, confidence: Float, duration: TimeInterval, processingTime: TimeInterval,
        tokenTimings: [TokenTiming]? = nil,
        performanceMetrics: ASRPerformanceMetrics? = nil
    ) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingTime = processingTime
        self.tokenTimings = tokenTimings
        self.performanceMetrics = performanceMetrics
    }

    /// Real-time factor (RTFx) - how many times faster than real-time
    public var rtfx: Float {
        Float(duration) / Float(processingTime)
    }
}

public struct TokenTiming: Codable, Sendable {
    public let token: String
    public let tokenId: Int
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let confidence: Float

    public init(
        token: String, tokenId: Int, startTime: TimeInterval, endTime: TimeInterval,
        confidence: Float
    ) {
        self.token = token
        self.tokenId = tokenId
        self.startTime = startTime
        self.endTime = endTime
        self.confidence = confidence
    }
}

// MARK: - Errors

public enum ASRError: Error, LocalizedError {
    case notInitialized
    case invalidAudioData
    case modelLoadFailed
    case processingFailed(String)
    case modelCompilationFailed
    case unsupportedPlatform(String)
    case streamingConversionFailed(Error)
    case fileAccessFailed(URL, Error)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "AsrManager not initialized. Call initialize() first."
        case .invalidAudioData:
            return "Invalid audio data provided. Must be at least 1 second of 16kHz audio."
        case .modelLoadFailed:
            return "Failed to load Parakeet CoreML models."
        case .processingFailed(let message):
            return "ASR processing failed: \(message)"
        case .modelCompilationFailed:
            return "CoreML model compilation failed after recovery attempts."
        case .unsupportedPlatform(let message):
            return message
        case .streamingConversionFailed(let error):
            return "Streaming audio conversion failed: \(error.localizedDescription)"
        case .fileAccessFailed(let url, let error):
            return "Failed to access audio file at \(url.path): \(error.localizedDescription)"
        }
    }
}
