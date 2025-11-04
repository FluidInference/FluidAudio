import Foundation

// MARK: - Configuration

public struct ASRConfig: Sendable {
    public let sampleRate: Int
    public let tdtConfig: TdtConfig
    public let localAgreementConfig: LocalAgreementConfig

    public static let `default` = ASRConfig()

    public init(
        sampleRate: Int = 16000,
        tdtConfig: TdtConfig = .default,
        localAgreementConfig: LocalAgreementConfig = .default
    ) {
        self.sampleRate = sampleRate
        self.tdtConfig = tdtConfig
        self.localAgreementConfig = localAgreementConfig
    }
}

/// Configuration for LocalAgreement-2 streaming validation.
public struct LocalAgreementConfig: Sendable {
    /// Minimum confidence for a token to be considered in agreement.
    /// Default 0.7 balances confirmation speed vs accuracy.
    public let confidenceThreshold: Float

    /// Maximum frames to hold as provisional before force-confirming.
    /// Default 60 frames (~0.5 seconds) prevents unbounded latency growth.
    public let maxProvisionalFrames: Int

    public static let `default` = LocalAgreementConfig()

    public init(
        confidenceThreshold: Float = 0.7,
        maxProvisionalFrames: Int = 60
    ) {
        self.confidenceThreshold = max(0.0, min(1.0, confidenceThreshold))
        self.maxProvisionalFrames = maxProvisionalFrames
    }
}

// MARK: - Results

public struct ASRResult: Sendable {
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

public struct TokenTiming: Sendable {
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
        }
    }
}
