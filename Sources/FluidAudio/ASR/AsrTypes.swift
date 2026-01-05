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

/// Represents a word correction applied during transcription (for UI highlighting)
public struct WordCorrection: Codable, Sendable, Equatable {
    /// Character range in the corrected/final text string
    public let range: Range<Int>
    /// The original (misspelled) word from the raw transcription
    public let original: String
    /// The corrected (canonical) word from custom vocabulary
    public let corrected: String

    public init(range: Range<Int>, original: String, corrected: String) {
        self.range = range
        self.original = original
        self.corrected = corrected
    }
}

public struct ASRResult: Codable, Sendable {
    public let text: String
    public let confidence: Float
    public let duration: TimeInterval
    public let processingTime: TimeInterval
    public let tokenTimings: [TokenTiming]?
    public let performanceMetrics: ASRPerformanceMetrics?
    public let ctcDetectedTerms: [String]?
    public let ctcAppliedTerms: [String]?
    /// Word corrections applied via CTC keyword boosting (for UI highlighting)
    public let corrections: [WordCorrection]?

    public init(
        text: String, confidence: Float, duration: TimeInterval, processingTime: TimeInterval,
        tokenTimings: [TokenTiming]? = nil,
        performanceMetrics: ASRPerformanceMetrics? = nil,
        ctcDetectedTerms: [String]? = nil,
        ctcAppliedTerms: [String]? = nil,
        corrections: [WordCorrection]? = nil
    ) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingTime = processingTime
        self.tokenTimings = tokenTimings
        self.performanceMetrics = performanceMetrics
        self.ctcDetectedTerms = ctcDetectedTerms
        self.ctcAppliedTerms = ctcAppliedTerms
        self.corrections = corrections
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

// MARK: - Streaming State

/// Streaming state for VAD-style chunk-by-chunk ASR processing with LocalAgreement validation.
/// Similar to VadStreamState for low-latency streaming transcription.
public struct AsrStreamState: Sendable {
    /// Decoder LSTM state carried forward between chunks (for context continuity)
    var decoderState: TdtDecoderState

    /// Previous chunk's hypothesis for LocalAgreement-2 temporal comparison
    var previousHypothesis: TdtHypothesis?

    /// Total audio samples processed (for frame offset calculation)
    var processedSamples: Int

    /// Accumulated confirmed tokens (stable, validated by LocalAgreement)
    var confirmedTokens: [Int]

    /// Tokens awaiting validation (provisional, shown as volatile during streaming)
    var provisionalTokens: [Int]

    /// Initialize streaming state (like VadManager.makeStreamState())
    static func initial() -> AsrStreamState {
        AsrStreamState(
            decoderState: TdtDecoderState.make(),
            previousHypothesis: nil,
            processedSamples: 0,
            confirmedTokens: [],
            provisionalTokens: []
        )
    }

    init(
        decoderState: TdtDecoderState,
        previousHypothesis: TdtHypothesis?,
        processedSamples: Int,
        confirmedTokens: [Int],
        provisionalTokens: [Int]
    ) {
        self.decoderState = decoderState
        self.previousHypothesis = previousHypothesis
        self.processedSamples = processedSamples
        self.confirmedTokens = confirmedTokens
        self.provisionalTokens = provisionalTokens
    }
}

/// Result of processing a single streaming chunk with LocalAgreement validation.
/// Contains updated state and confirmed/provisional token split.
public struct AsrStreamResult: Sendable {
    /// Updated state for next chunk (carries decoder state, previous hypothesis)
    public let state: AsrStreamState

    /// Tokens validated by LocalAgreement comparison with previous chunk (green/confirmed)
    public let confirmed: [Int]

    /// New tokens from current chunk awaiting validation (purple/volatile)
    public let provisional: [Int]

    /// Timestamps for all tokens in this result
    public let allTimestamps: [Int]

    /// Confidence scores for all tokens in this result
    public let allConfidences: [Float]

    init(
        state: AsrStreamState,
        confirmed: [Int],
        provisional: [Int],
        allTimestamps: [Int],
        allConfidences: [Float]
    ) {
        self.state = state
        self.confirmed = confirmed
        self.provisional = provisional
        self.allTimestamps = allTimestamps
        self.allConfidences = allConfidences
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
            return "Unsupported platform: \(message)"
        }
    }
}
