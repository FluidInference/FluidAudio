import Foundation

// MARK: - Configuration

public struct ASRConfig: Sendable {
    public let sampleRate: Int
    public let tdtConfig: TdtConfig

    /// Encoder hidden dimension (1024 for 0.6B, 512 for 110m)
    public let encoderHiddenSize: Int

    /// Number of long-form chunks to transcribe concurrently.
    /// Applies only to stateless chunked transcription paths.
    public let parallelChunkConcurrency: Int

    /// Enable streaming mode for large files to reduce memory usage.
    /// When enabled, files larger than `streamingThreshold` samples will be processed
    /// using streaming to maintain constant memory usage.
    public let streamingEnabled: Bool

    /// File size threshold in samples for enabling streaming.
    /// Files with more samples than this threshold will use streaming mode.
    /// Default: 480,000 samples (~30 seconds at 16kHz)
    public let streamingThreshold: Int

    /// Enable the 80ms (1 encoder frame) mel-context prepend on non-first
    /// chunks in the long-form batch path. Added in PR #264 to fix
    /// all-blank predictions at chunk boundaries on long English audio.
    ///
    /// Issue #594 root cause: on `parakeet-tdt-0.6b-v3-coreml` with
    /// non-English audio, the 80ms prepend shifts the FastConformer
    /// encoder's first-frame distribution just enough that the SOS-primed
    /// TDT decoder drifts back to its English-biased prior. Disabling this
    /// flag (`false`) restores clean French/multilingual transcription at
    /// chunk boundaries while keeping parallel chunk processing.
    ///
    /// Default `true` preserves PR #264's blank-prediction fix on English.
    /// Set to `false` for non-English long-form batch transcription.
    public let melChunkContext: Bool

    public static let `default` = ASRConfig()

    public init(
        sampleRate: Int = 16000,
        tdtConfig: TdtConfig = .default,
        encoderHiddenSize: Int = ASRConstants.encoderHiddenSize,
        parallelChunkConcurrency: Int = 4,
        streamingEnabled: Bool = true,
        streamingThreshold: Int = 480_000,
        melChunkContext: Bool = true
    ) {
        self.sampleRate = sampleRate
        self.tdtConfig = tdtConfig
        self.encoderHiddenSize = encoderHiddenSize
        self.parallelChunkConcurrency = max(1, parallelChunkConcurrency)
        self.streamingEnabled = streamingEnabled
        self.streamingThreshold = streamingThreshold
        self.melChunkContext = melChunkContext
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
    public let ctcDetectedTerms: [String]?
    public let ctcAppliedTerms: [String]?

    public init(
        text: String, confidence: Float, duration: TimeInterval, processingTime: TimeInterval,
        tokenTimings: [TokenTiming]? = nil,
        performanceMetrics: ASRPerformanceMetrics? = nil,
        ctcDetectedTerms: [String]? = nil,
        ctcAppliedTerms: [String]? = nil
    ) {
        self.text = text
        self.confidence = confidence
        self.duration = duration
        self.processingTime = processingTime
        self.tokenTimings = tokenTimings
        self.performanceMetrics = performanceMetrics
        self.ctcDetectedTerms = ctcDetectedTerms
        self.ctcAppliedTerms = ctcAppliedTerms
    }

    /// Real-time factor (RTFx) - how many times faster than real-time
    public var rtfx: Float {
        Float(duration) / Float(processingTime)
    }

    /// Create a copy of this result with rescored text and CTC metadata from vocabulary boosting.
    ///
    /// - Parameters:
    ///   - text: The rescored transcript text
    ///   - detected: Vocabulary terms detected by CTC (candidates considered for replacement)
    ///   - applied: Vocabulary terms actually applied as replacements
    /// - Returns: A new ASRResult with updated text and CTC metadata
    public func withRescoring(text: String, detected: [String]?, applied: [String]?) -> ASRResult {
        ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: tokenTimings,
            performanceMetrics: performanceMetrics,
            ctcDetectedTerms: detected,
            ctcAppliedTerms: applied
        )
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
            return "Invalid audio data provided. Must be at least 300ms of 16kHz audio."
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
