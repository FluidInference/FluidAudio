import Foundation

// MARK: - Configuration

/// Configuration for Sortformer streaming diarization.
///
/// Based on NVIDIA's Streaming Sortformer 4-speaker model.
/// Reference: NeMo sortformer_modules.py
public struct SortformerConfig: Sendable {
    public typealias ModelVariant = ModelNames.Sortformer.Variant

    // MARK: - Model Architecture

    public let modelVariant: ModelVariant?

    /// Number of speaker slots (fixed at 4 for current model)
    public let numSpeakers: Int = 4

    /// Pre-encoder embedding dimension
    public let preEncoderDims: Int = 512

    /// Subsampling factor (8:1 downsampling in encoder)
    public let subsamplingFactor: Int = 8

    // MARK: - Streaming Parameters

    /// Output diarization frames per chunk
    /// Must match the value used in CoreML conversion
    public var chunkLen: Int = 6

    /// Left context frames for chunk processing
    public var chunkLeftContext: Int = 1

    /// Right context frames for chunk processing
    public var chunkRightContext: Int = 7

    /// Maximum FIFO queue length (recent embeddings)
    /// Must match CoreML conversion: fifo_len=40
    public var fifoLen: Int = 40

    /// Maximum speaker cache length (historical embeddings)
    /// Must match CoreML conversion: spkcache_len=188
    public var spkcacheLen: Int = 188

    /// Period for speaker cache updates (frames)
    public var spkcacheUpdatePeriod: Int = 31

    /// Silence frames per speaker in compressed cache
    public var spkcacheSilFramesPerSpk: Int = 3

    // MARK: - Debug

    /// Enable debug logging
    public var debugMode: Bool = false

    // MARK: - Audio Parameters

    /// Sample rate in Hz
    public let sampleRate: Int = 16000

    /// Mel spectrogram window size in samples (25ms)
    public let melWindow: Int = 400

    /// Mel spectrogram stride in samples (10ms)
    public let melStride: Int = 160

    /// Number of mel filterbank features
    public let melFeatures: Int = 128

    // MARK: - Thresholds

    /// Threshold for silence detection (sum of speaker probs)
    public var silenceThreshold: Float = 0.2

    /// Threshold for speech prediction
    public var predScoreThreshold: Float = 0.25

    /// Boost factor for latest frames in cache compression
    public var scoresBoostLatest: Float = 0.05

    /// Strong boost rate for top-k selection
    public var strongBoostRate: Float = 0.75

    /// Weak boost rate for preventing speaker dominance
    public var weakBoostRate: Float = 1.5

    /// Minimum positive scores rate
    public var minPosScoresRate: Float = 0.5

    /// Maximum index placeholder for disabled frames in spkcache compression
    public let maxIndex: Int = 99999

    // MARK: - Computed Properties

    /// Total chunk frames for CoreML model input (includes left/right context)
    /// Formula: (chunk_len + left_context + right_context) * subsampling
    /// Default: (6 + 1 + 1) * 8 = 64 frames
    public var chunkMelFrames: Int {
        (chunkLen + chunkLeftContext + chunkRightContext) * subsamplingFactor
    }

    /// Core frames per chunk (without context)
    public var coreFrames: Int {
        chunkLen * subsamplingFactor
    }

    /// Frame duration in seconds
    public var frameDurationSeconds: Float {
        Float(subsamplingFactor) * Float(melStride) / Float(sampleRate)
    }

    // MARK: - Initialization

    /// Configuration matching Gradient Descent's Streaming-Sortformer-Conversion models
    public static let `default` = SortformerConfig(
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Fast config with Sortformer v2 weights (~1.04s latency, smallest context).
    /// May handle high-speaker-count scenarios better than v2.1 (v2.1 can degrade when many speakers overlap).
    public static let `fastV2` = SortformerConfig(
        modelVariant: .fastV2,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Fast config with Sortformer v2.1 weights (~1.04s latency, smallest context).
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let `fastV2_1` = SortformerConfig(
        modelVariant: .fastV2_1,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 31
    )

    /// Balanced config with Sortformer v2 weights (~1.04s latency, larger FIFO for better quality).
    /// 20.57% DER on AMI SDM. May handle high-speaker-count scenarios better than v2.1.
    public static let balancedV2 = SortformerConfig(
        modelVariant: .balancedV2,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 188,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 144
    )

    /// Balanced config with Sortformer v2.1 weights (~1.04s latency, larger FIFO for better quality).
    /// 20.57% DER on AMI SDM.
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let balancedV2_1 = SortformerConfig(
        modelVariant: .balancedV2_1,
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 188,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 144
    )

    /// High-context config with Sortformer v2 weights (~30.4s latency, most context window).
    /// May handle high-speaker-count scenarios better than v2.1.
    public static let highContextV2 = SortformerConfig(
        modelVariant: .highContextV2,
        chunkLen: 340,
        chunkLeftContext: 1,
        chunkRightContext: 40,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 300
    )

    /// High-context config with Sortformer v2.1 weights (~30.4s latency, most context window).
    /// - Note: v2.1 may degrade when many speakers are talking simultaneously.
    public static let highContextV2_1 = SortformerConfig(
        modelVariant: .highContextV2_1,
        chunkLen: 340,
        chunkLeftContext: 1,
        chunkRightContext: 40,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 300
    )

    /// - Warning: If you don't use one of the default configurations, you must use a local model converted with that configuration.
    public init(
        modelVariant: ModelVariant? = .fastV2_1,
        chunkLen: Int = 6,
        chunkLeftContext: Int = 1,
        chunkRightContext: Int = 7,
        fifoLen: Int = 40,
        spkcacheLen: Int = 188,
        spkcacheUpdatePeriod: Int = 31,
        silenceThreshold: Float = 0.2,
        spkcacheSilFramesPerSpk: Int = 3,
        predScoreThreshold: Float = 0.25,
        scoresBoostLatest: Float = 0.05,
        strongBoostRate: Float = 0.75,
        weakBoostRate: Float = 1.5,
        minPosScoresRate: Float = 0.5,
        debugMode: Bool = false
    ) {
        self.modelVariant = modelVariant
        self.chunkLen = max(1, chunkLen)
        self.chunkLeftContext = chunkLeftContext
        self.chunkRightContext = chunkRightContext
        self.fifoLen = fifoLen
        self.silenceThreshold = silenceThreshold
        self.spkcacheSilFramesPerSpk = spkcacheSilFramesPerSpk
        self.debugMode = debugMode
        self.predScoreThreshold = predScoreThreshold
        self.scoresBoostLatest = scoresBoostLatest
        self.strongBoostRate = strongBoostRate
        self.weakBoostRate = weakBoostRate
        self.minPosScoresRate = minPosScoresRate

        // The following parameters must meet certain constraints
        self.spkcacheLen = max(spkcacheLen, (1 + self.spkcacheSilFramesPerSpk) * self.numSpeakers)
        self.spkcacheUpdatePeriod = max(min(spkcacheUpdatePeriod, self.fifoLen + self.chunkLen), self.chunkLen)
    }

    public func isCompatible(with other: SortformerConfig) -> Bool {
        return
            (self.chunkMelFrames == other.chunkMelFrames && self.fifoLen == other.fifoLen
            && self.spkcacheLen == other.spkcacheLen)
    }
}

// MARK: - Streaming State

/// State maintained across streaming chunks for Sortformer diarization.
///
/// This mirrors NeMo's StreamingSortformerState dataclass.
/// Reference: NeMo sortformer_modules.py
public struct SortformerStreamingState: Sendable {
    /// Speaker cache embeddings from start of audio
    /// Shape: [spkcacheLen, fcDModel] (e.g., [188, 512])
    public var spkcache: [Float]

    /// Current valid length of speaker cache
    public var spkcacheLength: Int

    /// Speaker predictions for cached embeddings
    /// Shape: [spkcacheLen, numSpeakers] (e.g., [188, 4])
    public var spkcachePreds: [Float]?

    /// FIFO queue of recent chunk embeddings
    /// Shape: [fifoLen, fcDModel] (e.g., [188, 512])
    public var fifo: [Float]

    /// Current valid length of FIFO queue
    public var fifoLength: Int

    /// Speaker predictions for FIFO embeddings
    /// Shape: [fifoLen, numSpeakers] (e.g., [188, 4])
    public var fifoPreds: [Float]?

    /// Running mean of silence embeddings
    /// Shape: [fcDModel] (e.g., [512])
    public var meanSilenceEmbedding: [Float]

    /// Count of silence frames observed
    public var silenceFrameCount: Int

    /// Initialize empty streaming state
    public init(config: SortformerConfig) {
        self.spkcache = []
        self.spkcachePreds = nil
        self.spkcacheLength = 0

        self.fifo = []
        self.fifoPreds = nil
        self.fifoLength = 0

        self.fifo.reserveCapacity((config.fifoLen + config.chunkLen) * config.preEncoderDims)
        self.spkcache.reserveCapacity((config.spkcacheLen + config.spkcacheUpdatePeriod) * config.preEncoderDims)

        self.meanSilenceEmbedding = [Float](repeating: 0.0, count: config.preEncoderDims)
        self.silenceFrameCount = 0
    }

    public mutating func cleanup() {
        self.fifo.removeAll(keepingCapacity: false)
        self.spkcache.removeAll(keepingCapacity: false)
        self.fifoPreds = nil
        self.spkcachePreds = nil
        self.spkcacheLength = 0
        self.fifoLength = 0
        self.meanSilenceEmbedding.removeAll(keepingCapacity: false)
        self.silenceFrameCount = 0
    }
}

// MARK: - Streaming Feature Provider

/// Feature loader for Sortformer's file processing
public struct SortformerFeatureLoader: Sendable {
    public let numChunks: Int

    private let lc: Int
    private let rc: Int
    private let chunkLen: Int
    private let melFeatures: Int

    private let featSeq: [Float]
    private let featLength: Int
    private let featSeqLength: Int

    private var startFeat: Int
    private var endFeat: Int

    public init(config: SortformerConfig, audio: [Float]) {
        self.lc = config.chunkLeftContext * config.subsamplingFactor
        self.rc = config.chunkRightContext * config.subsamplingFactor
        self.chunkLen = config.chunkLen * config.subsamplingFactor
        self.melFeatures = config.melFeatures

        self.startFeat = 0
        self.endFeat = 0
        (self.featSeq, self.featLength, self.featSeqLength) = AudioMelSpectrogram().computeFlatTransposed(audio: audio)
        // numChunks accounts for right context requirement: need endFeat + rc <= featLength
        // Chunk n has endFeat = (n+1) * chunkLen, so valid when (n+1) * chunkLen + rc <= featLength
        // numChunks = floor((featLength - rc) / chunkLen)
        self.numChunks = max(0, (self.featLength - self.rc) / self.chunkLen)
    }

    public mutating func next() -> (chunkFeatures: [Float], chunkLength: Int, leftOffset: Int, rightOffset: Int)? {
        // Calculate end of core chunk
        endFeat = min(startFeat + chunkLen, featLength)

        // Need at least one core frame
        guard endFeat > startFeat else { return nil }

        // Require full right context (same as streaming getNextChunkFeatures)
        guard endFeat + rc <= featLength else { return nil }

        let leftOffset = min(lc, startFeat)
        let rightOffset = rc

        let chunkStartFrame = startFeat - leftOffset
        let chunkEndFrame = endFeat + rightOffset
        let chunkStartIndex = chunkStartFrame * melFeatures
        let chunkEndIndex = chunkEndFrame * melFeatures
        let chunkFeatures = Array(featSeq[chunkStartIndex..<chunkEndIndex])
        let chunkLength = max(min(featSeqLength - startFeat + leftOffset, chunkEndFrame - chunkStartFrame), 0)

        startFeat = endFeat
        return (chunkFeatures, chunkLength, leftOffset, rightOffset)
    }
}

// MARK: - Result Types

/// Result from streaming state update containing both confirmed and tentative predictions.
///
/// - `confirmed`: Predictions for frames that have passed beyond the right context window.
///   These are final and will not change.
/// - `tentative`: Predictions for frames still within the right context window.
///   These may change when the next chunk arrives with more future context.
///
/// This enables real-time UI display without waiting for the full right context delay.
/// With rightContext=7 and 80ms frames, tentative predictions provide 560ms earlier feedback.
public struct StreamingUpdateResult: Sendable {
    /// Final predictions for confirmed frames [chunkLen * numSpeakers]
    public let confirmed: [Float]

    /// Tentative predictions for right context frames [rightContext * numSpeakers]
    /// May change with next chunk. Empty if rightContext=0.
    public let tentative: [Float]

    /// Number of speakers
    public let numSpeakers: Int

    /// Number of confirmed frames
    public var confirmedFrameCount: Int { confirmed.count / numSpeakers }  // Assumes 4 speakers

    /// Number of tentative frames
    public var tentativeFrameCount: Int { tentative.count / numSpeakers }  // Assumes 4 speakers

    public init(confirmed: [Float], tentative: [Float], numSpeakers: Int = 4) {
        self.confirmed = confirmed
        self.tentative = tentative
        self.numSpeakers = numSpeakers
    }
}

// MARK: - Errors

public enum SortformerError: Error, LocalizedError {
    case notInitialized
    case modelLoadFailed(String)
    case preprocessorFailed(String)
    case inferenceFailed(String)
    case invalidAudioData
    case invalidState(String)
    case configurationError(String)
    case insufficientChunkLength(String)
    case insufficientPredsLength(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Sortformer diarizer not initialized. Call initialize() first."
        case .modelLoadFailed(let message):
            return "Failed to load Sortformer model: \(message)"
        case .preprocessorFailed(let message):
            return "Preprocessor failed: \(message)"
        case .inferenceFailed(let message):
            return "Inference failed: \(message)"
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .invalidState(let message):
            return "Invalid state: \(message)"
        case .configurationError(let message):
            return "Configuration error: \(message)"
        case .insufficientChunkLength(let message):
            return "Insufficient chunk length: \(message)"
        case .insufficientPredsLength(let message):
            return "Insufficient preds length: \(message)"
        }
    }
}
