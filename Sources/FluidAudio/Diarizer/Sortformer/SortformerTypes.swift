import Foundation

// MARK: - Configuration

/// Configuration for Sortformer streaming diarization.
///
/// Based on NVIDIA's Streaming Sortformer 4-speaker model.
/// Reference: NeMo sortformer_modules.py
public struct SortformerConfig: Sendable {

    // MARK: - Model Architecture

    /// Number of speaker slots (fixed at 4 for current model)
    public let numSpeakers: Int = 4

    /// Pre-encoder embedding dimension
    public let preEncoderDims: Int = 512

    /// NEST encoder embedding dimension
    public let nestEncoderDims: Int = 192

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
    
    /// Threshold for speech detection
    public var speechThreshold: Float = 0.5

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

    /// NVIDIA's 30.4s latency configuration
    public static let nvidiaHighLatency = SortformerConfig(
        chunkLen: 340,
        chunkLeftContext: 1,
        chunkRightContext: 40,
        fifoLen: 40,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 300
    )

    /// NVIDIA's 1.04s latency configuration (20.57% DER on AMI SDM)
    public static let nvidiaLowLatency = SortformerConfig(
        chunkLen: 6,
        chunkLeftContext: 1,
        chunkRightContext: 7,
        fifoLen: 188,
        spkcacheLen: 188,
        spkcacheUpdatePeriod: 144
    )

    /// - Warning: If you don't use one of the default configurations, you must use a local model converted with that configuration.
    public init(
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

/// Configuration for post-processing Sortformer diarizer predictions
public struct SortformerTimelineConfig {
    /// Onset threshold for detecting the beginning and end of a speech
    public var onsetThreshold: Float

    /// Offset threshold for detecting the end of a speech
    public var offsetThreshold: Float

    /// Adding frames before each speech segment
    public var onsetPadFrames: Int

    /// Adding frames after each speech segment
    public var offsetPadFrames: Int

    /// Threshold for short speech segment deletion in frames
    public var minFramesOn: Int

    /// Threshold for small non-speech deletion in frames
    public var minFramesOff: Int
    
    /// Number of frames in the FIFO queue that should not be used to update predictions
    public var filterLeftContext: Int
    
    /// Number of tentative frames
    internal var numFilteredFrames: Int

    /// Adding durations before each speech segment
    public var onsetPadSeconds: Float {
        get { Float(onsetPadFrames) * frameDurationSeconds }
        set { onsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    /// Adding durations after each speech segment
    public var offsetPadSeconds: Float {
        get { Float(offsetPadFrames) * frameDurationSeconds }
        set { offsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    /// Threshold for short speech segment deletion (seconds)
    public var minDurationOn: Float {
        get { Float(minFramesOn) * frameDurationSeconds }
        set { minFramesOn = Int(round(newValue / frameDurationSeconds)) }
    }

    /// Threshold for small non-speech deletion (seconds)
    public var minDurationOff: Float {
        get { Float(minFramesOff) * frameDurationSeconds }
        set { minFramesOff = Int(round(newValue / frameDurationSeconds)) }
    }
    
    /// Minimum gap between two segments
    public var minUnpaddedGap: Int {
        onsetPadFrames + offsetPadFrames + minFramesOff
    }

    /// Maximum number of predictions to retain
    public var maxStoredFrames: Int? = nil

    /// Number of speakers
    public let numSpeakers: Int = 4
    
    /// Clustering threshold to detect a new speaker
    public var clusteringThreshold: Float
    
    /// Chamfer distance threshold to match with another speaker profile
    public var matchThreshold: Float

    /// Duration of a frame in seconds
    public static let frameDurationSeconds: Float = 0.08
    
    /// Duration of a frame in seconds
    public var frameDurationSeconds: Float { Self.frameDurationSeconds }

    /// Default configurations
    public static func `default`(for config: SortformerConfig) -> SortformerTimelineConfig {
        SortformerTimelineConfig(
            for: config,
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 0,
            offsetPadFrames: 0,
            minFramesOn: 1,
            minFramesOff: 1,
            filterLeftContext: 1,
            clusteringThreshold: 0.25,
            matchThreshold: 0.3
        )
    }
    
    public init(
        for config: SortformerConfig,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadSeconds: Float = 0,
        offsetPadSeconds: Float = 0,
        minDurationOn: Float = 0,
        minDurationOff: Float = 0,
        maxStoredFrames: Int? = nil,
        filterLeftContext: Int = 1,
        clusteringThreshold: Float = 0.25,
        matchThreshold: Float = 0.3
    ) {
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = Int(round(onsetPadSeconds / Self.frameDurationSeconds))
        self.offsetPadFrames = Int(round(offsetPadSeconds / Self.frameDurationSeconds))
        self.minFramesOn = Int(round(minDurationOn / Self.frameDurationSeconds))
        self.minFramesOff = Int(round(minDurationOff / Self.frameDurationSeconds))
        self.maxStoredFrames = maxStoredFrames
        self.clusteringThreshold = clusteringThreshold
        self.matchThreshold = matchThreshold
        
        self.filterLeftContext = min(filterLeftContext, config.fifoLen - config.spkcacheUpdatePeriod)
        self.numFilteredFrames = config.fifoLen + config.chunkRightContext - self.filterLeftContext
    }

    public init(
        for config: SortformerConfig,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadFrames: Int = 0,
        offsetPadFrames: Int = 0,
        minFramesOn: Int = 0,
        minFramesOff: Int = 0,
        maxStoredFrames: Int? = nil,
        filterLeftContext: Int = 1,
        clusteringThreshold: Float = 0.25,
        matchThreshold: Float = 0.3
    ) {
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = onsetPadFrames
        self.offsetPadFrames = offsetPadFrames
        self.minFramesOn = minFramesOn
        self.minFramesOff = minFramesOff
        self.maxStoredFrames = maxStoredFrames
        self.clusteringThreshold = clusteringThreshold
        self.matchThreshold = matchThreshold
        
        self.filterLeftContext = max(filterLeftContext, config.fifoLen - config.spkcacheUpdatePeriod)
        self.numFilteredFrames = config.fifoLen + config.chunkRightContext - self.filterLeftContext
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
    public var spkcachePreds: [Float]

    /// FIFO queue of recent chunk embeddings
    /// Shape: [fifoLen, fcDModel] (e.g., [188, 512])
    public var fifo: [Float]

    /// Current valid length of FIFO queue
    public var fifoLength: Int

    /// Speaker predictions for FIFO embeddings
    /// Shape: [fifoLen, numSpeakers] (e.g., [188, 4])
    public var fifoPreds: [Float] = []

    /// Running mean of silence embeddings
    /// Shape: [fcDModel] (e.g., [512])
    public var meanSilenceEmbedding: [Float]

    /// Count of silence frames observed
    public var silenceFrameCount: Int
    
    /// Frame index of the next new frame
    public var nextNewFrame: Int
    
    /// Number of right context frames in the last chunk
    public var lastRightContext: Int

    /// Initialize empty streaming state
    public init(config: SortformerConfig) {
        self.spkcache = []
        self.spkcachePreds = []
        self.spkcacheLength = 0

        self.fifo = []
        self.fifoPreds = []
        self.fifoLength = 0

        self.fifo.reserveCapacity((config.fifoLen + config.chunkLen) * config.preEncoderDims)
        self.spkcache.reserveCapacity((config.spkcacheLen + config.spkcacheUpdatePeriod) * config.preEncoderDims)

        self.meanSilenceEmbedding = [Float](repeating: 0.0, count: config.preEncoderDims)
        self.silenceFrameCount = 0
        
        self.nextNewFrame = 0
        self.lastRightContext = 0
    }

    public mutating func cleanup() {
        self.fifo.removeAll(keepingCapacity: false)
        self.spkcache.removeAll(keepingCapacity: false)
        self.fifoPreds.removeAll(keepingCapacity: false)
        self.spkcachePreds.removeAll(keepingCapacity: false)
        self.spkcacheLength = 0
        self.fifoLength = 0
        self.meanSilenceEmbedding.removeAll(keepingCapacity: false)
        self.silenceFrameCount = 0
        self.nextNewFrame = 0
        self.lastRightContext = 0
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
    
    private var chunkIndex: Int = 0

    public init(config: SortformerConfig, audio: [Float]) {
        self.lc = config.chunkLeftContext * config.subsamplingFactor
        self.rc = config.chunkRightContext * config.subsamplingFactor
        self.chunkLen = config.chunkLen * config.subsamplingFactor
        self.melFeatures = config.melFeatures

        self.startFeat = 0
        self.endFeat = 0
        (self.featSeq, self.featLength, self.featSeqLength) = NeMoMelSpectrogram().computeFlatTransposed(audio: audio)
        self.numChunks = IndexUtils.ceilDiv(self.featLength, self.chunkLen)
    }

    public mutating func next() -> (chunkIndex: Int, chunkFeatures: [Float], chunkLength: Int, leftOffset: Int, rightOffset: Int)? {
        guard endFeat < featLength else { return nil }

        let leftOffset = min(lc, startFeat)
        endFeat = min(startFeat + chunkLen, featLength)
        let rightOffset = min(rc, featLength - endFeat)

        let chunkStartFrame = startFeat - leftOffset
        let chunkEndFrame = endFeat + rightOffset
        
        let chunkStartIndex = chunkStartFrame * melFeatures
        let chunkEndIndex = chunkEndFrame * melFeatures
        
        let chunkFeatures = Array(featSeq[chunkStartIndex..<chunkEndIndex])
        let chunkLength = max(min(featSeqLength - chunkStartFrame, chunkEndFrame - chunkStartFrame), 0)

        defer {
            startFeat = endFeat
            chunkIndex += 1
        }
        
        return (chunkIndex, chunkFeatures, chunkLength, leftOffset, rightOffset)
    }
}

// MARK: - Result Types

/// Result from a single streaming diarization step
public struct SortformerStateUpdateResult: Sendable {
    /// Speaker probabilities for frames in this chunk that don't overlap with anything else
    /// Shape: [chunkLen, numSpeakers] (e.g., [6, 4])
    public let newPredictions: [Float]

    /// Number of confirmed frames in this result
    public let newFrameCount: Int

    /// Global index of the first new frame
    public let firstNewFrame: Int
    
    /// Speaker probabilities for frames that overlap with old predictions
    /// Shape: [fifoLength + rc, numSpeakers] (e.g., [47, 4])
    public let oldPredictions: [Float]
    
    /// Number of frames in the FIFO queue
    public let oldFrameCount: Int

    /// Global index of the first frame in the FIFO queue
    public var firstOldFrame: Int { firstNewFrame - oldFrameCount }
    
    /// Number of frames that should be finalized
    public let finalizedFrameCount: Int
    
    public init(
        firstNewFrame: Int,
        newPredictions: [Float],
        newFrameCount: Int,
        oldPredictions: [Float],
        oldFrameCount: Int,
        finalizedFrameCount: Int
    ) {
        self.firstNewFrame = firstNewFrame
        self.newPredictions = newPredictions
        self.newFrameCount = newFrameCount
        self.oldPredictions = oldPredictions
        self.oldFrameCount = oldFrameCount
        self.finalizedFrameCount = finalizedFrameCount
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

public protocol SortformerFrameRange {
    /// Start frame index
    var startFrame: Int { get }
    
    /// End frame index
    var endFrame: Int { get }
    
    /// Range of frames spanned
    var frames: Range<Int> { get }
    
    /// Length of the frame range
    var length: Int { get }
    
    /// Check if the range contains a frame
    func contains(_ frame: Int) -> Bool
    
    /// Check if the ranges overlap or touch
    func isContiguous<T>(with other: T) -> Bool
    where T: SortformerFrameRange
    
    /// Check if the ranges share any frames
    func overlaps<T>(with other: T) -> Bool
    where T: SortformerFrameRange
    
    /// Count the number of overlapping frames
    func overlapLength<T>(with other: T) -> Int
    where T: SortformerFrameRange
}

public protocol SpeakerFrameRange: SortformerFrameRange {
    /// Speaker index in Sortformer's output
    var speakerId: Int { get }

    /// Check if the ranges overlap or touch
    func isContiguous<T>(with other: T, ensuringSameSpeaker: Bool) -> Bool
    where T: SpeakerFrameRange
    
    /// Check if the ranges share any frames
    func overlaps<T>(with other: T, ensuringSameSpeaker: Bool) -> Bool
    where T: SpeakerFrameRange
    
    /// Count the number of overlapping frames
    func overlapLength<T>(with other: T, ensuringSameSpeaker: Bool) -> Int
    where T: SpeakerFrameRange
}

internal struct SortformerFrameRangeHelpers {
    static func overlaps<L, R>(_ lhs: L, _ rhs: R, ensuringSameSpeaker: Bool) -> Bool where L: SpeakerFrameRange, R: SpeakerFrameRange {
        let sameSpeaker = !ensuringSameSpeaker || lhs.speakerId == rhs.speakerId
        return sameSpeaker && lhs.frames.overlaps(rhs.frames)
    }
    
    static func isContiguous<L, R>(_ lhs: L, _ rhs: R) -> Bool where L: SortformerFrameRange, R: SortformerFrameRange {
        return lhs.startFrame <= rhs.endFrame && lhs.endFrame >= rhs.startFrame
    }
    
    static func isContiguous<L, R>(_ lhs: L, _ rhs: R, ensuringSameSpeaker: Bool) -> Bool where L: SpeakerFrameRange, R: SpeakerFrameRange {
        let sameSpeaker = !ensuringSameSpeaker || lhs.speakerId == rhs.speakerId
        return sameSpeaker && lhs.startFrame <= rhs.endFrame && lhs.endFrame >= rhs.startFrame
    }
    
    static func overlapLength<L, R>(_ lhs: L, _ rhs: R) -> Int where L: SortformerFrameRange, R: SortformerFrameRange {
        let overlapStart = max(lhs.startFrame, rhs.startFrame)
        let overlapEnd = min(lhs.endFrame, rhs.endFrame)
        return max(0, overlapEnd - overlapStart)
    }
    
    static func overlapLength<L, R>(_ lhs: L, _ rhs: R, ensuringSameSpeaker: Bool) -> Int where L: SpeakerFrameRange, R: SpeakerFrameRange {
        guard !ensuringSameSpeaker || lhs.speakerId == rhs.speakerId else { return 0 }
        let overlapStart = max(lhs.startFrame, rhs.startFrame)
        let overlapEnd = min(lhs.endFrame, rhs.endFrame)
        return max(0, overlapEnd - overlapStart)
    }
    
    static func checkEqual<L, R>(_ lhs: L, _ rhs: R) -> Bool where L: SortformerFrameRange, R: SortformerFrameRange {
        return lhs.startFrame == rhs.startFrame && lhs.endFrame == rhs.endFrame
    }
    
    static func checkEqual<L, R>(_ lhs: L, _ rhs: R) -> Bool where L: SpeakerFrameRange, R: SpeakerFrameRange {
        return lhs.speakerId == rhs.speakerId && lhs.startFrame == rhs.startFrame && lhs.endFrame == rhs.endFrame
    }
    
    static func checkLessThan<L, R>(_ lhs: L, _ rhs: R) -> Bool where L: SortformerFrameRange, R: SortformerFrameRange {
        return (lhs.startFrame, lhs.endFrame) < (rhs.startFrame, rhs.endFrame)
    }
}
