import Foundation

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

    /// Sample rate in Hz}
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
public struct SortformerPostProcessingConfig {
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

    /// Maximum number of predictions to retain
    public var maxStoredFrames: Int? = nil

    /// Number of speakers
    public let numSpeakers: Int = 4

    /// Number of speakers
    public let frameDurationSeconds: Float = 0.08

    /// Default configurations
    public static var `default`: SortformerPostProcessingConfig {
        SortformerPostProcessingConfig(
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 0,
            offsetPadFrames: 0,
            minFramesOn: 0,
            minFramesOff: 0
        )
    }

    public init(
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadSeconds: Float = 0,
        offsetPadSeconds: Float = 0,
        minDurationOn: Float = 0,
        minDurationOff: Float = 0,
        maxStoredFrames: Int? = nil
    ) {
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = Int(round(onsetPadSeconds / frameDurationSeconds))
        self.offsetPadFrames = Int(round(offsetPadSeconds / frameDurationSeconds))
        self.minFramesOn = Int(round(minDurationOn / frameDurationSeconds))
        self.minFramesOff = Int(round(minDurationOff / frameDurationSeconds))
        self.maxStoredFrames = maxStoredFrames
    }

    public init(
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadFrames: Int = 0,
        offsetPadFrames: Int = 0,
        minFramesOn: Int = 0,
        minFramesOff: Int = 0,
        maxStoredFrames: Int? = nil
    ) {
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = onsetPadFrames
        self.offsetPadFrames = offsetPadFrames
        self.minFramesOn = minFramesOn
        self.minFramesOff = minFramesOff
        self.maxStoredFrames = maxStoredFrames
    }
}
