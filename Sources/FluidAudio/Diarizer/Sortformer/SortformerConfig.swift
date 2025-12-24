import Foundation

/// Configuration for Sortformer streaming diarization.
///
/// Based on NVIDIA's Streaming Sortformer 4-speaker model.
/// Reference: NeMo sortformer_modules.py
public struct SortformerConfig: Sendable {

    // MARK: - Model Architecture

    /// Number of speaker slots (fixed at 4 for current model)
    public let numSpeakers: Int = 4

    /// Embedding dimension from FastConformer encoder
    public let fcDModel: Int = 512

    /// Transformer hidden dimension
    public let tfDModel: Int = 192

    /// Subsampling factor (8:1 downsampling in encoder)
    public let subsamplingFactor: Int = 8

    // MARK: - Streaming Parameters

    /// Output diarization frames per chunk
    /// Must match the value used in CoreML conversion
    public var chunkLen: Int = 6

    /// Left context frames for chunk processing
    public var chunkLeftContext: Int = 1

    /// Right context frames for chunk processing
    public var chunkRightContext: Int = 1

    /// Maximum FIFO queue length (recent embeddings)
    /// Must match CoreML conversion: fifo_len=40
    public var fifoLen: Int = 40

    /// Maximum speaker cache length (historical embeddings)
    /// Must match CoreML conversion: spkcache_len=120
    public var spkcacheLen: Int = 120

    /// Period for speaker cache updates (frames)
    public var spkcacheUpdatePeriod: Int = 30

    /// Silence frames per speaker in compressed cache
    public var spkcacheSilFramesPerSpk: Int = 3

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

    // MARK: - Debug

    /// Enable debug logging
    public var debugMode: Bool = false

    /// Use simple state update (like Python test) instead of full NeMo logic
    public var useSimpleStateUpdate: Bool = false

    /// Use native Swift mel spectrogram instead of CoreML preprocessor
    /// This matches NeMo's full-audio preprocessing and produces exact frame counts
    public var useNativePreprocessing: Bool = false

    // MARK: - Computed Properties

    /// Total chunk frames for CoreML model input (includes left/right context)
    /// Formula: (chunk_len + left_context + right_context) * subsampling
    /// Default: (6 + 1 + 1) * 8 = 64 frames
    public var chunkFrames: Int {
        (chunkLen + chunkLeftContext + chunkRightContext) * subsamplingFactor
    }

    /// Override for preprocessor audio samples (NeMo adds internal padding)
    /// Set to nil to use the calculated value, or provide explicit count
    public var preprocessorAudioSamplesOverride: Int?

    /// Audio samples needed for one preprocessor chunk
    /// NeMo adds 16 samples padding on each side, so naive formula doesn't work exactly.
    /// Use preprocessorAudioSamplesOverride for accurate values determined empirically.
    public var preprocessorAudioSamples: Int {
        preprocessorAudioSamplesOverride ?? ((chunkFrames - 1) * melStride + melWindow)
    }

    /// Audio hop between preprocessor chunks
    public var audioHopSamples: Int {
        preprocessorAudioSamples - melWindow
    }

    /// Override for overlap frames (set explicitly to match Python's empirical value)
    public var overlapFramesOverride: Int?

    /// Overlap frames in mel features
    /// Python test uses OVERLAP_FRAMES = 3 empirically determined, not the formula
    public var overlapFrames: Int {
        overlapFramesOverride ?? ((melWindow - melStride) / melStride + 1)
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

    public static let `default` = SortformerConfig()

    /// Low-latency configuration matching the Python test that achieves ~74% DER
    /// Use with models from coreml_models/ (not coreml_models_nvidia/)
    public static var lowLatency: SortformerConfig {
        var config = SortformerConfig(
            chunkLen: 6,
            chunkLeftContext: 1,
            chunkRightContext: 1,
            fifoLen: 40,
            spkcacheLen: 120,
            spkcacheUpdatePeriod: 30
        )
        // For low-latency: (6 + 1 + 1) * 8 = 64 mel frames = 10480 samples
        config.preprocessorAudioSamplesOverride = 10480
        return config
    }

    /// NVIDIA's 1.04s latency configuration (20.57% DER on AMI SDM)
    /// Use with models from coreml_models_nvidia/
    public static var nvidia: SortformerConfig {
        var config = SortformerConfig(
            chunkLen: 6,
            chunkLeftContext: 1,
            chunkRightContext: 7,
            fifoLen: 188,
            spkcacheLen: 188,
            spkcacheUpdatePeriod: 144
        )
        // NeMo needs 17920 samples to produce exactly 112 mel frames
        // (not 18160 from naive formula due to internal padding)
        config.preprocessorAudioSamplesOverride = 17920
        // Note: Swift produces 105492 feature frames vs Python's 104935 due to
        // different preprocessing approaches (chunked CoreML vs full-audio NeMo).
        // This causes frame count differences in DER calculation.
        return config
    }

    public init(
        chunkLen: Int = 6,
        chunkLeftContext: Int = 1,
        chunkRightContext: Int = 1,
        fifoLen: Int = 40,
        spkcacheLen: Int = 120,
        spkcacheUpdatePeriod: Int = 30,
        silenceThreshold: Float = 0.2,
        debugMode: Bool = false
    ) {
        self.chunkLen = chunkLen
        self.chunkLeftContext = chunkLeftContext
        self.chunkRightContext = chunkRightContext
        self.fifoLen = fifoLen
        self.spkcacheLen = spkcacheLen
        self.spkcacheUpdatePeriod = spkcacheUpdatePeriod
        self.silenceThreshold = silenceThreshold
        self.debugMode = debugMode
    }
}
