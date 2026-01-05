import Foundation

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
        (self.featSeq, self.featLength, self.featSeqLength) = NeMoMelSpectrogram().computeFlatTransposed(audio: audio)
        self.numChunks = (self.featLength - 1) / (config.chunkLen * config.subsamplingFactor) + 1  // ceiling
    }

    public mutating func next() -> (chunkFeatures: [Float], chunkLength: Int, leftOffset: Int, rightOffset: Int)? {
        guard endFeat < featLength else {
            return nil
        }

        let leftOffset = min(lc, startFeat)
        endFeat = min(startFeat + chunkLen, featLength)
        let rightOffset = min(rc, featLength - endFeat)

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

    /// Number of confirmed frames
    public var confirmedFrameCount: Int { confirmed.count / 4 }  // Assumes 4 speakers

    /// Number of tentative frames
    public var tentativeFrameCount: Int { tentative.count / 4 }  // Assumes 4 speakers

    public init(confirmed: [Float], tentative: [Float]) {
        self.confirmed = confirmed
        self.tentative = tentative
    }
}

/// Result from a single streaming diarization step
public struct SortformerChunkResult: Sendable {
    /// Speaker probabilities for confirmed frames in this chunk
    /// Shape: [chunkLen, numSpeakers] (e.g., [6, 4])
    public let speakerPredictions: [Float]

    /// Number of confirmed frames in this result
    public let frameCount: Int

    /// Frame index of the first confirmed frame
    public let startFrame: Int

    /// Tentative predictions for right context frames (may change with next chunk)
    /// Shape: [rightContext, numSpeakers]. Empty if no right context.
    public let tentativePredictions: [Float]

    /// Number of tentative frames
    public let tentativeFrameCount: Int

    /// Frame index of first tentative frame
    public var tentativeStartFrame: Int {
        startFrame + frameCount
    }

    public init(
        startFrame: Int,
        speakerPredictions: [Float],
        frameCount: Int,
        tentativePredictions: [Float] = [],
        tentativeFrameCount: Int = 0
    ) {
        self.speakerPredictions = speakerPredictions
        self.frameCount = frameCount
        self.startFrame = startFrame
        self.tentativePredictions = tentativePredictions
        self.tentativeFrameCount = tentativeFrameCount
    }

    /// Get probability for a specific speaker at a specific confirmed frame
    public func getSpeakerPrediction(speaker: Int, frame: Int, numSpeakers: Int = 4) -> Float {
        guard frame < frameCount, speaker < numSpeakers else { return 0.0 }
        return speakerPredictions[frame * numSpeakers + speaker]
    }

    /// Get tentative probability for a specific speaker at a specific tentative frame
    public func getTentativePrediction(speaker: Int, frame: Int, numSpeakers: Int = 4) -> Float {
        guard frame < tentativeFrameCount, speaker < numSpeakers else { return 0.0 }
        return tentativePredictions[frame * numSpeakers + speaker]
    }
}

/// Complete diarization timeline managing streaming predictions and segments
public class SortformerTimeline {
    /// Post-processing configuration
    public let config: SortformerPostProcessingConfig

    /// Finalized frame-wise speaker predictions
    /// Shape: [numFrames, numSpeakers]
    public private(set) var framePredictions: [Float] = []

    /// Tentative predictions
    /// Shape: [numTentative, numSpeakers]
    public private(set) var tentativePredictions: [Float] = []

    /// Total number of finalized median-filtered frames
    public private(set) var numFrames: Int = 0

    /// Number of tentative frames (including right context frames from chunk)
    public var numTentative: Int {
        tentativePredictions.count / config.numSpeakers
    }

    /// Finalized segments (completely before the median filter boundary)
    public private(set) var segments: [[SortformerSegment]] = []

    /// Tentative segments (may change as more predictions arrive)
    public private(set) var tentativeSegments: [[SortformerSegment]] = []

    /// Get total duration of finalized predictions in seconds
    public var duration: Float {
        Float(numFrames) * config.frameDurationSeconds
    }

    /// Get total duration including tentative predictions in seconds
    public var tentativeDuration: Float {
        Float(numFrames + numTentative) * config.frameDurationSeconds
    }

    /// Active segments being built (one per speaker, nil if speaker not active)
    private var activeSpeakers: [Bool]
    private var activeStarts: [Int]
    private var recentSegments: [(start: Int, end: Int)]

    /// Logger for warnings
    private static let logger = AppLogger(category: "SortformerTimeline")

    /// Initialize with configuration for streaming usage
    /// - Parameters:
    ///   - config: Sortformer post-processing configuration
    public init(config: SortformerPostProcessingConfig = .default) {
        self.config = config
        self.activeStarts = Array(repeating: 0, count: config.numSpeakers)
        self.recentSegments = Array(repeating: (0, 0), count: config.numSpeakers)
        self.activeSpeakers = Array(repeating: false, count: config.numSpeakers)
        self.segments = Array(repeating: [], count: config.numSpeakers)
        self.tentativeSegments = Array(repeating: [], count: config.numSpeakers)
    }

    /// Initialize with existing probabilities (e.g. from batch processing or restored state)
    /// - Parameters:
    ///   - allPredictions: Raw speaker probabilities (flattened)
    ///   - config: Configuration object
    ///   - isComplete: If true, treats the provided probabilities as the complete timeline and finalizes everything immediately.
    ///                 If false, treats them as initial raw predictions that may be extended.
    public convenience init(
        allPredictions: [Float],
        config: SortformerPostProcessingConfig = .default,
        isComplete: Bool = true
    ) {
        self.init(config: config)
        let numFrames = allPredictions.count / config.numSpeakers
        self.updateSegments(
            predictions: allPredictions, numFrames: numFrames, isFinalized: true, addTrailingTentative: true)
        self.framePredictions = allPredictions
        self.numFrames = numFrames
        trimPredictions()

        if isComplete {
            // Finalize everything immediately
            finalize()
        }
    }

    /// Add a new chunk of predictions from the diarizer
    public func addChunk(_ chunk: SortformerChunkResult) {
        framePredictions.append(contentsOf: chunk.speakerPredictions)
        tentativePredictions = chunk.tentativePredictions
        for i in 0..<config.numSpeakers {
            tentativeSegments[i].removeAll(keepingCapacity: true)
        }

        updateSegments(
            predictions: chunk.speakerPredictions,
            numFrames: chunk.frameCount,
            isFinalized: true,
            addTrailingTentative: false  // Don't add here, will add after tentative processing
        )
        numFrames += chunk.frameCount

        updateSegments(
            predictions: chunk.tentativePredictions,
            numFrames: chunk.tentativeFrameCount,
            isFinalized: false,
            addTrailingTentative: true  // Add still-speaking segments here
        )
        trimPredictions()
    }

    private func updateSegments(
        predictions: [Float],
        numFrames: Int,
        isFinalized: Bool,
        addTrailingTentative: Bool
    ) {
        guard numFrames > 0 else { return }

        let frameOffset = self.numFrames
        let numSpeakers = config.numSpeakers
        let onset = config.onsetThreshold
        let offset = config.offsetThreshold
        let padOnset = config.onsetPadFrames
        let padOffset = config.offsetPadFrames
        let minFramesOn = config.minFramesOn
        let minFramesOff = config.minFramesOff

        // Segments ending after this frame should be tentative because:
        // 1. They might be extended by future predictions
        // 2. The gap-closer (minFramesOff) could merge them with future segments
        // We need buffer for: onset padding + offset padding + gap closer threshold
        let tentativeBuffer = padOnset + padOffset + minFramesOff
        let tentativeStartFrame = isFinalized ? (frameOffset + numFrames) - tentativeBuffer : 0

        for speakerIndex in 0..<numSpeakers {
            var start = activeStarts[speakerIndex]
            var speaking = activeSpeakers[speakerIndex]
            var lastSegment = recentSegments[speakerIndex]
            var wasLastSegmentFinal = isFinalized

            for i in 0..<numFrames {
                let index = speakerIndex + i * numSpeakers

                if speaking {
                    if predictions[index] >= offset {
                        continue
                    }

                    // Speaking -> not speaking
                    speaking = false
                    let end = frameOffset + i + padOffset

                    // Ensure segment is long enough
                    guard end - start > minFramesOn else {
                        continue
                    }

                    // Segment is only finalized if it ends BEFORE the tentative boundary
                    // This ensures gap-closer can still merge it with future segments
                    wasLastSegmentFinal = isFinalized && (end < tentativeStartFrame)

                    let newSegment = SortformerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: wasLastSegmentFinal,
                        frameDurationSeconds: config.frameDurationSeconds
                    )

                    if wasLastSegmentFinal {
                        segments[speakerIndex].append(newSegment)
                    } else {
                        tentativeSegments[speakerIndex].append(newSegment)
                    }
                    lastSegment = (start, end)

                } else if predictions[index] > onset {
                    // Not speaking -> speaking
                    start = max(0, frameOffset + i - padOnset)
                    speaking = true

                    if start - lastSegment.end <= minFramesOff {
                        // Merge with last segment to avoid overlap
                        start = lastSegment.start

                        if wasLastSegmentFinal {
                            _ = segments[speakerIndex].popLast()
                        } else {
                            _ = tentativeSegments[speakerIndex].popLast()
                        }
                    }
                }
            }

            if isFinalized {
                activeSpeakers[speakerIndex] = speaking
                activeStarts[speakerIndex] = start
                recentSegments[speakerIndex] = lastSegment
            }

            // Add still-speaking segment as tentative when requested
            // This is skipped during finalized processing in addChunk (tentative will be processed next)
            // But enabled for batch init and tentative processing
            if addTrailingTentative {
                let end = frameOffset + numFrames + padOffset
                if speaking && (end > start) {
                    let newSegment = SortformerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: false,
                        frameDurationSeconds: config.frameDurationSeconds
                    )
                    tentativeSegments[speakerIndex].append(newSegment)
                }
            }
        }
    }

    /// Reset the timeline to initial state
    public func reset() {
        framePredictions.removeAll()
        tentativePredictions.removeAll()
        numFrames = 0

        activeStarts = Array(repeating: 0, count: config.numSpeakers)
        activeSpeakers = Array(repeating: false, count: config.numSpeakers)
        recentSegments = Array(repeating: (0, 0), count: config.numSpeakers)
        segments = Array(repeating: [], count: config.numSpeakers)
        tentativeSegments = Array(repeating: [], count: config.numSpeakers)
    }

    /// Finalize all tentative data at end of recording
    /// Call this when no more chunks will be added to convert all tentative predictions and segments to finalized
    public func finalize() {
        framePredictions.append(contentsOf: self.tentativePredictions)
        numFrames += numTentative
        tentativePredictions.removeAll()
        for i in 0..<config.numSpeakers {
            segments[i].append(contentsOf: tentativeSegments[i])
            tentativeSegments[i].removeAll()

            if let lastSegment = segments[i].last, lastSegment.length < config.minFramesOn {
                segments[i].removeLast()
            }
        }
        trimPredictions()
    }

    /// Get probability for a specific speaker at a specific finalized frame
    public func probability(speaker: Int, frame: Int) -> Float {
        guard frame < numFrames, speaker < config.numSpeakers else { return 0.0 }
        return framePredictions[frame * config.numSpeakers + speaker]
    }

    /// Get tentative probability for a specific speaker at a specific tentative frame
    public func tentativeProbability(speaker: Int, frame: Int) -> Float {
        guard frame < numTentative, speaker < config.numSpeakers else { return 0.0 }
        return tentativePredictions[frame * config.numSpeakers + speaker]
    }

    /// Trim predictions to not take up so much space
    private func trimPredictions() {
        guard let maxStoredFrames = config.maxStoredFrames else {
            return
        }

        let numToRemove = framePredictions.count - maxStoredFrames * config.numSpeakers

        if numToRemove > 0 {
            framePredictions.removeFirst(numToRemove)
        }
    }
}

/// A single speaker segment from Sortformer
/// Can be mutated during streaming processing
public struct SortformerSegment: Sendable, Identifiable {
    /// Segment ID
    public let id: UUID

    /// Speaker index in Sortformer output
    public var speakerIndex: Int

    /// Index of segment start frame
    public var startFrame: Int

    /// Index of segment end frame
    public var endFrame: Int

    /// Length of the segment in frames
    public var length: Int { endFrame - startFrame }

    /// Whether this segment is finalized
    public var isFinalized: Bool

    /// Start time in seconds
    public var startTime: Float { Float(startFrame) * frameDurationSeconds }

    /// End time in seconds
    public var endTime: Float { Float(endFrame) * frameDurationSeconds }

    /// Duration in seconds
    public var duration: Float { Float(endFrame - startFrame) * frameDurationSeconds }

    /// Duration of one frame in seconds
    public let frameDurationSeconds: Float

    /// Speaker label (e.g., "Speaker 0")
    public var speakerLabel: String {
        "Speaker \(speakerIndex)"
    }

    public init(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        frameDurationSeconds: Float = 0.08
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
    }

    public init(
        speakerIndex: Int,
        startTime: Float,
        endTime: Float,
        finalized: Bool = true,
        frameDurationSeconds: Float = 0.08
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = Int(round(startTime / frameDurationSeconds))
        self.endFrame = Int(round(endTime / frameDurationSeconds))
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
    }

    /// Check if this overlaps with another segment
    public func overlaps(with other: SortformerSegment) -> Bool {
        return (self.startFrame <= other.endFrame) && (other.startFrame <= self.endFrame)
    }

    /// Merge another segment into this one
    public mutating func absorb(_ other: SortformerSegment) {
        self.startFrame = min(self.startFrame, other.startFrame)
        self.endFrame = max(self.endFrame, other.endFrame)
    }

    /// Extend the end of this segment
    public mutating func extendEnd(toFrame endFrame: Int) {
        self.endFrame = max(self.endFrame, endFrame)
    }

    /// Extend the start of this segment
    public mutating func extendStart(toFrame startFrame: Int) {
        self.startFrame = min(self.startFrame, startFrame)
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
