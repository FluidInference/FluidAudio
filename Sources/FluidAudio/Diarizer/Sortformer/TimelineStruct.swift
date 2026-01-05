import Foundation

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
