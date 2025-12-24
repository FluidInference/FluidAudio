import Accelerate
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

    /// Count of chunks processed (for simple state update)
    public var chunkCount: Int

    /// Speaker permutation (for training, not used in inference)
    public var speakerPermutation: [Int]?

    /// Initialize empty streaming state
    public init(config: SortformerConfig) {
        let fcDModel = config.fcDModel

        // Start with empty caches (synchronous mode)
        self.spkcache = []
        self.spkcacheLength = 0
        self.spkcachePreds = nil

        self.fifo = []
        self.fifoLength = 0
        self.fifoPreds = nil

        self.meanSilenceEmbedding = [Float](repeating: 0.0, count: fcDModel)
        self.silenceFrameCount = 0
        self.chunkCount = 0
        self.speakerPermutation = nil
    }

    /// Initialize with pre-allocated buffers (async streaming mode)
    public init(config: SortformerConfig, preallocate: Bool) {
        let fcDModel = config.fcDModel
        let numSpeakers = config.numSpeakers
        let spkcacheLen = config.spkcacheLen
        let fifoLen = config.fifoLen

        if preallocate {
            self.spkcache = [Float](repeating: 0.0, count: spkcacheLen * fcDModel)
            self.spkcacheLength = 0
            self.spkcachePreds = [Float](repeating: 0.0, count: spkcacheLen * numSpeakers)

            self.fifo = [Float](repeating: 0.0, count: fifoLen * fcDModel)
            self.fifoLength = 0
            self.fifoPreds = [Float](repeating: 0.0, count: fifoLen * numSpeakers)
        } else {
            self.spkcache = []
            self.spkcacheLength = 0
            self.spkcachePreds = nil

            self.fifo = []
            self.fifoLength = 0
            self.fifoPreds = nil
        }

        self.meanSilenceEmbedding = [Float](repeating: 0.0, count: fcDModel)
        self.silenceFrameCount = 0
        self.chunkCount = 0
        self.speakerPermutation = nil
    }
}

// MARK: - Result Types

/// Result from a single streaming diarization step
public struct SortformerChunkResult: Sendable {
    /// Speaker probabilities for this chunk
    /// Shape: [chunkLen, numSpeakers] (e.g., [4, 4])
    public let probabilities: [Float]

    /// Number of frames in this result
    public let frameCount: Int

    /// Time offset of first frame in seconds
    public let startTimeSeconds: Float

    public init(probabilities: [Float], frameCount: Int, startTimeSeconds: Float) {
        self.probabilities = probabilities
        self.frameCount = frameCount
        self.startTimeSeconds = startTimeSeconds
    }

    /// Get probability for a specific speaker at a specific frame
    public func probability(speaker: Int, frame: Int, numSpeakers: Int = 4) -> Float {
        guard frame < frameCount, speaker < numSpeakers else { return 0.0 }
        return probabilities[frame * numSpeakers + speaker]
    }
}

/// Complete diarization result with speaker segments
public struct SortformerResult: Sendable {
    /// All speaker probability frames
    public let allProbabilities: [Float]

    /// Total number of frames processed
    public let totalFrames: Int

    /// Frame duration in seconds
    public let frameDurationSeconds: Float

    /// Derived speaker segments (threshold-based)
    public var segments: [SortformerSegment] {
        extractSegments(threshold: 0.5)
    }

    public init(allProbabilities: [Float], totalFrames: Int, frameDurationSeconds: Float) {
        self.allProbabilities = allProbabilities
        self.totalFrames = totalFrames
        self.frameDurationSeconds = frameDurationSeconds
    }

    /// Compute statistics about the probability distribution
    public func probabilityStats() -> (min: Float, max: Float, mean: Float, above05: Int, total: Int) {
        guard !allProbabilities.isEmpty else {
            return (0, 0, 0, 0, 0)
        }
        var minProb: Float = 1.0
        var maxProb: Float = 0.0
        var sum: Float = 0.0
        var above05 = 0
        for p in allProbabilities {
            minProb = min(minProb, p)
            maxProb = max(maxProb, p)
            sum += p
            if p > 0.5 {
                above05 += 1
            }
        }
        return (
            min: minProb, max: maxProb, mean: sum / Float(allProbabilities.count),
            above05: above05, total: allProbabilities.count
        )
    }

    /// Extract speaker segments using a probability threshold
    /// Applies median filtering (kernel=7) matching Python's scipy.ndimage.median_filter
    public func extractSegments(threshold: Float = 0.5, medianKernel: Int = 7) -> [SortformerSegment] {
        var segments: [SortformerSegment] = []
        let numSpeakers = 4

        // Apply median filter per speaker (matches Python's median_filter(probs, size=(7, 1)))
        let filteredProbs = applyMedianFilter(kernel: medianKernel, numSpeakers: numSpeakers)

        for speaker in 0..<numSpeakers {
            var isActive = false
            var startFrame = 0

            for frame in 0..<totalFrames {
                let prob = filteredProbs[frame * numSpeakers + speaker]

                if prob > threshold && !isActive {
                    isActive = true
                    startFrame = frame
                } else if prob <= threshold && isActive {
                    let segment = SortformerSegment(
                        speakerIndex: speaker,
                        startFrame: startFrame,
                        endFrame: frame,
                        frameDurationSeconds: frameDurationSeconds
                    )
                    segments.append(segment)
                    isActive = false
                }
            }

            // Handle segment that extends to end
            if isActive {
                let segment = SortformerSegment(
                    speakerIndex: speaker,
                    startFrame: startFrame,
                    endFrame: totalFrames,
                    frameDurationSeconds: frameDurationSeconds
                )
                segments.append(segment)
            }
        }

        // Sort by start time
        return segments.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
    }

    /// Apply 1D median filter along time axis for each speaker
    /// Matches Python's scipy.ndimage.median_filter(probs, size=(kernel, 1))
    public func applyMedianFilter(kernel: Int = 7, numSpeakers: Int = 4) -> [Float] {
        guard kernel > 1 else { return allProbabilities }

        var filtered = [Float](repeating: 0.0, count: allProbabilities.count)
        let halfKernel = kernel / 2

        for speaker in 0..<numSpeakers {
            // Extract time series for this speaker
            var timeSeries = [Float](repeating: 0.0, count: totalFrames)
            for frame in 0..<totalFrames {
                timeSeries[frame] = allProbabilities[frame * numSpeakers + speaker]
            }

            // Apply median filter
            for frame in 0..<totalFrames {
                // Get window bounds (handle edges)
                let windowStart = max(0, frame - halfKernel)
                let windowEnd = min(totalFrames, frame + halfKernel + 1)

                // Extract window values and sort to find median
                var window = [Float]()
                for i in windowStart..<windowEnd {
                    window.append(timeSeries[i])
                }
                window.sort()

                // Median is middle element
                let median = window[window.count / 2]
                filtered[frame * numSpeakers + speaker] = median
            }
        }

        return filtered
    }
}

/// A single speaker segment from Sortformer
public struct SortformerSegment: Sendable, Identifiable {
    public let id = UUID()

    /// Speaker index (0-3)
    public let speakerIndex: Int

    /// Start frame index
    public let startFrame: Int

    /// End frame index (exclusive)
    public let endFrame: Int

    /// Frame duration in seconds
    public let frameDurationSeconds: Float

    /// Start time in seconds
    public var startTimeSeconds: Float {
        Float(startFrame) * frameDurationSeconds
    }

    /// End time in seconds
    public var endTimeSeconds: Float {
        Float(endFrame) * frameDurationSeconds
    }

    /// Duration in seconds
    public var durationSeconds: Float {
        endTimeSeconds - startTimeSeconds
    }

    /// Speaker label (e.g., "Speaker_0")
    public var speakerLabel: String {
        "Speaker_\(speakerIndex)"
    }

    public init(speakerIndex: Int, startFrame: Int, endFrame: Int, frameDurationSeconds: Float) {
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.frameDurationSeconds = frameDurationSeconds
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
        }
    }
}
