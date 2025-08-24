import CoreML
import Foundation
import OSLog

/// Shared sliding-window utilities and lightweight state for chunked and streaming ASR
/// Centralizes: start-frame offset computation and token de-duplication across windows
struct SlidingWindowAdapter {
    // Window configuration (kept consistent with ChunkProcessor defaults)
    let sampleRate: Int
    let centerSeconds: Double
    let leftContextSeconds: Double
    let rightContextSeconds: Double
    let enableDebug: Bool

    // Streaming/chunk continuity state
    private(set) var segmentIndex: Int = 0
    private(set) var lastProcessedFrame: Int = 0
    private var accumulatedTokens: [Int] = []

    init(
        sampleRate: Int = 16_000,
        centerSeconds: Double = 10.0,
        leftContextSeconds: Double = 2.0,
        rightContextSeconds: Double = 2.0,
        enableDebug: Bool = false
    ) {
        self.sampleRate = sampleRate
        self.centerSeconds = centerSeconds
        self.leftContextSeconds = leftContextSeconds
        self.rightContextSeconds = rightContextSeconds
        self.enableDebug = enableDebug
    }

    /// Compute a conservative start-frame offset for subsequent windows
    /// Matches the logic used in ChunkProcessor and NeMo-style context trimming
    func startFrameOffset(forSegment segment: Int) -> Int {
        guard segment > 0 else { return 0 }
        let exactEncoderFrameRate = 12.6
        let leftContextFrames = Int(round(leftContextSeconds * exactEncoderFrameRate))
        return leftContextFrames + 2 // ~29 frames at 12.6 fps for 2s left context
    }

    /// Remove duplicate token sequences at the start of the current list that overlap
    /// with the tail of the previous accumulated tokens. Returns deduplicated current tokens
    /// and the number of removed leading tokens so caller can drop aligned timestamps.
    func removeDuplicateSequence(previous: [Int], current: [Int]) -> (deduped: [Int], removedCount: Int) {
        let maxSearchLength = min(15, previous.count) // last 15 tokens of previous
        let maxMatchLength = min(12, current.count)   // first 12 tokens of current

        guard maxSearchLength >= 3 && maxMatchLength >= 3 else {
            return (current, 0)
        }

        for overlapLength in (3...min(maxSearchLength, maxMatchLength)).reversed() {
            let prevStart = max(0, previous.count - maxSearchLength)
            let prevEnd = previous.count - overlapLength + 1
            if prevEnd <= prevStart { continue }
            for startIndex in prevStart..<prevEnd {
                let prevSub = Array(previous[startIndex..<(startIndex + overlapLength)])
                let currEnd = max(0, current.count - overlapLength + 1)
                for currentStart in 0..<min(5, currEnd) {
                    let currSub = Array(current[currentStart..<(currentStart + overlapLength)])
                    if prevSub == currSub {
                        if enableDebug {
                            print("Duplicate sequence length=\(overlapLength) at currStart=\(currentStart): \(prevSub)")
                        }
                        let removed = currentStart + overlapLength
                        return (Array(current.dropFirst(removed)), removed)
                    }
                }
            }
        }
        return (current, 0)
    }

    /// Process a single chunk/window for streaming using the shared logic.
    /// - Returns: deduplicated tokens/timestamps and updates internal continuity state.
    mutating func processStreamingChunk(
        manager: AsrManager,
        source: AudioSource,
        chunkSamples: [Float]
    ) async throws -> (tokens: [Int], timestamps: [Int], maxFrame: Int) {
        let startOffset = startFrameOffset(forSegment: segmentIndex)

        // Delegate to AsrManager to run ML inference with proper decoder state
        let (tokens, timestamps, encoderSeqLen) = try await manager.transcribeStreamingChunk(
            chunkSamples,
            source: source,
            startFrameOffset: startOffset,
            lastProcessedFrame: lastProcessedFrame,
            enableDebug: enableDebug
        )

        guard !tokens.isEmpty, encoderSeqLen > 0 else {
            // advance indices even if empty to avoid stalling
            segmentIndex += 1
            return ([], [], 0)
        }

        let prev = accumulatedTokens
        let (deduped, removedCount) = removeDuplicateSequence(previous: prev, current: tokens)
        let adjustedTimestamps = removedCount > 0 ? Array(timestamps.dropFirst(removedCount)) : timestamps
        let maxFrame = adjustedTimestamps.max() ?? 0

        if enableDebug, removedCount > 0 {
            print("Streaming chunk \(segmentIndex): removed \(removedCount) duplicate tokens")
        }

        // Update continuity state
        accumulatedTokens.append(contentsOf: deduped)
        lastProcessedFrame = max(lastProcessedFrame, maxFrame)
        segmentIndex += 1

        return (deduped, adjustedTimestamps, maxFrame)
    }
}

