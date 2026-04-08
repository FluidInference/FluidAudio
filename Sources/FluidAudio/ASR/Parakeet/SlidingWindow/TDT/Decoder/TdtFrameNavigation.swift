import Foundation

/// Frame navigation utilities for TDT decoding.
///
/// Handles time index calculations for streaming ASR with chunk-based processing,
/// including timeJump management for decoder position tracking across chunks.
internal struct TdtFrameNavigation {

    /// Calculate initial time indices for chunk processing.
    ///
    /// Determines where to start processing in the current chunk based on:
    /// - Previous timeJump (how far past the previous chunk the decoder advanced)
    /// - Context frame adjustment (adaptive overlap compensation)
    ///
    /// - Parameters:
    ///   - timeJump: Optional timeJump from previous chunk (nil for first chunk)
    ///   - contextFrameAdjustment: Frame offset for adaptive context
    ///
    /// - Returns: Starting frame index for this chunk
    static func calculateInitialTimeIndices(
        timeJump: Int?,
        contextFrameAdjustment: Int
    ) -> Int {
        // First chunk: start from beginning, accounting for any context frames already processed
        guard let prevTimeJump = timeJump else {
            return contextFrameAdjustment
        }

        // Streaming continuation: timeJump represents decoder position beyond previous chunk
        // For the new chunk, we need to account for:
        // 1. How far the decoder advanced past the previous chunk (prevTimeJump)
        // 2. The overlap/context between chunks (contextFrameAdjustment)
        //
        // If prevTimeJump > 0: decoder went past previous chunk's frames
        // If contextFrameAdjustment < 0: decoder should skip frames (overlap with previous chunk)
        // If contextFrameAdjustment > 0: decoder should start later (adaptive context)
        // Net position = prevTimeJump + contextFrameAdjustment (add adjustment to decoder position)

        // SPECIAL CASE: When prevTimeJump = 0 and contextFrameAdjustment = 0,
        // decoder finished exactly at boundary but chunk has physical overlap
        // Need to skip the overlap frames to avoid re-processing
        if prevTimeJump == 0 && contextFrameAdjustment == 0 {
            // Skip standard overlap (2.0s = 25 frames at 0.08s per frame)
            return ASRConstants.standardOverlapFrames
        }

        // Normal streaming continuation
        return max(0, prevTimeJump + contextFrameAdjustment)
    }

    /// Initialize frame navigation state for decoding loop.
    ///
    /// - Parameters:
    ///   - timeIndices: Initial time index calculated from timeJump
    ///   - encoderSequenceLength: Total encoder frames in this chunk
    ///   - actualAudioFrames: Actual audio frames (excluding padding)
    ///
    /// - Returns: Tuple of navigation state values
    static func initializeNavigationState(
        timeIndices: Int,
        encoderSequenceLength: Int,
        actualAudioFrames: Int
    ) -> (
        effectiveSequenceLength: Int,
        safeTimeIndices: Int,
        lastTimestep: Int,
        activeMask: Bool
    ) {
        // Use the minimum of encoder sequence length and actual audio frames to avoid processing padding
        let effectiveSequenceLength = min(encoderSequenceLength, actualAudioFrames)

        // Key variables for frame navigation:
        let safeTimeIndices = min(timeIndices, effectiveSequenceLength - 1)  // Bounds-checked index
        let lastTimestep = effectiveSequenceLength - 1  // Maximum valid frame index
        let activeMask = timeIndices < effectiveSequenceLength  // Start processing only if we haven't exceeded bounds

        return (effectiveSequenceLength, safeTimeIndices, lastTimestep, activeMask)
    }

    /// Calculate final timeJump for streaming continuation.
    ///
    /// TimeJump tracks how far beyond the current chunk the decoder has advanced,
    /// which is used to properly position the decoder in the next chunk.
    ///
    /// - Parameters:
    ///   - currentTimeIndices: Final time index after processing
    ///   - effectiveSequenceLength: Number of valid frames in this chunk
    ///   - isLastChunk: Whether this is the last chunk (no more chunks to process)
    ///
    /// - Returns: TimeJump value (nil for last chunk, otherwise offset from chunk boundary)
    static func calculateFinalTimeJump(
        currentTimeIndices: Int,
        effectiveSequenceLength: Int,
        isLastChunk: Bool
    ) -> Int? {
        // For the last chunk, clear timeJump since there are no more chunks
        if isLastChunk {
            return nil
        }

        // Always store time jump for streaming: how far beyond this chunk we've processed
        // Used to align timestamps when processing next chunk
        // Formula: timeJump = finalPosition - effectiveFrames
        return currentTimeIndices - effectiveSequenceLength
    }
}
