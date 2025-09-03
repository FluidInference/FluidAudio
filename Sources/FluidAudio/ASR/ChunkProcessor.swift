import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]
    let enableDebug: Bool

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ChunkProcessor")

    // Frame-aligned configuration: 8 + 3.2 + 3.2 seconds context at 16kHz
    // 8s center = exactly 100 encoder frames
    // 3.2s context = exactly 40 encoder frames each
    // Total: 14.4s (within 15s model limit, 180 total frames)
    private let sampleRate: Int = 16000
    private let centerSeconds: Double = 11.2  // Exactly 100 frames (8.0 * 12.5)
    private let leftContextSeconds: Double = 1.6  // Exactly 40 frames (3.2 * 12.5)
    private let rightContextSeconds: Double = 1.6  // Exactly 40 frames (3.2 * 12.5)

    private var centerSamples: Int { Int(centerSeconds * Double(sampleRate)) }
    private var leftContextSamples: Int { Int(leftContextSeconds * Double(sampleRate)) }
    private var rightContextSamples: Int { Int(rightContextSeconds * Double(sampleRate)) }
    private var maxModelSamples: Int { 240_000 }  // 15 seconds window capacity

    func process(
        using manager: AsrManager, decoderState: inout TdtDecoderState, startTime: Date
    ) async throws -> ASRResult {
        var allTokens: [Int] = []
        var allTimestamps: [Int] = []

        var centerStart = 0
        var segmentIndex = 0
        var lastProcessedFrame = 0  // Track the last frame processed by previous chunk

        while centerStart < audioSamples.count {
            // Determine if this is the last chunk
            let isLastChunk = (centerStart + centerSamples) >= audioSamples.count

            // Process chunk with explicit last chunk detection

            let (windowTokens, windowTimestamps, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &decoderState
            )

            // Update last processed frame for next chunk
            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }

            // For chunks after the first, check for and remove duplicated token sequences
            if segmentIndex > 0 && !allTokens.isEmpty && !windowTokens.isEmpty {
                let (deduped, removedCount) = manager.removeDuplicateTokenSequence(
                    previous: allTokens, current: windowTokens)
                let adjustedTimestamps = Array(windowTimestamps.dropFirst(removedCount))

                allTokens.append(contentsOf: deduped)
                allTimestamps.append(contentsOf: adjustedTimestamps)
            } else {
                allTokens.append(contentsOf: windowTokens)
                allTimestamps.append(contentsOf: windowTimestamps)
            }
            centerStart += centerSamples
            segmentIndex += 1
        }
        return manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    private func processWindowWithTokens(
        centerStart: Int,
        segmentIndex: Int,
        lastProcessedFrame: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], maxFrame: Int) {
        let remainingSamples = audioSamples.count - centerStart

        // Calculate adaptive context and frame adjustment for last chunk
        let adaptiveLeftContextSamples: Int
        let contextFrameAdjustment: Int

        if isLastChunk && remainingSamples < centerSamples {
            // Last chunk can't fill center - maximize context usage
            // Try to use full model capacity (15s) if available
            let desiredTotalSamples = min(maxModelSamples, audioSamples.count)
            let maxLeftContext = centerStart  // Can't go before start

            // Calculate how much left context we need
            let neededLeftContext = desiredTotalSamples - remainingSamples
            adaptiveLeftContextSamples = min(neededLeftContext, maxLeftContext)

            // CRITICAL: Calculate frame adjustment for timeJump
            // If we're pulling in more left context than usual, we need to adjust timeJump
            // to account for the extra frames that were already processed
            if adaptiveLeftContextSamples > leftContextSamples {
                // Extra context beyond standard = frames to adjust timeJump by
                let extraContextSamples = adaptiveLeftContextSamples - leftContextSamples
                // Convert samples to encoder frames (1 frame = 0.08s = 1280 samples at 16kHz)
                contextFrameAdjustment = extraContextSamples / ASRConstants.samplesPerEncoderFrame
            } else {
                contextFrameAdjustment = 0
            }

            if enableDebug {
                logger.debug(
                    """
                    Last chunk adaptive context:
                    - Remaining: \(remainingSamples) samples (\(String(format: "%.2f", Double(remainingSamples)/16000.0))s)
                    - Adaptive left context: \(adaptiveLeftContextSamples) samples (\(String(format: "%.2f", Double(adaptiveLeftContextSamples)/16000.0))s)
                    - Context frame adjustment: \(contextFrameAdjustment) frames (adjusting timeJump for \(contextFrameAdjustment * ASRConstants.samplesPerEncoderFrame) samples)
                    - Total chunk: \(adaptiveLeftContextSamples + remainingSamples) samples
                    """)
            }
        } else {
            // Standard context for non-last chunks
            adaptiveLeftContextSamples = leftContextSamples
            contextFrameAdjustment = 0
        }

        // Compute window bounds in samples: [leftStart, rightEnd)
        let leftStart = max(0, centerStart - adaptiveLeftContextSamples)
        let centerEnd = min(audioSamples.count, centerStart + centerSamples)
        let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

        // If nothing to process, return empty
        if leftStart >= rightEnd {
            return ([], [], 0)
        }

        let chunkSamples = Array(audioSamples[leftStart..<rightEnd])

        // Pad to model capacity (15s) if needed; keep track of actual chunk length
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: maxModelSamples)

        // Calculate actual encoder frames from unpadded chunk samples using shared constants
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: chunkSamples.count)

        let (tokens, timestamps, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            actualAudioFrames: actualFrameCount,
            enableDebug: enableDebug,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk
        )

        if tokens.isEmpty || encLen == 0 {
            return ([], [], 0)
        }

        // Take all tokens from decoder (it already processed only the relevant frames)
        let filteredTokens = tokens
        let filteredTimestamps = timestamps
        let maxFrame = timestamps.max() ?? 0

        return (filteredTokens, filteredTimestamps, maxFrame)
    }
}
