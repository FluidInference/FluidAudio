import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]
    let enableDebug: Bool

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ChunkProcessor")

    // Frame-aligned configuration: 11.2 + 1.6 + 1.6 seconds context at 16kHz
    // 11.2s center = exactly 140 encoder frames
    // 1.6s context = exactly 20 encoder frames each
    // Total: 14.4s (within 15s model limit, 180 total frames)
    private let sampleRate: Int = 16000
    private let centerSeconds: Double = 11.2  // Exactly 140 frames (11.2 * 12.5)
    private let leftContextSeconds: Double = 1.6  // Exactly 20 frames (1.6 * 12.5)
    private let rightContextSeconds: Double = 1.6  // Exactly 20 frames (1.6 * 12.5)

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

            print("[Segment \(segmentIndex)] Processed \(windowTokens.count) tokens, maxFrame = \(maxFrame)")
            print("[Segment \(segmentIndex)] Total tokens so far: \(allTokens.count)")
            print("[Segment \(segmentIndex)] lastProcessedFrame = \(lastProcessedFrame) frames")
            print("[Segment \(segmentIndex)] Center Processed = \(centerStart) with \(centerSamples) processed")
            print("\t[Segment \(segmentIndex)] Window Tokens: \(windowTokens)")
            print("\t[Segment \(segmentIndex)] Window Timestamps: \(windowTimestamps)")

            centerStart += centerSamples

            segmentIndex += 1
        }
        // Validation: Ensure all audio frames were accounted for
        let expectedTotalFrames = ASRConstants.calculateEncoderFrames(from: audioSamples.count)
        let processedCenterFrames =
            segmentIndex * Int(centerSeconds * Double(sampleRate)) / ASRConstants.samplesPerEncoderFrame

        if enableDebug {
            print("📊 FINAL VALIDATION:")
            print("  - Total audio samples: \(audioSamples.count)")
            print("  - Expected total frames: \(expectedTotalFrames)")
            print("  - Processed center frames: \(processedCenterFrames)")
            print("  - Total segments: \(segmentIndex)")
            print("  - Final tokens count: \(allTokens.count)")
            print("  - Final timestamps count: \(allTimestamps.count)")
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

        // Calculate context and frame adjustment for all chunks
        let adaptiveLeftContextSamples: Int
        let contextFrameAdjustment: Int

        if segmentIndex == 0 {
            // First chunk: no overlap, standard context
            adaptiveLeftContextSamples = leftContextSamples
            contextFrameAdjustment = 0
        } else if isLastChunk && remainingSamples < centerSamples {
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
                // Convert samples to encoder frames with proper rounding (1 frame = 0.08s = 1280 samples at 16kHz)
                // Use floating-point division and rounding to avoid losing partial frames
                contextFrameAdjustment = Int(
                    (Double(extraContextSamples) / Double(ASRConstants.samplesPerEncoderFrame)).rounded())
            } else {
                // Standard left context for last chunk - need to skip overlap
                let standardOverlapSamples = leftContextSamples  // 1.6s
                contextFrameAdjustment = -Int(
                    (Double(standardOverlapSamples) / Double(ASRConstants.samplesPerEncoderFrame)).rounded())
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
            // Standard non-first, non-last chunk
            adaptiveLeftContextSamples = leftContextSamples

            // Standard chunks have 1.6s left context that overlaps with previous chunk
            // Need NEGATIVE adjustment to skip already-processed frames
            let standardOverlapSamples = leftContextSamples  // 1.6s = 25,600 samples
            contextFrameAdjustment = -Int(
                (Double(standardOverlapSamples) / Double(ASRConstants.samplesPerEncoderFrame)).rounded())

            if enableDebug {
                logger.debug(
                    """
                    Standard chunk overlap handling:
                    - Left context: \(leftContextSamples) samples (\(String(format: "%.2f", Double(leftContextSamples)/16000.0))s)
                    - Context frame adjustment: \(contextFrameAdjustment) frames (skip overlap)
                    - Total chunk: \(adaptiveLeftContextSamples + centerSamples + rightContextSamples) samples
                    """)
            }
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

        // Calculate global frame offset for this chunk
        let globalFrameOffset = leftStart / ASRConstants.samplesPerEncoderFrame

        if enableDebug {
            print("🔢 CHUNK FRAME CALC [Segment \(segmentIndex)]:")
            print("  - centerStart: \(centerStart) samples")
            print("  - leftStart: \(leftStart) samples")
            print("  - chunkSamples.count: \(chunkSamples.count) samples")
            print("  - actualFrameCount: \(actualFrameCount) frames")
            print("  - globalFrameOffset: \(globalFrameOffset) frames")
        }

        let (tokens, timestamps, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            actualAudioFrames: actualFrameCount,
            enableDebug: enableDebug,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
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
