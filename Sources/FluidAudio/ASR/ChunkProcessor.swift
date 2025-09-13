import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]

    private let logger = AppLogger(category: "ChunkProcessor")

    // Frame-aligned configuration: 11.2 + 1.6 + 1.6 seconds context at 16kHz
    // 11.2s center = exactly 140 encoder frames
    // 1.6s context = exactly 20 encoder frames each
    // Total: 14.4s (within 15s model limit, 180 total frames)
    private let sampleRate: Int = 16000
    private let centerSeconds: Double = 11.2  // Reduced to allow for more overlap
    private let leftContextSeconds: Double = 1.6  // Increased overlap to 30 frames to avoid missing speech
    private let rightContextSeconds: Double = 1.6  // Exactly 20 frames (1.6 * 12.5)

    private var centerSamples: Int { Int(centerSeconds * Double(sampleRate)) }
    private var leftContextSamples: Int { Int(leftContextSeconds * Double(sampleRate)) }
    private var rightContextSamples: Int { Int(rightContextSeconds * Double(sampleRate)) }
    private var maxModelSamples: Int { 240_000 }  // 15 seconds window capacity

    func process(
        using manager: AsrManager, decoderState: inout TdtDecoderState, startTime: Date
    ) async throws -> ASRResult {
        // Use a combined structure to keep tokens, timestamps, and confidences aligned
        var allTokenData: [(token: Int, timestamp: Int, confidence: Float)] = []

        var centerStart = 0
        var segmentIndex = 0
        var lastProcessedFrame = 0  // Track the last frame processed by previous chunk
        while centerStart < audioSamples.count {
            // Determine if this is the last chunk
            let isLastChunk = (centerStart + centerSamples) >= audioSamples.count

            // Process chunk with explicit last chunk detection

            let (windowTokens, windowTimestamps, windowConfidences, maxFrame) = try await processWindowWithTokens(
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

            // Combine tokens, timestamps, and confidences into aligned tuples
            guard windowTokens.count == windowTimestamps.count && windowTokens.count == windowConfidences.count else {
                throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
            }

            let windowData = zip(zip(windowTokens, windowTimestamps), windowConfidences).map {
                (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
            }

            // For chunks after the first, check for and remove duplicated token sequences
            if segmentIndex > 0 && !allTokenData.isEmpty && !windowData.isEmpty {
                let previousTokens = allTokenData.map { $0.token }
                let currentTokens = windowData.map { $0.token }

                let (_, removedCount) = manager.removeDuplicateTokenSequence(
                    previous: previousTokens, current: currentTokens, maxOverlap: 30)
                // Only keep the non-duplicate portion of window data
                let adjustedWindowData = Array(windowData.dropFirst(removedCount))
                allTokenData.append(contentsOf: adjustedWindowData)
            } else {
                allTokenData.append(contentsOf: windowData)
            }

            centerStart += centerSamples

            segmentIndex += 1
        }

        // Sort by timestamp to ensure chronological order
        allTokenData.sort { $0.timestamp < $1.timestamp }

        // Extract sorted arrays
        let allTokens = allTokenData.map { $0.token }
        let allTimestamps = allTokenData.map { $0.timestamp }
        let allConfidences = allTokenData.map { $0.confidence }

        return manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
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
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], maxFrame: Int) {
        let remainingSamples = audioSamples.count - centerStart

        // Calculate context and frame adjustment for all chunks
        let adaptiveLeftContextSamples: Int
        var contextFrameAdjustment: Int

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

            // CRITICAL: For last chunks, handle overlap carefully based on where previous chunk ended
            // The goal is to continue from where we left off while allowing deduplication to work

            if segmentIndex > 0 && lastProcessedFrame > 0 {
                // Calculate where this chunk starts in global frame space
                let chunkLeftStart = max(0, centerStart - adaptiveLeftContextSamples)
                let chunkStartFrame = chunkLeftStart / ASRConstants.samplesPerEncoderFrame

                // Calculate the theoretical overlap
                let theoreticalOverlap = lastProcessedFrame - chunkStartFrame

                if theoreticalOverlap > 0 {
                    // For last chunk, be more conservative with overlap to avoid missing content
                    // Use smaller buffer (15 frames instead of 25) to ensure we don't skip too much
                    contextFrameAdjustment = max(0, theoreticalOverlap - 15)
                } else {
                    // No overlap or gap - use minimal overlap for continuity
                    contextFrameAdjustment = 5  // 0.4s minimal overlap
                }
            } else {
                // First chunk - no adjustment needed
                contextFrameAdjustment = 0
            }

        } else {
            // Standard non-first, non-last chunk
            adaptiveLeftContextSamples = leftContextSamples

            // Standard chunks use physical overlap in audio windows for context
            // Don't skip frames - let the decoder handle continuity with its timeJump mechanism
            contextFrameAdjustment = 0
        }

        // Compute window bounds in samples: [leftStart, rightEnd)
        let leftStart = max(0, centerStart - adaptiveLeftContextSamples)
        let centerEnd = min(audioSamples.count, centerStart + centerSamples)
        let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

        // If nothing to process, return empty
        if leftStart >= rightEnd {
            return ([], [], [], 0)
        }

        let chunkSamples = Array(audioSamples[leftStart..<rightEnd])

        // Pad to model capacity (15s) if needed; keep track of actual chunk length
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: maxModelSamples)

        // Calculate actual encoder frames from unpadded chunk samples using shared constants
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: chunkSamples.count)

        // Calculate global frame offset for this chunk
        let globalFrameOffset = leftStart / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            actualAudioFrames: actualFrameCount,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        if hypothesis.isEmpty || encLen == 0 {
            return ([], [], [], 0)
        }

        // Take all tokens from decoder (it already processed only the relevant frames)
        let filteredTokens = hypothesis.ySequence
        let filteredTimestamps = hypothesis.timestamps
        let filteredConfidences = hypothesis.tokenConfidences
        let maxFrame = hypothesis.maxTimestamp

        return (filteredTokens, filteredTimestamps, filteredConfidences, maxFrame)
    }
}
