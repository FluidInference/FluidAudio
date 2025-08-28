import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]
    let enableDebug: Bool

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ChunkProcessor")

    // Total: 14.4s (within 15s model limit, 180 total frames)
    private let sampleRate: Int = 16000
    private let centerSeconds: Double = 8.0
    private let leftContextSeconds: Double = 4.8
    private let rightContextSeconds: Double = 1.6
    // Encoder subsampling: exactly 1280 samples = 1 encoder frame (80ms at 16kHz)
    // This avoids floating-point drift over time
    private let samplesPerEncoderFrame = 1280

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

            // Calculate global frame offset based on the left edge of the processing window
            // This represents where we actually start processing (not the center)
            let leftStart = max(0, centerStart - leftContextSamples)
            let globalFrameOffset = leftStart / samplesPerEncoderFrame

            // Process chunk with explicit last chunk detection
            let (windowTokens, windowTimestamps, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset,
                using: manager,
                decoderState: &decoderState
            )

            // Update last processed frame for next chunk
            // maxFrame is already a global frame index, keep it as-is
            // The TDT decoder will convert it to local coordinates when needed
            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }

            let rightContextFrames = rightContextSamples / samplesPerEncoderFrame

            print(
                "Window \(segmentIndex): centerStart=\(centerStart), globalFrameOffset=\(globalFrameOffset), lastProcessedFrame=\(lastProcessedFrame)"
            )
            print("Window Tokens: \(windowTokens)")
            print("Window Timestamps: \(windowTimestamps)")
            print("Right Context Frames: \(rightContextFrames)")
            print("Max Frame: \(maxFrame)")

            // For chunks after the first, use timestamp-based merging to avoid overlaps
            if segmentIndex > 0 {
                let (mergedTokens, mergedTimestamps) = manager.mergeChunksByTimestamp(
                    previousTokens: allTokens,
                    previousTimestamps: allTimestamps,
                    currentTokens: windowTokens,
                    currentTimestamps: windowTimestamps,
                    centerStart: centerStart,
                    rightContextFrames: rightContextFrames,
                    isFirstChunk: segmentIndex == 0,
                    globalFrameOffset: globalFrameOffset
                )
                allTokens = mergedTokens
                allTimestamps = mergedTimestamps
            } else {
                // For first chunk, still apply right context filtering
                let (filteredTokens, filteredTimestamps) = manager.mergeChunksByTimestamp(
                    previousTokens: [],
                    previousTimestamps: [],
                    currentTokens: windowTokens,
                    currentTimestamps: windowTimestamps,
                    centerStart: centerStart,
                    rightContextFrames: rightContextFrames,
                    isFirstChunk: true,
                    globalFrameOffset: globalFrameOffset
                )
                allTokens.append(contentsOf: filteredTokens)
                allTimestamps.append(contentsOf: filteredTimestamps)
            }
            // Advance by center window minus right context to ensure overlap coverage
            // This ensures frames in the right context of current chunk become left context of next chunk
            let advanceAmount = centerSamples - rightContextSamples
            centerStart += advanceAmount
            segmentIndex += 1
        }

        print("Final Tokens: \(allTokens)")
        print("Final Timestamps: \(allTimestamps)")

        let finalResult = manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )

        return ASRResult(
            text: finalResult.text,
            confidence: finalResult.confidence,
            duration: finalResult.duration,
            processingTime: finalResult.processingTime,
            tokenTimings: finalResult.tokenTimings,
            performanceMetrics: finalResult.performanceMetrics
        )
    }

    private func processWindowWithTokens(
        centerStart: Int,
        segmentIndex: Int,
        lastProcessedFrame: Int,
        isLastChunk: Bool,
        globalFrameOffset: Int,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], maxFrame: Int) {
        // Use standard context for all chunks - the TdtDecoder handles last chunk finalization
        let adaptiveLeftContextSamples = leftContextSamples

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

        // For chunk processing, don't use startFrameOffset - rely on lastProcessedFrame instead
        // startFrameOffset is designed for different use cases and conflicts with lastProcessedFrame logic
        let startFrameOffset = 0

        let (tokens, timestamps, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            enableDebug: false,
            decoderState: &decoderState,
            startFrameOffset: startFrameOffset,
            lastProcessedFrame: lastProcessedFrame,
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
