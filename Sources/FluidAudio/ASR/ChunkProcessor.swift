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

        let totalSamples = audioSamples.count
        let totalDuration = Double(totalSamples) / Double(sampleRate)

        // Verify perfect frame alignment
        let centerFrames = centerSeconds * 12.5  // Should be exactly 100.0
        let leftContextFrames = leftContextSeconds * 12.5  // Should be exactly 40.0
        let rightContextFrames = rightContextSeconds * 12.5  // Should be exactly 40.0
        let totalFrames = centerFrames + leftContextFrames + rightContextFrames  // Should be exactly 180.0

        logger.debug(
            "🎵 ChunkProcessor starting: totalSamples=\(totalSamples), duration=\(String(format: "%.2f", totalDuration))s"
        )
        logger.debug(
            "🎯 Frame alignment verification: center=\(centerFrames) frames, left=\(leftContextFrames) frames, right=\(rightContextFrames) frames, total=\(totalFrames) frames"
        )

        while centerStart < audioSamples.count {
            let centerStartTime = Double(centerStart) / Double(sampleRate)
            logger.debug(
                "📦 Processing chunk \(segmentIndex): centerStart=\(centerStart) (\(String(format: "%.2f", centerStartTime))s), lastProcessedFrame=\(lastProcessedFrame)"
            )

            let (windowTokens, windowTimestamps, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                using: manager,
                decoderState: &decoderState
            )

            // Update last processed frame for next chunk
            let previousLastFrame = lastProcessedFrame
            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }
            logger.debug(
                "📊 Chunk \(segmentIndex) results: \(windowTokens.count) tokens, timestamps: [\(windowTimestamps.min() ?? -1)...\(windowTimestamps.max() ?? -1)], maxFrame: \(previousLastFrame)→\(lastProcessedFrame)"
            )

            // For chunks after the first, check for and remove duplicated token sequences
            if segmentIndex > 0 && !allTokens.isEmpty && !windowTokens.isEmpty {
                let (deduped, removedCount) = manager.removeDuplicateTokenSequence(
                    previous: allTokens, current: windowTokens)
                let adjustedTimestamps = Array(windowTimestamps.dropFirst(removedCount))

                logger.debug(
                    "🔄 Deduplication: removed \(removedCount) duplicate tokens, keeping \(deduped.count) new tokens")
                allTokens.append(contentsOf: deduped)
                allTimestamps.append(contentsOf: adjustedTimestamps)
            } else {
                logger.debug("➕ First chunk or no overlap: adding all \(windowTokens.count) tokens")
                allTokens.append(contentsOf: windowTokens)
                allTimestamps.append(contentsOf: windowTimestamps)
            }
            centerStart += centerSamples
            segmentIndex += 1
        }

        logger.debug(
            "🏁 ChunkProcessor finished: processed \(segmentIndex) chunks, total tokens: \(allTokens.count), total duration: \(String(format: "%.2f", totalDuration))s"
        )

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
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], maxFrame: Int) {
        // Check if this is likely the final chunk and if it needs adaptive context
        let isLikelyFinalChunk = (centerStart + centerSamples) >= audioSamples.count
        let actualCenterSize = min(centerSamples, audioSamples.count - centerStart)
        let adaptiveLeftContextSamples: Int

        if isLikelyFinalChunk && actualCenterSize < Int(Double(centerSamples) * 0.7) {
            // For small final chunks, use adaptive left context to minimize padding
            // Calculate how much left context to use so total window has minimal padding
            let remainingAudio = audioSamples.count - centerStart
            let maxUsableContext = min(centerStart, maxModelSamples - remainingAudio - rightContextSamples)
            adaptiveLeftContextSamples = max(leftContextSamples, maxUsableContext)

            logger.debug(
                "🔄 ADAPTIVE CONTEXT: Final chunk detected, actualCenterSize=\(actualCenterSize), adaptiveLeftContext=\(Double(adaptiveLeftContextSamples)/Double(sampleRate))s (was \(Double(leftContextSamples)/Double(sampleRate))s)"
            )
        } else {
            adaptiveLeftContextSamples = leftContextSamples
        }

        // Compute window bounds in samples: [leftStart, rightEnd)
        let leftStart = max(0, centerStart - adaptiveLeftContextSamples)
        let centerEnd = min(audioSamples.count, centerStart + centerSamples)
        let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

        let leftStartTime = Double(leftStart) / Double(sampleRate)
        let centerStartTime = Double(centerStart) / Double(sampleRate)
        let centerEndTime = Double(centerEnd) / Double(sampleRate)
        let rightEndTime = Double(rightEnd) / Double(sampleRate)

        logger.debug(
            "📐 Window bounds (samples): left=\(leftStart), center=[\(centerStart)...\(centerEnd)], right=\(rightEnd)")
        logger.debug(
            "⏱️  Window bounds (time): left=\(String(format: "%.2f", leftStartTime))s, center=[\(String(format: "%.2f", centerStartTime))...\(String(format: "%.2f", centerEndTime))]s, right=\(String(format: "%.2f", rightEndTime))s"
        )

        // If nothing to process, return empty
        if leftStart >= rightEnd {
            logger.debug("⚠️ Empty window: leftStart(\(leftStart)) >= rightEnd(\(rightEnd))")
            return ([], [], 0)
        }

        let chunkSamples = Array(audioSamples[leftStart..<rightEnd])
        let chunkAudioDuration = Double(chunkSamples.count) / Double(sampleRate)

        // Pad to model capacity (15s) if needed; keep track of actual chunk length
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: maxModelSamples)
        let paddingAdded = paddedChunk.count - chunkSamples.count
        logger.debug(
            "🎵 Audio chunk: original=\(chunkSamples.count) samples (\(String(format: "%.2f", chunkAudioDuration))s), padded=\(paddedChunk.count) (+\(paddingAdded) padding)"
        )

        // Calculate encoder frame offset based on where previous chunk ended
        // Use adaptive context duration for proper frame offset calculation
        let actualLeftContextSeconds = Double(adaptiveLeftContextSamples) / Double(sampleRate)
        let startFrameOffset = manager.calculateStartFrameOffset(
            segmentIndex: segmentIndex,
            leftContextSeconds: actualLeftContextSeconds
        )

        // Calculate expected encoder frames for debugging
        let expectedEncoderFrames = Int(Double(chunkSamples.count) / 1280.0)  // 16kHz / 12.5 fps = 1280 samples per frame
        logger.debug(
            "🔢 Frame calculations: startFrameOffset=\(startFrameOffset), expectedEncoderFrames=\(expectedEncoderFrames), lastProcessedFrame=\(lastProcessedFrame)"
        )

        let (tokens, timestamps, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            enableDebug: false,
            decoderState: &decoderState,
            startFrameOffset: startFrameOffset,
            lastProcessedFrame: lastProcessedFrame
        )

        logger.debug("🧠 ML Inference results: encLen=\(encLen), tokens=\(tokens.count), timestamps=\(timestamps.count)")

        if tokens.isEmpty || encLen == 0 {
            logger.debug("⚠️ Empty inference result: tokens.isEmpty=\(tokens.isEmpty), encLen=\(encLen)")
            return ([], [], 0)
        }

        // Take all tokens from decoder (it already processed only the relevant frames)
        let filteredTokens = tokens
        let filteredTimestamps = timestamps
        let maxFrame = timestamps.max() ?? 0

        // Check if we have any frame gaps or overlaps
        if !timestamps.isEmpty {
            let minFrame = timestamps.min() ?? 0
            logger.debug(
                "🎯 Frame analysis: range=[\(minFrame)...\(maxFrame)], expectedFrames=\(expectedEncoderFrames), actualFrames=\(encLen)"
            )

            // Check for potential frame calculation issues
            if maxFrame > encLen * 2 {
                logger.debug(
                    "⚠️ WARNING: maxFrame (\(maxFrame)) >> encLen (\(encLen)) - potential frame calculation issue")
            }
            if minFrame < 0 {
                logger.debug("⚠️ WARNING: negative frame detected: minFrame=\(minFrame)")
            }
        }

        return (filteredTokens, filteredTimestamps, maxFrame)
    }
}
