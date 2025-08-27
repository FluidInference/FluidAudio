import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]
    let enableDebug: Bool

    // 10 + 2 + 2 seconds context at 16kHz
    private let sampleRate: Int = 16000
    private let centerSeconds: Double = 11.0
    private let leftContextSeconds: Double = 2.0
    private let rightContextSeconds: Double = 2.0

    private var centerSamples: Int { Int(centerSeconds * Double(sampleRate)) }
    private var leftContextSamples: Int { Int(leftContextSeconds * Double(sampleRate)) }
    private var rightContextSamples: Int { Int(rightContextSeconds * Double(sampleRate)) }
    private var maxModelSamples: Int { 240_000 }  // 15 seconds window capacity

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "ChunkProcessor")

    func process(
        using manager: AsrManager, decoderState: inout TdtDecoderState, startTime: Date
    ) async throws -> ASRResult {
        var allTokens: [Int] = []
        var allTimestamps: [Int] = []

        var centerStart = 0
        var segmentIndex = 0
        var lastProcessedFrame = 0  // Track the last frame processed by previous chunk

        while centerStart < audioSamples.count {
            let (windowTokens, windowTimestamps, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                using: manager,
                decoderState: &decoderState
            )

            // Debug logging: Show tokens from this chunk
            if enableDebug {
                logger.debug(
                    "🔍 Chunk \(segmentIndex): centerStart=\(centerStart), produced \(windowTokens.count) tokens")
                if !windowTokens.isEmpty {
                    let tokenTexts = windowTokens.compactMap { tokenId in
                        manager.vocabulary[tokenId] ?? "token_\(tokenId)"
                    }
                    let chunkText = tokenTexts.joined().replacingOccurrences(of: "▁", with: " ").trimmingCharacters(
                        in: .whitespaces)
                    logger.debug("🔍 Chunk \(segmentIndex) text: '\(chunkText)'")
                    logger.debug(
                        "🔍 Chunk \(segmentIndex) timestamps: \(windowTimestamps.prefix(5))...\(windowTimestamps.suffix(5))"
                    )
                    logger.debug(
                        "🔍 Chunk \(segmentIndex) time range: \(windowTimestamps.min() ?? 0) - \(windowTimestamps.max() ?? 0)"
                    )
                }
            }

            // Update last processed frame for next chunk
            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }

            // For chunks after the first, check for and remove duplicated token sequences
            if segmentIndex > 0 && !allTokens.isEmpty && !windowTokens.isEmpty {
                if enableDebug {
                    logger.debug(
                        "🔍 Before deduplication: previous=\(allTokens.suffix(10)), current=\(windowTokens.prefix(10))")
                }

                let (deduped, removedCount) = manager.removeDuplicateTokenSequence(
                    previous: allTokens, current: windowTokens)
                let adjustedTimestamps = Array(windowTimestamps.dropFirst(removedCount))

                if enableDebug && removedCount > 0 {
                    logger.debug("🔍 Deduplication removed \(removedCount) tokens from chunk \(segmentIndex)")
                }

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
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], maxFrame: Int) {
        // Compute window bounds in samples: [leftStart, rightEnd)
        let leftStart = max(0, centerStart - leftContextSamples)
        let centerEnd = min(audioSamples.count, centerStart + centerSamples)
        let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

        // If nothing to process, return empty
        if leftStart >= rightEnd { return ([], [], 0) }

        let chunkSamples = Array(audioSamples[leftStart..<rightEnd])
        let chunkAudioDuration = Double(chunkSamples.count) / Double(sampleRate)

        // Pad to model capacity (15s) if needed; keep track of actual chunk length
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: maxModelSamples)

        // Calculate encoder frame offset based on where previous chunk ended
        let startFrameOffset = manager.calculateStartFrameOffset(
            segmentIndex: segmentIndex,
            leftContextSeconds: leftContextSeconds
        )

        if enableDebug {
            logger.debug(
                "🔍 Processing chunk \(segmentIndex): samples [\(leftStart)..\(rightEnd-1)] = \(chunkSamples.count) samples (\(String(format: "%.2f", chunkAudioDuration))s)"
            )
            logger.debug("🔍 startFrameOffset=\(startFrameOffset), lastProcessedFrame=\(lastProcessedFrame)")
        }

        let (tokens, timestamps, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            enableDebug: enableDebug,
            decoderState: &decoderState,
            startFrameOffset: startFrameOffset,
            lastProcessedFrame: lastProcessedFrame
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
