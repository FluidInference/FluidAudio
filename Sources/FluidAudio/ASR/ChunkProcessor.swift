import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]
    let enableDebug: Bool

    // 10 + 2 + 2 seconds context at 16kHz
    private let sampleRate: Int = 16000
    private let centerSeconds: Double = 10.0
    private let leftContextSeconds: Double = 2.0
    private let rightContextSeconds: Double = 2.0

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
            let (windowTokens, windowTimestamps, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                using: manager,
                decoderState: &decoderState
            )

            // Update last processed frame for next chunk
            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }

            // For chunks after the first, check for and remove duplicated token sequences
            if segmentIndex > 0 && !allTokens.isEmpty && !windowTokens.isEmpty {
                let adapter = SlidingWindowAdapter(
                    sampleRate: sampleRate,
                    centerSeconds: centerSeconds,
                    leftContextSeconds: leftContextSeconds,
                    rightContextSeconds: rightContextSeconds,
                    enableDebug: enableDebug
                )
                let (deduped, removedCount) = adapter.removeDuplicateSequence(previous: allTokens, current: windowTokens)
                let adjustedTimestamps = Array(windowTimestamps.dropFirst(removedCount))

                allTokens.append(contentsOf: deduped)
                allTimestamps.append(contentsOf: adjustedTimestamps)

                if enableDebug && removedCount > 0 {
                    print("CHUNK \(segmentIndex): removed \(removedCount) duplicate tokens")
                }
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
        let adapter = SlidingWindowAdapter(
            sampleRate: sampleRate,
            centerSeconds: centerSeconds,
            leftContextSeconds: leftContextSeconds,
            rightContextSeconds: rightContextSeconds,
            enableDebug: enableDebug
        )
        let startFrameOffset: Int = adapter.startFrameOffset(forSegment: segmentIndex)
        if segmentIndex > 0 {
            print(
                "CHUNK \(segmentIndex): lastProcessedFrame=\(lastProcessedFrame), startFrameOffset=\(startFrameOffset)"
            )
        }

        print(
            "CHUNK \(segmentIndex): processing \(String(format: "%.2f", chunkAudioDuration))s audio, leftStart=\(leftStart), centerStart=\(centerStart), rightEnd=\(rightEnd)"
        )
        print(
            "CHUNK \(segmentIndex): startFrameOffset=\(startFrameOffset), skipping \(startFrameOffset) encoder frames"
        )

        let (tokens, timestamps, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: chunkSamples.count,
            enableDebug: false,
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

        print(
            "CHUNK \(segmentIndex): audio_duration=\(String(format: "%.2f", chunkAudioDuration))s, startFrameOffset=\(startFrameOffset), taking all \(tokens.count) tokens"
        )

        return (filteredTokens, filteredTimestamps, maxFrame)
    }
}
