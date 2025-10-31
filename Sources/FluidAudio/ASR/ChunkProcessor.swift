import CoreML
import Foundation
import OSLog

struct ChunkProcessor {
    let audioSamples: [Float]

    private let logger = AppLogger(category: "ChunkProcessor")

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (239,360 samples) to stay under encoder limit
    // - 2.0s overlap (32,000 samples) to give the decoder slack when merging windows
    private let sampleRate: Int = 16000
    private let overlapSeconds: Double = 2.0

    private var maxModelSamples: Int { 240_000 }  // CoreML encoder capacity (15 seconds)
    private var chunkSamples: Int {
        // Keep chunk length aligned to encoder frames to maintain integer timestamp offsets
        let safeSamples = maxModelSamples - ASRConstants.melHopSize
        let frames = max(safeSamples / ASRConstants.samplesPerEncoderFrame, 1)
        return frames * ASRConstants.samplesPerEncoderFrame  // 239,360 samples (≈14.96s)
    }
    private var overlapSamples: Int {
        let requested = Int(overlapSeconds * Double(sampleRate))
        return min(requested, chunkSamples / 2)
    }
    private var strideSamples: Int {
        max(chunkSamples - overlapSamples, ASRConstants.samplesPerEncoderFrame)
    }

    func process(
        using manager: AsrManager, startTime: Date
    ) async throws -> ASRResult {
        // Use a combined structure to keep tokens, timestamps, and confidences aligned
        var allTokenData: [(token: Int, timestamp: Int, confidence: Float)] = []

        var chunkStart = 0
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make()

        while chunkStart < audioSamples.count {
            let candidateEnd = chunkStart + chunkSamples
            let isLastChunk = candidateEnd >= audioSamples.count
            let chunkEnd = isLastChunk ? audioSamples.count : candidateEnd

            if chunkEnd <= chunkStart {
                logger.warning("ChunkProcessor received empty chunk window, stopping at index \(chunkIndex)")
                break
            }

            chunkDecoderState.reset()

            let chunkRange = chunkStart..<chunkEnd
            let chunkSamplesSlice = Array(audioSamples[chunkRange])

            let (windowTokens, windowTimestamps, windowConfidences) = try await transcribeChunk(
                samples: chunkSamplesSlice,
                chunkStart: chunkStart,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &chunkDecoderState
            )

            // Combine tokens, timestamps, and confidences into aligned tuples
            guard windowTokens.count == windowTimestamps.count && windowTokens.count == windowConfidences.count else {
                throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
            }

            let windowData = zip(zip(windowTokens, windowTimestamps), windowConfidences).map {
                (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
            }

            // For chunks after the first, check for and remove duplicated token sequences
            if chunkIndex > 0 && !allTokenData.isEmpty && !windowData.isEmpty {
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

            chunkIndex += 1

            if isLastChunk {
                break
            }

            chunkStart += strideSamples
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

    private func transcribeChunk(
        samples: [Float],
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float]) {
        guard !samples.isEmpty else { return ([], [], []) }

        let paddedChunk = manager.padAudioIfNeeded(samples, targetLength: maxModelSamples)
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: samples.count)
        let globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,
            actualAudioFrames: actualFrameCount,
            decoderState: &decoderState,
            contextFrameAdjustment: 0,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            return ([], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences)
    }
}
