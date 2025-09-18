import CoreML
import Foundation
import OSLog

struct ChunkProcessingResult {
    let tokens: [Int]
    let timestamps: [Int]
    let confidences: [Float]
}

struct ChunkProcessingEngine {
    let audioSamples: [Float]
    let startSampleInOriginalAudio: Int

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

    func run(
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> ChunkProcessingResult {
        var allTokenData: [(token: Int, timestamp: Int, confidence: Float)] = []

        if audioSamples.isEmpty {
            logger.debug("Chunk processing skipped: no audio samples available")
            return ChunkProcessingResult(tokens: [], timestamps: [], confidences: [])
        }

        var centerStart = 0
        var segmentIndex = 0
        var lastProcessedFrame = 0  // Track the last frame processed by previous chunk

        while centerStart < audioSamples.count {
            // Determine if this is the last chunk
            let isLastChunk = (centerStart + centerSamples + rightContextSamples) >= audioSamples.count

            let (windowTokens, windowTimestamps, windowConfidences, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &decoderState
            )

            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }

            guard windowTokens.count == windowTimestamps.count && windowTokens.count == windowConfidences.count else {
                throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
            }

            let windowData = zip(zip(windowTokens, windowTimestamps), windowConfidences).map {
                (token: $0.0.0, timestamp: $0.0.1, confidence: $0.1)
            }

            if segmentIndex > 0 && !allTokenData.isEmpty && !windowData.isEmpty {
                let previousTokens = allTokenData.map { $0.token }
                let currentTokens = windowData.map { $0.token }

                let (_, removedCount) = manager.removeDuplicateTokenSequence(
                    previous: previousTokens, current: currentTokens, maxOverlap: 30)
                let adjustedWindowData = Array(windowData.dropFirst(removedCount))
                allTokenData.append(contentsOf: adjustedWindowData)
            } else {
                allTokenData.append(contentsOf: windowData)
            }

            centerStart += centerSamples

            segmentIndex += 1

            if isLastChunk {
                break
            }
        }

        allTokenData.sort { $0.timestamp < $1.timestamp }

        return ChunkProcessingResult(
            tokens: allTokenData.map { $0.token },
            timestamps: allTokenData.map { $0.timestamp },
            confidences: allTokenData.map { $0.confidence }
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

            if segmentIndex > 0 && lastProcessedFrame > 0 {
                let chunkLeftStart = max(0, centerStart - adaptiveLeftContextSamples)
                let absoluteChunkLeftStart = startSampleInOriginalAudio + chunkLeftStart
                let chunkStartFrame = absoluteChunkLeftStart / ASRConstants.samplesPerEncoderFrame

                let theoreticalOverlap = lastProcessedFrame - chunkStartFrame

                if theoreticalOverlap > 0 {
                    contextFrameAdjustment = max(0, theoreticalOverlap - 15)
                } else {
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
        let effectiveLength = calculateEffectiveLength(chunkSamples)

        // Calculate global frame offset for this chunk using absolute position
        let absoluteLeftStart = startSampleInOriginalAudio + leftStart
        let globalFrameOffset = absoluteLeftStart / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encLen) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: effectiveLength,
            actualAudioFrames: nil,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        if hypothesis.isEmpty || encLen == 0 {
            logger.debug(
                "Chunk window \(segmentIndex) returned empty tokens (centerStart=\(centerStart), effectiveLength=\(effectiveLength), contextAdjust=\(contextFrameAdjustment), encoderLength=\(encLen))"
            )
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

extension ChunkProcessingEngine {
    private func calculateEffectiveLength(_ samples: [Float]) -> Int {
        guard !samples.isEmpty else { return 0 }

        let silenceThreshold: Float = 1e-4
        let trailingBufferSamples = ASRConstants.samplesPerEncoderFrame / 2

        // Track the loudest absolute sample so we can distinguish true silence from
        // extremely quiet utterances (e.g. low-volume or TTS-normalized data).
        var maxAbsoluteSample: Float = 0
        for sample in samples {
            let magnitude = abs(sample)
            if magnitude > maxAbsoluteSample {
                maxAbsoluteSample = magnitude
            }
        }

        if maxAbsoluteSample <= Float.ulpOfOne {
            // Pure or near-pure silence: keep a tiny buffer so downstream frame math stays valid.
            return min(max(trailingBufferSamples, 1), samples.count)
        }

        if maxAbsoluteSample < silenceThreshold {
            // Legitimate speech that was recorded at a very low amplitude.
            // Treat the entire chunk as effective audio so we don't collapse it to a single frame.
            return samples.count
        }

        var lastNonSilenceIndex = samples.count - 1
        while lastNonSilenceIndex > 0 && abs(samples[lastNonSilenceIndex]) < silenceThreshold {
            lastNonSilenceIndex -= 1
        }

        let effectiveLength = min(samples.count, lastNonSilenceIndex + 1 + trailingBufferSamples)
        return max(effectiveLength, 1)
    }
}

struct ChunkProcessor {
    let audioSamples: [Float]
    private let startSampleInOriginalAudio: Int

    init(audioSamples: [Float], startSampleInOriginalAudio: Int = 0) {
        self.audioSamples = audioSamples
        self.startSampleInOriginalAudio = startSampleInOriginalAudio
    }

    func process(
        using manager: AsrManager,
        decoderState: inout TdtDecoderState,
        startTime: Date
    ) async throws -> ASRResult {
        let engine = ChunkProcessingEngine(
            audioSamples: audioSamples,
            startSampleInOriginalAudio: startSampleInOriginalAudio
        )

        let chunkResult = try await engine.run(using: manager, decoderState: &decoderState)

        return manager.processTranscriptionResult(
            tokenIds: chunkResult.tokens,
            timestamps: chunkResult.timestamps,
            confidences: chunkResult.confidences,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: audioSamples,
            processingTime: Date().timeIntervalSince(startTime)
        )
    }
}
