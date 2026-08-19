import Foundation

extension AsrManager {

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout TdtDecoderState, language: Language? = nil
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard audioSamples.count >= minimumRequiredSamples else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Route to appropriate processing method based on audio length
        if audioSamples.count <= ASRConstants.maxModelSamples {
            let (alignedSamples, frameAlignedLength) = frameAlignedAudio(audioSamples)
            let paddedAudio: [Float] = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
            let (hypothesis, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: frameAlignedLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState,
                isLastChunk: true,  // Single-chunk: always first and last
                language: language
            )

            let result = processTranscriptionResult(
                tokenIds: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                tokenDurations: hypothesis.tokenDurations,
                encoderSequenceLength: encoderSequenceLength,
                audioSampleCount: audioSamples.count,
                processingTime: Date().timeIntervalSince(startTime)
            )

            return result
        }

        // ChunkProcessor handles stateless chunked transcription for long audio
        let processor = ChunkProcessor(audioSamples: audioSamples)
        let result = try await processor.process(
            using: self,
            startTime: startTime,
            progressHandler: { [weak self] progress in
                guard let self else { return }
                await self.progressEmitter.report(progress: progress)
            },
            language: language
        )

        return result
    }

    /// Cross-window emission jitter allowance for the final-window re-decode: a
    /// re-decoded token can land a few frames from its original emission, so the
    /// suppression cutoff backs off this much and dedup strips what remains.
    internal static let redecodeEmissionJitterFrames = 5

    /// Decoder-entry plan for the final streaming window (issue #855).
    ///
    /// Returns `initialTimeIndexOverride: 0` so the decoder re-decodes the window
    /// from frame 0 with its carried state (a mid-window entry into a short flush
    /// window can blank out the trailing speech), plus an emission cutoff in
    /// window-local frames: tokens for audio the previous windows already emitted
    /// are suppressed at the source, leaving dedup only the jitter margin.
    /// Non-final windows and callers without accumulated timestamps get `(nil, nil)`
    /// — the legacy navigation.
    nonisolated internal static func lastChunkRedecodePlan(
        isLastChunk: Bool,
        previousTokens: [Int],
        previousTokenTimestamps: [Int]?,
        globalFrameOffset: Int
    ) -> (initialTimeIndexOverride: Int?, emitTokensAfterFrame: Int?) {
        guard isLastChunk, let previousTimestamps = previousTokenTimestamps, !previousTokens.isEmpty else {
            return (nil, nil)
        }
        let lastEmittedGlobalFrame = previousTimestamps.max() ?? 0
        let cutoff = max(0, lastEmittedGlobalFrame - globalFrameOffset - redecodeEmissionJitterFrames)
        return (0, cutoff)
    }

    /// Chunk transcription that preserves decoder state between calls.
    /// Used by SlidingWindowAsrManager for overlapping-window processing with token deduplication.
    func transcribeChunk(
        _ chunkSamples: [Float],
        decoderState: inout TdtDecoderState,
        previousTokens: [Int] = [],
        previousTokenTimestamps: [Int]? = nil,
        globalFrameOffset: Int = 0,
        isLastChunk: Bool = false,
        language: Language? = nil
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], encoderSequenceLength: Int) {
        let (alignedSamples, frameAlignedLength) = frameAlignedAudio(
            chunkSamples, allowAlignment: previousTokens.isEmpty)
        let padded = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        // Last streaming window: decode from frame 0 instead of skipping the overlap.
        // Jumping mid-window into a short flush window can blank out the trailing
        // speech entirely (issue #855: the joint emits a boundary punctuation, then
        // blanks to the end, dropping the final words). Re-decoding the overlap with
        // the carried state is robust; emissions for audio the previous windows
        // already covered are suppressed at the source (the decoder still updates
        // its LSTM state through them), so dedup only sees the few-frame jitter
        // margin — a token-dense overlap cannot outgrow dedup's bounded search.
        let redecodePlan = Self.lastChunkRedecodePlan(
            isLastChunk: isLastChunk,
            previousTokens: previousTokens,
            previousTokenTimestamps: previousTokenTimestamps,
            globalFrameOffset: globalFrameOffset
        )
        let (hypothesis, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: frameAlignedLength,
            actualAudioFrames: nil,  // Will be calculated from originalLength
            decoderState: &decoderState,
            contextFrameAdjustment: 0,  // Non-streaming chunks don't use adaptive context
            isLastChunk: isLastChunk,
            language: language,
            emitTokensAfterGlobalFrame: redecodePlan.emitTokensAfterFrame,
            initialTimeIndexOverride: redecodePlan.initialTimeIndexOverride
        )

        // Apply token deduplication if previous tokens are provided
        if !previousTokens.isEmpty && hypothesis.hasTokens {
            // Convert this chunk's local frame timestamps into the same global frame
            // space as `previousTokenTimestamps` so dedup can require temporal adjacency.
            let currentGlobalTimestamps: [Int]? =
                previousTokenTimestamps != nil ? hypothesis.timestamps.map { $0 + globalFrameOffset } : nil
            let (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: previousTokens, current: hypothesis.ySequence,
                previousTimestamps: previousTokenTimestamps,
                currentTimestamps: currentGlobalTimestamps)
            let adjustedTimestamps =
                removedCount > 0 ? Array(hypothesis.timestamps.dropFirst(removedCount)) : hypothesis.timestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(hypothesis.tokenConfidences.dropFirst(removedCount)) : hypothesis.tokenConfidences

            return (deduped, adjustedTimestamps, adjustedConfidences, encLen)
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, encLen)
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        confidences: [Float] = [],
        tokenDurations: [Int] = [],
        encoderSequenceLength: Int,
        audioSampleCount: Int,
        processingTime: TimeInterval
    ) -> ASRResult {

        let text = convertTokensToText(tokenIds)
        let duration = TimeInterval(audioSampleCount) / TimeInterval(config.sampleRate)

        let resultTimings = createTokenTimings(
            from: tokenIds, timestamps: timestamps, confidences: confidences, tokenDurations: tokenDurations)

        let confidence = calculateConfidence(
            tokenCount: tokenIds.count,
            isEmpty: text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            tokenConfidences: confidences
        )

        return ASRResult(
            text: text,
            confidence: confidence,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: resultTimings
        )
    }

}
