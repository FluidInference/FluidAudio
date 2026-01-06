import Foundation

extension AsrManager {

    /// Create initial state for streaming chunk-by-chunk ASR processing.
    /// Similar to `VadManager.makeStreamState()` for low-latency streaming.
    ///
    /// Usage:
    /// ```swift
    /// var state = asrManager.makeStreamState()
    /// for await chunk in audioStream {
    ///     let result = try await asrManager.processStreamingChunk(chunk, state: state)
    ///     print(result.confirmed)  // Green - confirmed text
    ///     print(result.provisional) // Purple - awaiting validation
    ///     state = result.state
    /// }
    /// ```
    public func makeStreamState() -> AsrStreamState {
        AsrStreamState.initial()
    }

    /// Process a single audio chunk with streaming ASR and LocalAgreement-2 validation.
    ///
    /// Each chunk is transcribed with a fresh decoder state that's carried forward for continuity.
    /// Results are compared with the previous chunk to find validated prefixes (LocalAgreement-2).
    /// Matching portions are confirmed (green), while new tokens remain provisional (purple).
    ///
    /// - Parameters:
    ///   - audioChunk: Audio samples (PCM, typically 500ms-1s for responsive streaming)
    ///   - state: Current streaming state (carries decoder state and previous hypothesis)
    ///   - config: LocalAgreement validation configuration (default: 0.7 threshold, 60 frame max)
    ///
    /// - Returns: Result containing updated state and confirmed/provisional token split
    ///
    /// - Throws: ASRError if transcription fails
    public func processStreamingChunk(
        _ audioChunk: [Float],
        state: AsrStreamState,
        config: LocalAgreementConfig = .default
    ) async throws -> AsrStreamResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard !audioChunk.isEmpty else { throw ASRError.invalidAudioData }

        // Transcribe chunk with carried-forward decoder state
        var decoderStateCopy = state.decoderState
        let (hypothesis, _) = try await executeMLInferenceWithTimings(
            padAudioIfNeeded(audioChunk, targetLength: ASRConstants.maxModelSamples),
            originalLength: audioChunk.count,
            actualAudioFrames: ASRConstants.calculateEncoderFrames(from: audioChunk.count),
            decoderState: &decoderStateCopy,
            contextFrameAdjustment: 0,
            isLastChunk: false,
            globalFrameOffset: state.processedSamples / ASRConstants.samplesPerEncoderFrame
        )

        // LocalAgreement-2 temporal comparison with previous chunk
        let (confirmed, provisional, allTimestamps, allConfidences): ([Int], [Int], [Int], [Float])
        if let previous = state.previousHypothesis {
            // Find longest common prefix (tokens + confidences must match)
            let commonPrefixLen = findCommonPrefixLength(
                previousTokens: previous.ySequence,
                currentTokens: hypothesis.ySequence,
                previousConf: previous.tokenConfidences,
                currentConf: hypothesis.tokenConfidences,
                threshold: config.confidenceThreshold
            )

            confirmed = Array(hypothesis.ySequence.prefix(commonPrefixLen))
            provisional = Array(hypothesis.ySequence.dropFirst(commonPrefixLen))
            allTimestamps = hypothesis.timestamps
            allConfidences = hypothesis.tokenConfidences
        } else {
            // First chunk - everything is provisional (no previous to compare against)
            confirmed = []
            provisional = hypothesis.ySequence
            allTimestamps = hypothesis.timestamps
            allConfidences = hypothesis.tokenConfidences
        }

        // Update state for next chunk
        var nextState = state
        nextState.decoderState = decoderStateCopy
        nextState.previousHypothesis = hypothesis
        nextState.processedSamples += audioChunk.count
        nextState.confirmedTokens.append(contentsOf: confirmed)
        nextState.provisionalTokens = provisional

        // Force-confirm provisional tokens if too many accumulate (prevent latency growth)
        let maxProvisional = config.maxProvisionalTokens
        if nextState.provisionalTokens.count > maxProvisional {
            let forceConfirmCount = nextState.provisionalTokens.count - maxProvisional
            let toConfirm = Array(nextState.provisionalTokens.prefix(forceConfirmCount))
            nextState.confirmedTokens.append(contentsOf: toConfirm)
            nextState.provisionalTokens = Array(nextState.provisionalTokens.dropFirst(forceConfirmCount))

            logger.debug("Forced confirmation of \(forceConfirmCount) provisional tokens to prevent latency growth")
        }

        return AsrStreamResult(
            state: nextState,
            confirmed: confirmed,
            provisional: provisional,
            allTimestamps: allTimestamps,
            allConfidences: allConfidences
        )
    }

    /// Finalize streaming and promote all remaining provisional tokens to confirmed.
    /// Call this when the audio stream ends to get the final transcription.
    ///
    /// - Parameter state: Final streaming state after processing all chunks
    /// - Returns: All accumulated tokens (confirmed + promoted provisional)
    public func finalizeStreaming(_ state: AsrStreamState) -> [Int] {
        state.confirmedTokens + state.provisionalTokens
    }

    // MARK: - Private Helpers

    /// Find the length of the longest common token prefix with confidence validation.
    /// Used by LocalAgreement-2 to determine stable output across consecutive chunks.
    private func findCommonPrefixLength(
        previousTokens: [Int],
        currentTokens: [Int],
        previousConf: [Float],
        currentConf: [Float],
        threshold: Float
    ) -> Int {
        guard !previousTokens.isEmpty && !currentTokens.isEmpty else {
            return 0
        }
        guard previousConf.count == previousTokens.count && currentConf.count == currentTokens.count else {
            return 0
        }

        let minLength = min(previousTokens.count, currentTokens.count)
        var matchLength = 0

        for i in 0..<minLength {
            let tokensMatch = previousTokens[i] == currentTokens[i]
            let bothAboveThreshold = previousConf[i] >= threshold && currentConf[i] >= threshold

            if tokensMatch && bothAboveThreshold {
                matchLength += 1
            } else {
                // Stop at first disagreement (longest prefix)
                break
            }
        }

        return matchLength
    }
}
