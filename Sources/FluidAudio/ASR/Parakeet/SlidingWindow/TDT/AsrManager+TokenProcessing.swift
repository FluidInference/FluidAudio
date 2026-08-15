import Foundation

extension AsrManager {

    /// Calculate confidence score based purely on TDT model token confidence scores
    /// Returns the average of token-level softmax probabilities from the decoder
    /// Range: 0.1 (empty transcription) to 1.0 (perfect confidence)
    nonisolated internal func calculateConfidence(
        tokenCount: Int, isEmpty: Bool, tokenConfidences: [Float]
    ) -> Float {
        // Empty transcription gets low confidence
        if isEmpty {
            return 0.1
        }

        // We should always have token confidence scores from the TDT decoder
        guard !tokenConfidences.isEmpty && tokenConfidences.count == tokenCount else {
            logger.warning("Expected token confidences but got none - this should not happen")
            return 0.5  // Default middle confidence if something went wrong
        }

        // Return pure model confidence: average of token-level softmax probabilities
        let meanConfidence = tokenConfidences.reduce(0.0, +) / Float(tokenConfidences.count)

        // Ensure confidence is in valid range (clamp to avoid edge cases)
        return max(0.1, min(1.0, meanConfidence))
    }

    /// Convert frame timestamps to TokenTiming objects
    internal func createTokenTimings(
        from tokenIds: [Int], timestamps: [Int], confidences: [Float], tokenDurations: [Int] = []
    ) -> [TokenTiming] {
        guard
            !tokenIds.isEmpty && !timestamps.isEmpty && tokenIds.count == timestamps.count
                && confidences.count == tokenIds.count
        else {
            return []
        }

        var timings: [TokenTiming] = []

        // Create combined data for sorting
        let combinedData = zip(
            zip(zip(tokenIds, timestamps), confidences),
            tokenDurations.isEmpty ? Array(repeating: 0, count: tokenIds.count) : tokenDurations
        ).map {
            (tokenId: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
        }

        // Sort by timestamp to ensure chronological order
        let sortedData = combinedData.sorted { $0.timestamp < $1.timestamp }

        let frameDuration = ASRConstants.secondsPerEncoderFrame

        // TDT emission-delay correction: tokens are emitted ~1 encoder frame after
        // the acoustic event (median offsetStart = +1 frame on earnings22 across
        // both v2 and v3). Shift each token's frame index down by one so timestamps
        // line up with where the rescorer's CTC head sees the peak.
        // Override via TDT_EMISSION_DELAY_FRAMES env var for sweeps.
        let emissionDelayFrames =
            ProcessInfo.processInfo.environment["TDT_EMISSION_DELAY_FRAMES"]
            .flatMap { Int($0) } ?? 1

        for i in 0..<sortedData.count {
            let data = sortedData[i]
            let tokenId = data.tokenId
            let frameIndex = max(0, data.timestamp - emissionDelayFrames)
            let nextFrameIndex =
                i < sortedData.count - 1
                ? max(0, sortedData[i + 1].timestamp - emissionDelayFrames)
                : frameIndex

            let startTime = TimeInterval(frameIndex) * frameDuration

            // Calculate end time using actual token duration if available
            let endTime: TimeInterval
            if !tokenDurations.isEmpty && data.duration > 0 {
                let durationInSeconds = TimeInterval(data.duration) * frameDuration
                endTime = startTime + max(durationInSeconds, frameDuration)
            } else if i < sortedData.count - 1 {
                let nextStartTime = TimeInterval(nextFrameIndex) * frameDuration
                endTime = max(nextStartTime, startTime + frameDuration)
            } else {
                endTime = startTime + frameDuration
            }

            // Validate that end time is after start time
            let validatedEndTime = max(endTime, startTime + 0.001)  // Minimum 1ms gap

            // Get token text from vocabulary if available and normalize for timing display
            let rawToken = vocabulary[tokenId] ?? "token_\(tokenId)"
            let tokenText = normalizedTimingToken(rawToken)

            // Use actual confidence score from TDT decoder
            let tokenConfidence = data.confidence

            let timing = TokenTiming(
                token: tokenText,
                tokenId: tokenId,
                startTime: startTime,
                endTime: validatedEndTime,
                confidence: tokenConfidence
            )

            timings.append(timing)
        }
        return timings
    }

    /// A token paired with the global encoder-frame index at which it was emitted.
    /// Used to make dedup matching temporally aware; `timestamp < 0` means "unknown".
    private struct TimedToken {
        let id: Int
        let timestamp: Int
    }

    /// Remove duplicate token sequences at the start of the current list that overlap
    /// with the tail of the previous accumulated tokens. Returns deduplicated current tokens
    /// and the number of removed leading tokens so caller can drop aligned timestamps.
    ///
    /// When `previousTimestamps` and `currentTimestamps` (both in **global** encoder frames)
    /// are supplied, a token match only counts as a genuine chunk-boundary duplicate if the
    /// two occurrences fall within `frameTolerance` frames of each other — i.e. they describe
    /// the same acoustic moment in the overlap region. This prevents false positives where two
    /// unrelated words spoken seconds apart share a short subword prefix (issue #787). Without
    /// timestamps the matcher falls back to pure token-ID equality (legacy behavior).
    ///
    /// Ideally this is not needed. We need to make some more fixes to the TDT decoding logic,
    /// this should be a temporary workaround.
    nonisolated internal func removeDuplicateTokenSequence(
        previous: [Int], current: [Int], maxOverlap: Int = 12,
        previousTimestamps: [Int]? = nil,
        currentTimestamps: [Int]? = nil,
        frameTolerance: Int = ASRConstants.duplicateFrameTolerance
    ) -> (deduped: [Int], removedCount: Int) {

        // Handle single punctuation token duplicates first (domain-specific)
        let punctuationTokens = ASRConstants.punctuationTokens
        var workingCurrent = current
        var workingCurrentTimestamps = currentTimestamps
        var removedCount = 0

        if !previous.isEmpty && !workingCurrent.isEmpty && previous.last == workingCurrent.first
            && punctuationTokens.contains(workingCurrent.first!)
        {
            workingCurrent = Array(workingCurrent.dropFirst())
            workingCurrentTimestamps = workingCurrentTimestamps.map { Array($0.dropFirst()) }
            removedCount += 1
        }

        // Pair tokens with their global frame timestamps so the matcher can reject
        // token-ID coincidences that don't line up in time. When timestamps are absent
        // (or misaligned) the sentinel -1 makes the matcher fall back to ID-only equality.
        func timed(_ ids: [Int], _ timestamps: [Int]?) -> [TimedToken] {
            if let timestamps, timestamps.count == ids.count {
                return zip(ids, timestamps).map { TimedToken(id: $0, timestamp: $1) }
            }
            return ids.map { TimedToken(id: $0, timestamp: -1) }
        }

        let previousTimed = timed(previous, previousTimestamps)
        let currentTimed = timed(workingCurrent, workingCurrentTimestamps)

        // Token IDs must match, and — when both timestamps are known — the two
        // occurrences must describe the same acoustic moment (within tolerance).
        let temporalMatcher: (TimedToken, TimedToken) -> Bool = { a, b in
            guard a.id == b.id else { return false }
            guard a.timestamp >= 0 && b.timestamp >= 0 else { return true }
            return abs(a.timestamp - b.timestamp) <= frameTolerance
        }

        // STAGE 2: Suffix-prefix overlap using extracted utility
        if let match = SequenceMatcher.findSuffixPrefixMatch(
            previous: previousTimed,
            current: currentTimed,
            maxOverlap: maxOverlap,
            matcher: temporalMatcher
        ) {
            logger.debug("Found exact suffix-prefix overlap of length \(match.length)")
            let finalRemoved = removedCount + match.length
            return (Array(workingCurrent.dropFirst(match.length)), finalRemoved)
        }

        // STAGE 3: Extended search using extracted utility
        let boundarySearchFrames = config.tdtConfig.boundarySearchFrames
        let maxSearchLength = min(15, previous.count)

        if let match = SequenceMatcher.findBoundedSubstringMatch(
            previous: previousTimed,
            current: currentTimed,
            maxSearchLength: maxSearchLength,
            boundarySearchFrames: boundarySearchFrames,
            matcher: temporalMatcher
        ) {
            logger.debug(
                "Found duplicate sequence length=\(match.length) at currStart=\(match.rightStartIndex) (boundarySearch=\(boundarySearchFrames))"
            )
            let finalRemoved = removedCount + match.rightStartIndex + match.length
            return (Array(workingCurrent.dropFirst(match.rightStartIndex + match.length)), finalRemoved)
        }

        return (workingCurrent, removedCount)
    }

}
