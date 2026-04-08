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

        for i in 0..<sortedData.count {
            let data = sortedData[i]
            let tokenId = data.tokenId
            let frameIndex = data.timestamp

            let startTime = TimeInterval(frameIndex) * frameDuration

            // Calculate end time using actual token duration if available
            let endTime: TimeInterval
            if !tokenDurations.isEmpty && data.duration > 0 {
                let durationInSeconds = TimeInterval(data.duration) * frameDuration
                endTime = startTime + max(durationInSeconds, frameDuration)
            } else if i < sortedData.count - 1 {
                let nextStartTime = TimeInterval(sortedData[i + 1].timestamp) * frameDuration
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

    /// Remove duplicate token sequences at the start of the current list that overlap
    /// with the tail of the previous accumulated tokens. Returns deduplicated current tokens
    /// and the number of removed leading tokens so caller can drop aligned timestamps.
    /// Ideally this is not needed. We need to make some more fixes to the TDT decoding logic,
    /// this should be a temporary workaround.
    nonisolated internal func removeDuplicateTokenSequence(
        previous: [Int], current: [Int], maxOverlap: Int = 12
    ) -> (deduped: [Int], removedCount: Int) {

        // Handle single punctuation token duplicates first (domain-specific)
        let punctuationTokens = ASRConstants.punctuationTokens
        var workingCurrent = current
        var removedCount = 0

        if !previous.isEmpty && !workingCurrent.isEmpty && previous.last == workingCurrent.first
            && punctuationTokens.contains(workingCurrent.first!)
        {
            workingCurrent = Array(workingCurrent.dropFirst())
            removedCount += 1
        }

        // STAGE 2: Suffix-prefix overlap using extracted utility
        let exactMatcher: (Int, Int) -> Bool = { $0 == $1 }
        if let match = SequenceMatcher.findSuffixPrefixMatch(
            previous: previous,
            current: workingCurrent,
            maxOverlap: maxOverlap,
            matcher: exactMatcher
        ) {
            logger.debug("Found exact suffix-prefix overlap of length \(match.length)")
            let finalRemoved = removedCount + match.length
            return (Array(workingCurrent.dropFirst(match.length)), finalRemoved)
        }

        // STAGE 3: Extended search using extracted utility
        let boundarySearchFrames = config.tdtConfig.boundarySearchFrames
        let maxSearchLength = min(15, previous.count)

        if let match = SequenceMatcher.findBoundedSubstringMatch(
            previous: previous,
            current: workingCurrent,
            maxSearchLength: maxSearchLength,
            boundarySearchFrames: boundarySearchFrames,
            matcher: exactMatcher
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
