import CoreML
import Foundation
import OSLog

extension AsrManager {

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout TdtDecoderState
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }

        let startTime = Date()

        // Route to appropriate processing method based on audio length
        if audioSamples.count <= 320_000 {
            let originalLength = audioSamples.count
            let paddedAudio: [Float] = padAudioIfNeeded(audioSamples, targetLength: 240_000)
            let (hypothesis, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: originalLength,
                actualAudioFrames: nil,  // Will be calculated from originalLength
                decoderState: &decoderState
            )

            let result = processTranscriptionResult(
                tokenIds: hypothesis.ySequence,
                timestamps: hypothesis.timestamps,
                confidences: hypothesis.tokenConfidences,
                tokenDurations: hypothesis.tokenDurations,
                timestampSemantics: hypothesis.timestampSemantics,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )
            return result
        }

        // ChunkProcessor now uses the passed-in decoder state for continuity
        let processor = ChunkProcessor(audioSamples: audioSamples)
        return try await processor.process(using: self, decoderState: &decoderState, startTime: startTime)
    }

    internal func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        actualAudioFrames: Int? = nil,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {

        let preprocessorInput = try await preparePreprocessorInput(
            paddedAudio, actualLength: originalLength)

        guard let preprocessorModel = preprocessorModel, let encoderModel = encoderModel else {
            throw ASRError.notInitialized
        }

        let preprocessorOutput = try await preprocessorModel.compatPrediction(
            from: preprocessorInput,
            options: predictionOptions
        )

        let encoderInput = try prepareEncoderInput(
            encoder: encoderModel,
            preprocessorOutput: preprocessorOutput,
            originalInput: preprocessorInput
        )

        let encoderOutputProvider = try await encoderModel.compatPrediction(
            from: encoderInput,
            options: predictionOptions
        )

        let rawEncoderOutput = try extractFeatureValue(
            from: encoderOutputProvider, key: "encoder", errorMessage: "Invalid encoder output")
        let encoderLength = try extractFeatureValue(
            from: encoderOutputProvider, key: "encoder_length",
            errorMessage: "Invalid encoder output length")

        let encoderSequenceLength = encoderLength[0].intValue

        // Calculate actual audio frames if not provided using shared constants
        let actualFrames =
            actualAudioFrames ?? ASRConstants.calculateEncoderFrames(from: originalLength ?? paddedAudio.count)

        let hypothesis = try await tdtDecodeWithTimings(
            encoderOutput: rawEncoderOutput,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualFrames,
            originalAudioSamples: paddedAudio,
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset
        )

        return (hypothesis, encoderSequenceLength)
    }

    private func prepareEncoderInput(
        encoder: MLModel,
        preprocessorOutput: MLFeatureProvider,
        originalInput: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        let inputDescriptions = encoder.modelDescription.inputDescriptionsByName

        let missingNames = inputDescriptions.keys.filter { name in
            preprocessorOutput.featureValue(for: name) == nil
        }

        if missingNames.isEmpty {
            return preprocessorOutput
        }

        var features: [String: MLFeatureValue] = [:]

        for name in inputDescriptions.keys {
            if let value = preprocessorOutput.featureValue(for: name) {
                features[name] = value
                continue
            }

            if let fallback = originalInput.featureValue(for: name) {
                features[name] = fallback
                continue
            }

            let availableInputs = preprocessorOutput.featureNames.sorted().joined(separator: ", ")
            let fallbackInputs = originalInput.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Missing required encoder input: \(name). Available inputs: \(availableInputs), "
                    + "fallback inputs: \(fallbackInputs)"
            )
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Streaming-friendly chunk transcription that preserves decoder state and supports start-frame offset.
    /// This is used by both sliding window chunking and streaming paths to unify behavior.
    internal func transcribeStreamingChunk(
        _ chunkSamples: [Float],
        source: AudioSource,
        previousTokens: [Int] = []
    ) async throws -> (
        tokens: [Int],
        timestamps: [Int],
        confidences: [Float],
        durations: [Int],
        semantics: TdtHypothesis.TimestampSemantics,
        encoderSequenceLength: Int
    ) {
        // Select and copy decoder state for the source
        var state = (source == .microphone) ? microphoneDecoderState : systemDecoderState

        let originalLength = chunkSamples.count
        let padded = padAudioIfNeeded(chunkSamples, targetLength: 240_000)
        let (hypothesis, encLen) = try await executeMLInferenceWithTimings(
            padded,
            originalLength: originalLength,
            actualAudioFrames: nil,  // Will be calculated from originalLength
            decoderState: &state,
            contextFrameAdjustment: 0  // Non-streaming chunks don't use adaptive context
        )

        // Persist updated state back to the source-specific slot
        if source == .microphone {
            microphoneDecoderState = state
        } else {
            systemDecoderState = state
        }

        let durations = normalizedDurations(
            tokenDurations: hypothesis.tokenDurations,
            timestamps: hypothesis.timestamps,
            semantics: hypothesis.timestampSemantics
        )

        // Apply token deduplication if previous tokens are provided
        if !previousTokens.isEmpty && hypothesis.hasTokens {
            let (deduped, removedCount) = removeDuplicateTokenSequence(
                previous: previousTokens, current: hypothesis.ySequence)
            let adjustedTimestamps =
                removedCount > 0 ? Array(hypothesis.timestamps.dropFirst(removedCount)) : hypothesis.timestamps
            let adjustedConfidences =
                removedCount > 0
                ? Array(hypothesis.tokenConfidences.dropFirst(removedCount)) : hypothesis.tokenConfidences
            let adjustedDurations =
                removedCount > 0 ? Array(durations.dropFirst(removedCount)) : durations

            return (
                deduped,
                adjustedTimestamps,
                adjustedConfidences,
                adjustedDurations,
                hypothesis.timestampSemantics,
                encLen
            )
        }

        return (
            hypothesis.ySequence,
            hypothesis.timestamps,
            hypothesis.tokenConfidences,
            durations,
            hypothesis.timestampSemantics,
            encLen
        )
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        timestamps: [Int] = [],
        confidences: [Float] = [],
        tokenDurations: [Int] = [],
        timestampSemantics: TdtHypothesis.TimestampSemantics = .start,
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval,
        tokenTimings: [TokenTiming] = []
    ) -> ASRResult {

        let (text, finalTimings) = convertTokensWithExistingTimings(tokenIds, timings: tokenTimings)
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)

        // Convert timestamps to TokenTiming objects if provided
        let timingsFromTimestamps = createTokenTimings(
            from: tokenIds,
            timestamps: timestamps,
            confidences: confidences,
            tokenDurations: tokenDurations,
            semantics: timestampSemantics)

        // Use existing timings if provided, otherwise use timings from timestamps
        let resultTimings = tokenTimings.isEmpty ? timingsFromTimestamps : finalTimings

        // Calculate confidence based on actual model confidence scores from TDT decoder
        let confidence = calculateConfidence(
            duration: duration,
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

    internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }

    /// Calculate confidence score based purely on TDT model token confidence scores
    /// Returns the average of token-level softmax probabilities from the decoder
    /// Range: 0.1 (empty transcription) to 1.0 (perfect confidence)
    private func calculateConfidence(
        duration: Double, tokenCount: Int, isEmpty: Bool, tokenConfidences: [Float]
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
    private func createTokenTimings(
        from tokenIds: [Int],
        timestamps: [Int],
        confidences: [Float],
        tokenDurations: [Int] = [],
        semantics: TdtHypothesis.TimestampSemantics = .start
    ) -> [TokenTiming] {
        guard
            !tokenIds.isEmpty && !timestamps.isEmpty && tokenIds.count == timestamps.count
                && confidences.count == tokenIds.count
        else {
            return []
        }

        let durations = normalizedDurations(
            tokenDurations: tokenDurations,
            timestamps: timestamps,
            semantics: semantics
        )

        var timings: [TokenTiming] = []

        // Create combined data for sorting
        let combinedData = zip(
            zip(zip(tokenIds, timestamps), confidences),
            durations
        ).map {
            (tokenId: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
        }

        // Sort by timestamp to ensure chronological order
        let sortedData = combinedData.sorted { $0.timestamp < $1.timestamp }

        for data in sortedData {
            let tokenId = data.tokenId
            let referenceFrame = data.timestamp

            let durationFrames = max(data.duration, 1)

            let startFrame: Int
            let endFrame: Int

            switch semantics {
            case .start:
                startFrame = referenceFrame
                endFrame = referenceFrame + durationFrames
            case .end:
                endFrame = referenceFrame
                startFrame = max(0, referenceFrame - durationFrames)
            }

            // Convert encoder frame index to time (80ms per frame)
            let startTime = TimeInterval(startFrame) * 0.08
            let endTime = TimeInterval(endFrame) * 0.08

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

    private func normalizedDurations(
        tokenDurations: [Int],
        timestamps: [Int],
        semantics: TdtHypothesis.TimestampSemantics
    ) -> [Int] {
        guard !timestamps.isEmpty else { return [] }

        var derived = deriveDurationsFromTimestamps(timestamps, semantics: semantics)

        if tokenDurations.count == timestamps.count {
            for idx in 0..<derived.count {
                let provided = tokenDurations[idx]
                if provided > 0 {
                    derived[idx] = provided
                }
            }
        } else if !tokenDurations.isEmpty {
            logger.info(
                "AsrManager: token durations count \(tokenDurations.count) "
                    + "does not match timestamps \(timestamps.count); using derived durations."
            )
            if let lastProvided = tokenDurations.last, lastProvided > 0 {
                derived[derived.count - 1] = lastProvided
            }
        }

        return derived.map { max($0, 1) }
    }

    private func deriveDurationsFromTimestamps(
        _ timestamps: [Int],
        semantics: TdtHypothesis.TimestampSemantics
    ) -> [Int] {
        guard !timestamps.isEmpty else { return [] }

        var durations = Array(repeating: 1, count: timestamps.count)

        switch semantics {
        case .start:
            for index in 0..<timestamps.count {
                let current = timestamps[index]
                if index + 1 < timestamps.count {
                    let next = timestamps[index + 1]
                    durations[index] = max(1, next - current)
                } else {
                    durations[index] = max(durations[index], 1)
                }
            }
        case .end:
            var previousEnd = 0
            for (index, currentEnd) in timestamps.enumerated() {
                let delta = currentEnd - previousEnd
                durations[index] = max(1, delta)
                previousEnd = currentEnd
            }
        }

        return durations
    }

    /// Slice encoder output to remove left context frames (following NeMo approach)
    private func sliceEncoderOutput(
        _ encoderOutput: MLMultiArray,
        from startFrame: Int,
        newLength: Int
    ) throws -> MLMultiArray {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let hiddenSize = shape[2].intValue

        // Create new array with sliced dimensions
        let slicedArray = try MLMultiArray(
            shape: [batchSize, newLength, hiddenSize] as [NSNumber],
            dataType: encoderOutput.dataType
        )

        // Copy data from startFrame onwards
        let sourcePtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
        let destPtr = slicedArray.dataPointer.bindMemory(to: Float.self, capacity: slicedArray.count)

        for t in 0..<newLength {
            for h in 0..<hiddenSize {
                let sourceIndex = (startFrame + t) * hiddenSize + h
                let destIndex = t * hiddenSize + h
                destPtr[destIndex] = sourcePtr[sourceIndex]
            }
        }

        return slicedArray
    }

    /// Remove duplicate token sequences at chunk boundaries while keeping
    /// the removal conservative so we don't wipe out fresh content.
    internal func removeDuplicateTokenSequence(
        previous: [Int], current: [Int], maxOverlap: Int = 12
    ) -> (deduped: [Int], removedCount: Int) {

        let punctuationTokens = [7883, 7952, 7948]  // period, question, exclamation
        var workingCurrent = current
        var removedCount = 0

        if !previous.isEmpty && !workingCurrent.isEmpty && previous.last == workingCurrent.first
            && punctuationTokens.contains(workingCurrent.first!)
        {
            workingCurrent = Array(workingCurrent.dropFirst())
            removedCount += 1
        }

        let cappedOverlap = min(maxOverlap, previous.count, workingCurrent.count)
        if cappedOverlap > 0 {
            let previousString = previous.compactMap { vocabulary[$0] }.joined()
            let searchWindow = String(previousString.suffix(200))
            var accumulated = ""
            var sequentialDrop = 0
            let tolerance = 120
            for (index, token) in workingCurrent.prefix(cappedOverlap).enumerated() {
                guard let piece = vocabulary[token] else { break }
                accumulated += piece
                if let range = searchWindow.range(of: accumulated, options: .backwards) {
                    let distanceToEnd = searchWindow.distance(from: range.upperBound, to: searchWindow.endIndex)
                    if distanceToEnd <= tolerance {
                        sequentialDrop = index + 1
                        continue
                    }
                }
                break
            }

            if sequentialDrop > 0 {
                removedCount += sequentialDrop
                workingCurrent = Array(workingCurrent.dropFirst(sequentialDrop))
            }

            let remainingOverlap = min(maxOverlap, previous.count, workingCurrent.count)
            if remainingOverlap > 0 {
                for overlapLength in stride(from: remainingOverlap, through: 1, by: -1) {
                    let prevSuffix = Array(previous.suffix(overlapLength))
                    let currPrefix = Array(workingCurrent.prefix(overlapLength))

                    if prevSuffix == currPrefix || tokenPiecesMatch(prevSuffix, currPrefix) {
                        let finalRemoved = removedCount + overlapLength
                        let trimmed = Array(workingCurrent.dropFirst(overlapLength))
                        return (trimmed, finalRemoved)
                    }
                }
            }
        }

        return (workingCurrent, removedCount)
    }

    /// Calculate start frame offset for a sliding window segment (deprecated - now handled by timeJump)
    internal func calculateStartFrameOffset(segmentIndex: Int, leftContextSeconds: Double) -> Int {
        // This method is deprecated as frame tracking is now handled by the decoder's timeJump mechanism
        // Kept for test compatibility
        return 0
    }

    private func tokenPiecesMatch(_ lhs: [Int], _ rhs: [Int]) -> Bool {
        guard lhs.count == rhs.count else { return false }
        if lhs.isEmpty { return false }

        let lhsString = lhs.compactMap { vocabulary[$0] }.joined()
        let rhsString = rhs.compactMap { vocabulary[$0] }.joined()
        if lhsString == rhsString { return true }
        if lhsString.hasSuffix(rhsString) { return true }
        if rhsString.hasSuffix(lhsString) { return true }
        return false
    }
}
