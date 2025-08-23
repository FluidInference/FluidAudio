import CoreML
import Foundation
import OSLog

extension AsrManager {

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout TdtDecoderState
    ) async throws -> ASRResult {
        guard isAvailable else { throw ASRError.notInitialized }
        guard audioSamples.count >= 16_000 else { throw ASRError.invalidAudioData }

        if config.enableDebug {
            logger.debug("transcribeWithState: processing \(audioSamples.count) samples")
            // Log decoder state values before processing
            let hiddenBefore = (
                decoderState.hiddenState[0].intValue, decoderState.hiddenState[1].intValue
            )
            let cellBefore = (
                decoderState.cellState[0].intValue, decoderState.cellState[1].intValue
            )
            logger.debug(
                "Decoder state before: hidden[\(hiddenBefore.0),\(hiddenBefore.1)], cell[\(cellBefore.0),\(cellBefore.1)]"
            )
        }

        let startTime = Date()

        if audioSamples.count <= 240_000 {
            let originalLength = audioSamples.count
            let paddedAudio = padAudioIfNeeded(audioSamples, targetLength: 240_000)
            let (tokens, _timestamps, encoderSequenceLength) = try await executeMLInferenceWithTimings(
                paddedAudio,
                originalLength: originalLength,
                enableDebug: config.enableDebug,
                decoderState: &decoderState
            )

            let result = processTranscriptionResult(
                tokenIds: tokens,
                encoderSequenceLength: encoderSequenceLength,
                audioSamples: audioSamples,
                processingTime: Date().timeIntervalSince(startTime)
            )

            if config.enableDebug {
                // Log decoder state values after processing
                let hiddenAfter = (
                    decoderState.hiddenState[0].intValue, decoderState.hiddenState[1].intValue
                )
                let cellAfter = (decoderState.cellState[0].intValue, decoderState.cellState[1].intValue)
                logger.debug(
                    "Decoder state after: hidden[\(hiddenAfter.0),\(hiddenAfter.1)], cell[\(cellAfter.0),\(cellAfter.1)]"
                )
                logger.debug("Transcription result: '\(result.text)'")
            }

            return result
        }

        // ChunkProcessor now uses the passed-in decoder state for continuity
        let processor = ChunkProcessor(audioSamples: audioSamples, enableDebug: config.enableDebug)
        return try await processor.process(using: self, decoderState: &decoderState, startTime: startTime)
    }

    /// Deprecated: use executeMLInferenceWithTimings and ignore timestamps if not needed
    @available(*, deprecated, message: "Use executeMLInferenceWithTimings to also retrieve emission timestamps")
    internal func executeMLInference(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        enableDebug: Bool = false,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokenIds: [Int], encoderSequenceLength: Int) {
        let (tokens, _, encLen) = try await executeMLInferenceWithTimings(
            paddedAudio,
            originalLength: originalLength,
            enableDebug: enableDebug,
            decoderState: &decoderState
        )
        return (tokens, encLen)
    }

    /// Execute ML inference and return tokens with emission timestamps (encoder frame indices)
    internal func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        enableDebug: Bool = false,
        decoderState: inout TdtDecoderState,
        startFrameOffset: Int = 0,
        lastProcessedFrame: Int = 0
    ) async throws -> (tokens: [Int], timestamps: [Int], encoderSequenceLength: Int) {

        let melspectrogramInput = try await prepareMelSpectrogramInput(
            paddedAudio, actualLength: originalLength)

        guard
            let melspectrogramOutput = try melspectrogramModel?.prediction(
                from: melspectrogramInput,
                options: predictionOptions
            )
        else {
            throw ASRError.processingFailed("Mel-spectrogram model failed")
        }

        let encoderInput = try prepareEncoderInput(melspectrogramOutput)

        guard
            let encoderOutput = try encoderModel?.prediction(
                from: encoderInput,
                options: predictionOptions
            )
        else {
            throw ASRError.processingFailed("Encoder model failed")
        }

        let rawEncoderOutput = try extractFeatureValue(
            from: encoderOutput, key: "encoder_output", errorMessage: "Invalid encoder output")
        let encoderLength = try extractFeatureValue(
            from: encoderOutput, key: "encoder_output_length",
            errorMessage: "Invalid encoder output length")

        let encoderHiddenStates = rawEncoderOutput
        let encoderSequenceLength = encoderLength[0].intValue

        let (tokens, timestamps) = try await tdtDecodeWithTimings(
            encoderOutput: encoderHiddenStates,
            encoderSequenceLength: encoderSequenceLength,
            originalAudioSamples: paddedAudio,
            decoderState: &decoderState,
            startFrameOffset: startFrameOffset,
            lastProcessedFrame: lastProcessedFrame
        )

        return (tokens, timestamps, encoderSequenceLength)
    }

    internal func processTranscriptionResult(
        tokenIds: [Int],
        encoderSequenceLength: Int,
        audioSamples: [Float],
        processingTime: TimeInterval,
        tokenTimings: [TokenTiming] = []
    ) -> ASRResult {

        let (text, finalTimings) = convertTokensWithExistingTimings(tokenIds, timings: tokenTimings)
        let duration = TimeInterval(audioSamples.count) / TimeInterval(config.sampleRate)

        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && duration > 1.0 {
            logger.warning(
                "⚠️ Empty transcription for \(String(format: "%.1f", duration))s audio (tokens: \(tokenIds.count))"
            )
        }

        return ASRResult(
            text: text,
            confidence: 1.0,
            duration: duration,
            processingTime: processingTime,
            tokenTimings: finalTimings
        )
    }

    internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
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

}

private struct ChunkProcessor {
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
        let audioLength = Double(audioSamples.count) / Double(sampleRate)

        var centerStart = 0
        var segmentIndex = 0
        var lastProcessedFrame = 0  // Track the last frame processed by previous chunk

        while centerStart < audioSamples.count {
            let chunkStartTime = Date()
            let (windowTokens, maxFrame) = try await processWindowWithTokens(
                centerStart: centerStart,
                segmentIndex: segmentIndex,
                lastProcessedFrame: lastProcessedFrame,
                using: manager,
                decoderState: &decoderState
            )
            let chunkDuration = Date().timeIntervalSince(chunkStartTime)

            // Update last processed frame for next chunk
            if maxFrame > 0 {
                lastProcessedFrame = maxFrame
            }

            // For chunks after the first, check for and remove duplicated token sequences
            if segmentIndex > 0 && !allTokens.isEmpty && !windowTokens.isEmpty {
                let deduplicatedTokens = removeDuplicateSequence(previous: allTokens, current: windowTokens)
                allTokens.append(contentsOf: deduplicatedTokens)

                if enableDebug && deduplicatedTokens.count != windowTokens.count {
                    print(
                        "CHUNK \(segmentIndex): removed \(windowTokens.count - deduplicatedTokens.count) duplicate tokens"
                    )
                }
            } else {
                allTokens.append(contentsOf: windowTokens)
            }

            if enableDebug {
                // Debug: Convert tokens to text for this chunk to see what was produced
                let (chunkText, _) = manager.convertTokensWithExistingTimings(windowTokens, timings: [])
                print("CHUNK \(segmentIndex) tokens: \(windowTokens)")
                print("CHUNK \(segmentIndex) text: '\(chunkText)'")
                print("ALL tokens so far: \(allTokens)")
                let (totalText, _) = manager.convertTokensWithExistingTimings(allTokens, timings: [])
                print("TOTAL text so far: '\(totalText)'")
                print(
                    "CHUNK \(segmentIndex): duration=\(String(format: "%.3f", chunkDuration))s, tokens=\(windowTokens.count), total_tokens=\(allTokens.count)"
                )
            }

            centerStart += centerSamples
            segmentIndex += 1
        }

        let (text, _) = manager.convertTokensWithExistingTimings(allTokens, timings: [])

        return ASRResult(
            text: text,
            confidence: 1.0,
            duration: audioLength,
            processingTime: Date().timeIntervalSince(startTime),
            tokenTimings: nil
        )
    }

    private func processWindowWithTokens(
        centerStart: Int,
        segmentIndex: Int,
        lastProcessedFrame: Int,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], maxFrame: Int) {
        // Compute window bounds in samples: [leftStart, rightEnd)
        let leftStart = max(0, centerStart - leftContextSamples)
        let centerEnd = min(audioSamples.count, centerStart + centerSamples)
        let rightEnd = min(audioSamples.count, centerEnd + rightContextSamples)

        // If nothing to process, return empty
        if leftStart >= rightEnd { return ([], 0) }

        let chunkSamples = Array(audioSamples[leftStart..<rightEnd])
        let chunkAudioDuration = Double(chunkSamples.count) / Double(sampleRate)

        // Pad to model capacity (15s) if needed; keep track of actual chunk length
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: maxModelSamples)

        // Calculate encoder frame offset based on where previous chunk ended
        let startFrameOffset: Int
        if segmentIndex == 0 {
            // First chunk: process all frames
            startFrameOffset = 0
        } else {
            // Subsequent chunks: use fixed offset calculation but track last processed frame for filtering
            let exactEncoderFrameRate = 12.6
            let leftContextFrames = Int(round(leftContextSeconds * exactEncoderFrameRate))
            let fixedOffset = leftContextFrames + 2  // Original calculation: ~29

            // Use the fixed offset for consistency, rely on decoder filtering to prevent repetition
            startFrameOffset = fixedOffset

            print(
                "CHUNK \(segmentIndex): lastProcessedFrame=\(lastProcessedFrame), fixedOffset=\(fixedOffset), startFrameOffset=\(startFrameOffset)"
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
            return ([], 0)
        }

        // Take all tokens from decoder (it already processed only the relevant frames)
        let filteredTokens = tokens
        let maxFrame = timestamps.max() ?? 0

        print(
            "CHUNK \(segmentIndex): audio_duration=\(String(format: "%.2f", chunkAudioDuration))s, startFrameOffset=\(startFrameOffset), taking all \(tokens.count) tokens"
        )

        return (filteredTokens, maxFrame)
    }

    /// Remove duplicate token sequences from the beginning of current chunk that match the end of previous chunk
    private func removeDuplicateSequence(previous: [Int], current: [Int]) -> [Int] {
        // Look for subsequence matches within reasonable bounds
        let maxSearchLength = min(15, previous.count)  // Look at last 15 tokens of previous
        let maxMatchLength = min(12, current.count)  // Look at first 12 tokens of current

        // Ensure we have at least 3 tokens to check for duplication
        guard maxSearchLength >= 3 && maxMatchLength >= 3 else {
            return current
        }

        // Try different overlap lengths, prioritizing longer matches
        for overlapLength in (3...min(maxSearchLength, maxMatchLength)).reversed() {
            // Look for this overlap length anywhere in the tail of previous
            for startIndex in max(0, previous.count - maxSearchLength)..<(previous.count - overlapLength + 1) {
                let previousSubsequence = Array(previous[startIndex..<startIndex + overlapLength])

                // Check multiple positions in current chunk, not just the beginning
                for currentStartIndex in 0..<min(5, current.count - overlapLength + 1) {
                    let currentSubsequence = Array(current[currentStartIndex..<currentStartIndex + overlapLength])

                    if previousSubsequence == currentSubsequence {
                        print(
                            "Found duplicate sequence of length \(overlapLength) at position \(currentStartIndex): \(previousSubsequence)"
                        )
                        return Array(current.dropFirst(currentStartIndex + overlapLength))
                    }
                }
            }
        }

        // No duplication found, return original tokens
        return current
    }
}
