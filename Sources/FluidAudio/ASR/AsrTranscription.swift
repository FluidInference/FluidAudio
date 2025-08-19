import CoreML
import Foundation
import OSLog

extension AsrManager {

    /// Transpose encoder output from [B, H, T] to [B, T, H]
    private func transposeEncoderOutput(_ input: MLMultiArray) throws -> MLMultiArray {
        let shape = input.shape
        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid encoder output shape for transpose: \(shape)")
        }

        let batchSize = shape[0].intValue  // Should be 1
        let hiddenSize = shape[1].intValue  // Should be 1024
        let timeSteps = shape[2].intValue  // Should be ~126

        // Create output array with transposed shape [B, T, H]
        let output = try MLMultiArray(
            shape: [batchSize, timeSteps, hiddenSize] as [NSNumber],
            dataType: .float32
        )

        // Transpose by copying data
        // Input layout: [batch][hidden][time]
        // Output layout: [batch][time][hidden]
        for b in 0..<batchSize {
            for t in 0..<timeSteps {
                for h in 0..<hiddenSize {
                    // Input index: b * (hiddenSize * timeSteps) + h * timeSteps + t
                    let inputIndex = b * (hiddenSize * timeSteps) + h * timeSteps + t
                    // Output index: b * (timeSteps * hiddenSize) + t * hiddenSize + h
                    let outputIndex = b * (timeSteps * hiddenSize) + t * hiddenSize + h
                    output[outputIndex] = input[inputIndex]
                }
            }
        }

        return output
    }

    internal func transcribeWithState(
        _ audioSamples: [Float], decoderState: inout DecoderState
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

        if audioSamples.count <= 160_000 {
            let originalLength = audioSamples.count
            let paddedAudio = padAudioIfNeeded(audioSamples, targetLength: 160_000)
            let (tokenIds, encoderSequenceLength) = try await executeMLInference(
                paddedAudio,
                originalLength: originalLength,
                enableDebug: config.enableDebug,
                decoderState: &decoderState
            )

            let result = processTranscriptionResult(
                tokenIds: tokenIds,
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

        let processor = ChunkProcessor(
            audioSamples: audioSamples,
            chunkSize: 160_000,  // 10 seconds at 16kHz
            enableDebug: config.enableDebug
        )
        let result = try await processor.process(using: self, startTime: startTime)

        // Note: ChunkProcessor uses its own decoder state, so we don't update the passed-in state
        return result
    }

    internal func executeMLInference(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        enableDebug: Bool = false,
        decoderState: inout DecoderState
    ) async throws -> (tokenIds: [Int], encoderSequenceLength: Int) {

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

        // The encoder outputs in [B, H, T] format but we need [B, T, H]
        // where B=batch, T=time, H=hidden
        // So we need to transpose dimensions 1 and 2
        let transposedOutput = try transposeEncoderOutput(rawEncoderOutput)
        let encoderHiddenStates = transposedOutput
        let encoderSequenceLength = encoderLength[0].intValue

        if enableDebug {
            let audioLengthSeconds = Double(originalLength ?? paddedAudio.count) / 16000.0
            let expectedFrames = Int(audioLengthSeconds * 16000.0 / 160.0)  // ~100 frames per second
            logger.debug(
                "Audio length: \(String(format: "%.2f", audioLengthSeconds))s, Expected frames: ~\(expectedFrames), Encoder sequence length: \(encoderSequenceLength)"
            )

            if encoderSequenceLength < expectedFrames / 2 {
                logger.warning(
                    "⚠️ Encoder sequence length (\(encoderSequenceLength)) seems too short for \(String(format: "%.2f", audioLengthSeconds))s audio"
                )
            }
        }

        let tokenIds = try await tdtDecode(
            encoderOutput: encoderHiddenStates,
            encoderSequenceLength: encoderSequenceLength,
            originalAudioSamples: paddedAudio,
            decoderState: &decoderState
        )

        return (tokenIds, encoderSequenceLength)
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
            if tokenIds.isEmpty {
                logger.warning("   No tokens predicted - model only output blanks")
            } else {
                logger.warning("   Predicted tokens (first 10): \(Array(tokenIds.prefix(10)))")
            }
        } else if !text.isEmpty {
            logger.info("✅ Transcribed: '\(text)'")
            if tokenIds.count > 0 {
                logger.info("   Token IDs (first 10): \(Array(tokenIds.prefix(10)))")
            }
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
}

private struct ChunkProcessor {
    let audioSamples: [Float]
    let chunkSize: Int
    let enableDebug: Bool

    // Tactic 2: Remove overlap to avoid state corruption
    // No overlap between chunks - process sequentially without repetition
    private let overlap: Int = 0  // No overlap to prevent state mismatch
    private let stepSize: Int  // Calculated as chunkSize - overlap

    init(audioSamples: [Float], chunkSize: Int, enableDebug: Bool) {
        self.audioSamples = audioSamples
        self.chunkSize = chunkSize
        self.enableDebug = enableDebug
        self.stepSize = chunkSize  // No overlap, full chunk size
    }

    func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        let audioLength = Double(audioSamples.count) / 16000.0

        // For audio shorter than chunk size, process directly
        if audioSamples.count <= chunkSize {
            var decoderState = try DecoderState()
            let (tokenIds, _) = try await manager.executeMLInference(
                audioSamples, originalLength: audioSamples.count, enableDebug: enableDebug, decoderState: &decoderState)
            let (text, _) = manager.convertTokensWithExistingTimings(tokenIds, timings: [])

            return ASRResult(
                text: text,
                confidence: 1.0,
                duration: audioLength,
                processingTime: Date().timeIntervalSince(startTime),
                tokenTimings: nil
            )
        }

        var allTokens: [Int] = []
        var position = 0
        var chunkIndex = 0
        var decoderState = try DecoderState()

        // Tactic 1: Reset decoder state between chunks by default
        // This improves WER from 9% to 7.4% by preventing state corruption
        let resetStateBetweenChunks = true

        while position < audioSamples.count {
            // Reset decoder state for each chunk to prevent state corruption
            if resetStateBetweenChunks && chunkIndex > 0 {
                decoderState = try DecoderState()
                if enableDebug {
                    print("DEBUG: Reset decoder state for chunk \(chunkIndex)")
                }
            }

            let (chunkTokens, newDecoderState) = try await processChunk(
                at: position, chunkIndex: chunkIndex, using: manager, decoderState: decoderState)

            // Simply append all tokens from all chunks
            allTokens.append(contentsOf: chunkTokens)

            // Only update state if not resetting between chunks
            if !resetStateBetweenChunks {
                decoderState = newDecoderState
            }

            position += stepSize
            chunkIndex += 1

            if enableDebug {
                print(
                    "DEBUG: Chunk \(chunkIndex): position \(position)/\(audioSamples.count), tokens: \(chunkTokens.count)"
                )
                if chunkTokens.count < 10 && chunkIndex > 0 {
                    print("WARNING: Chunk \(chunkIndex) produced very few tokens (\(chunkTokens.count))")
                }
            }
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

    private func processChunk(
        at position: Int, chunkIndex: Int, using manager: AsrManager,
        decoderState: DecoderState
    ) async throws -> ([Int], DecoderState) {
        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)

        var workingState = decoderState
        let (tokenIds, _) = try await manager.executeMLInference(
            paddedChunk, originalLength: chunkSamples.count, enableDebug: enableDebug, decoderState: &workingState)

        if enableDebug && chunkIndex > 0 {
            print(
                "DEBUG: Chunk \(chunkIndex) state - lastToken: \(workingState.lastToken ?? -1), tokens generated: \(tokenIds.count)"
            )
        }

        return (tokenIds, workingState)
    }

    /// Deduplicate overlapping tokens between chunks (not used when overlap=0)
    private func deduplicateTokens(
        previousTokens: [Int],
        currentTokens: [Int],
        overlap: Int
    ) -> [Int] {
        // With no overlap, this function shouldn't be called
        // But keep it for potential future use
        guard overlap > 0 else {
            return currentTokens
        }

        // Estimate how many tokens might overlap based on audio overlap
        // Rough estimate: 1 second of audio ≈ 2-4 tokens for normal speech
        let estimatedOverlapTokens = min(6, currentTokens.count / 2, previousTokens.count / 2)

        guard estimatedOverlapTokens > 0 else {
            return currentTokens
        }

        // Look for the best match between end of previous and start of current
        let previousEnd = Array(previousTokens.suffix(estimatedOverlapTokens))

        // Find the longest matching suffix-prefix
        for skipCount in 0..<min(estimatedOverlapTokens, currentTokens.count) {
            let currentStart = Array(currentTokens.prefix(estimatedOverlapTokens - skipCount))

            if previousEnd.suffix(currentStart.count) == currentStart {
                // Found overlap, skip the duplicated part
                if enableDebug {
                    print("DEBUG: Found \(currentStart.count) overlapping tokens, skipping them")
                }
                return Array(currentTokens.dropFirst(currentStart.count))
            }
        }

        // No overlap found, use all tokens but be conservative
        return currentTokens
    }
}
