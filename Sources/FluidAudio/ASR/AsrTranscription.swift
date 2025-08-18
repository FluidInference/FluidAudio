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

        let result = try await ChunkProcessor(
            audioSamples: audioSamples,
            chunkSize: 160_000,
            enableDebug: config.enableDebug
        ).process(using: self, startTime: startTime)

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

    func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        var allTexts: [String] = []
        let audioLength = Double(audioSamples.count) / 16000.0

        var position = 0
        var chunkIndex = 0
        var decoderState = try DecoderState()

        while position < audioSamples.count {
            let text = try await processChunk(
                at: position, chunkIndex: chunkIndex, using: manager, decoderState: &decoderState)
            allTexts.append(text)
            position += chunkSize
            chunkIndex += 1
        }

        return ASRResult(
            text: allTexts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            confidence: 1.0,
            duration: audioLength,
            processingTime: Date().timeIntervalSince(startTime),
            tokenTimings: nil
        )
    }

    private func processChunk(
        at position: Int, chunkIndex: Int, using manager: AsrManager,
        decoderState: inout DecoderState
    ) async throws -> String {
        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)

        let (tokenIds, _) = try await manager.executeMLInference(
            paddedChunk, originalLength: chunkSamples.count, enableDebug: false, decoderState: &decoderState)
        let (text, _) = manager.convertTokensWithExistingTimings(tokenIds, timings: [])

        return text
    }
}
