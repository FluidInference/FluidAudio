import CoreML
import Foundation
import OSLog

extension AsrManager {

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

        let encoderHiddenStates = rawEncoderOutput
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

    private let stepSize: Int  // Calculated as chunkSize - overlap
    private let overlapSize: Int  // Overlap between chunks

    init(audioSamples: [Float], chunkSize: Int, enableDebug: Bool) {
        self.audioSamples = audioSamples
        self.chunkSize = chunkSize
        self.enableDebug = enableDebug

        // Check if sliding window is enabled
        let useSlidingWindow = ProcessInfo.processInfo.environment["USE_SLIDING_WINDOW"] != nil

        if useSlidingWindow {
            // Default to 25% overlap (2.5 seconds for 10-second chunks)
            let overlapRatioStr = ProcessInfo.processInfo.environment["OVERLAP_RATIO"] ?? "0.25"
            let overlapRatio = Double(overlapRatioStr) ?? 0.25

            // Calculate overlap in samples (16kHz sample rate)
            self.overlapSize = Int(Double(chunkSize) * overlapRatio)
            self.stepSize = chunkSize - overlapSize

            if enableDebug {
                print("Sliding window enabled: chunk=\(chunkSize), overlap=\(overlapSize), step=\(stepSize)")
                print("Overlap ratio: \(overlapRatio) (\(Double(overlapSize)/16000.0) seconds)")
            }
        } else {
            // No overlap, traditional chunking
            self.overlapSize = 0
            self.stepSize = chunkSize
        }
    }

    func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        let audioLength = Double(audioSamples.count) / 16000.0

        // For audio shorter than chunk size, process directly

        var allTokens: [Int] = []
        var position = 0
        var chunkIndex = 0
        var decoderState = try DecoderState()
        var previousChunkTokens: [Int] = []
        var overlapTokenCount = 0

        while position < audioSamples.count {
            let (chunkTokens, newDecoderState) = try await processChunk(
                at: position, chunkIndex: chunkIndex, using: manager, decoderState: decoderState)

            // Handle overlapping tokens when using sliding window
            if overlapSize > 0 && chunkIndex > 0 {
                // Estimate how many tokens to skip from the beginning of the current chunk
                // This is approximate - ideally we'd track exact token positions
                let overlapRatio = Double(overlapSize) / Double(chunkSize)
                let estimatedOverlapTokens = Int(Double(chunkTokens.count) * overlapRatio * 0.8)  // Conservative estimate

                // Skip the overlapping tokens at the beginning of this chunk
                let tokensToAdd = Array(chunkTokens.dropFirst(max(0, estimatedOverlapTokens)))

                if enableDebug {
                    print(
                        "Chunk \(chunkIndex): \(chunkTokens.count) tokens, skipping first \(estimatedOverlapTokens), adding \(tokensToAdd.count)"
                    )
                }

                allTokens.append(contentsOf: tokensToAdd)
            } else {
                // First chunk or no overlap - add all tokens
                allTokens.append(contentsOf: chunkTokens)
            }

            previousChunkTokens = chunkTokens
            decoderState = newDecoderState  // Maintain decoder state across chunks
            position += stepSize
            chunkIndex += 1
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
}
