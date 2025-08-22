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
        ).process(using: self, startTime: startTime, decoderState: &decoderState)

        // ChunkProcessor now updates the passed-in decoder state for continuity
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

        if enableDebug {
            print("  Encoder output: sequence_length=\(encoderSequenceLength), shape=\(encoderHiddenStates.shape)")
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
        // Enable sliding window by default for better accuracy
        // Can be disabled with USE_SLIDING_WINDOW=0
        let slidingWindowEnabled = ProcessInfo.processInfo.environment["USE_SLIDING_WINDOW"] != "0"

        if slidingWindowEnabled && audioSamples.count > chunkSize {
            // Default to 10% overlap (1 second for 10-second chunks)
            // Optimal ratio that outperforms NeMo Python while avoiding artifacts
            let overlapRatioStr = ProcessInfo.processInfo.environment["OVERLAP_RATIO"] ?? "0.1"
            let overlapRatio = min(0.5, max(0.0, Double(overlapRatioStr) ?? 0.1))

            // Calculate overlap in samples (16kHz sample rate)
            self.overlapSize = Int(Double(chunkSize) * overlapRatio)
            self.stepSize = chunkSize - overlapSize

            if enableDebug {
                print("Sliding window ENABLED: chunk=\(chunkSize), overlap=\(overlapSize), step=\(stepSize)")
                print(
                    "Overlap: \(overlapRatio * 100)% (\(Double(overlapSize)/16000.0)s overlap, \(Double(stepSize)/16000.0)s step)"
                )
            }
        } else {
            // No overlap for short audio or if explicitly disabled
            self.overlapSize = 0
            self.stepSize = chunkSize

            if enableDebug && audioSamples.count > chunkSize {
                print("Sliding window DISABLED: processing in \(chunkSize)-sample chunks")
            }
        }
    }

    func process(using manager: AsrManager, startTime: Date, decoderState: inout DecoderState) async throws -> ASRResult
    {
        let audioLength = Double(audioSamples.count) / 16000.0

        // For audio shorter than chunk size, process directly

        var allTokens: [Int] = []
        var position = 0
        var chunkIndex = 0
        // Use the passed-in decoder state for continuity
        var workingDecoderState = decoderState
        var previousChunkTokens: [Int] = []
        var overlapTokenCount = 0

        while position < audioSamples.count {
            let chunkTokens = try await processChunk(
                at: position, chunkIndex: chunkIndex, using: manager, decoderState: &workingDecoderState)

            // Handle overlapping tokens when using sliding window
            if overlapSize > 0 && chunkIndex > 0 {
                // Smart token deduplication for overlapping chunks
                // Compare the end of previous tokens with start of current tokens
                let overlapRatio = Double(overlapSize) / Double(chunkSize)

                // Find where the new unique content starts
                var skipCount = 0
                if !previousChunkTokens.isEmpty && !chunkTokens.isEmpty {
                    // Look for the longest matching suffix/prefix between chunks
                    let searchWindow = min(previousChunkTokens.count, chunkTokens.count, 20)

                    for windowSize in (1...searchWindow).reversed() {
                        let suffix = Array(previousChunkTokens.suffix(windowSize))
                        if chunkTokens.starts(with: suffix) {
                            skipCount = windowSize
                            break
                        }
                    }

                    // If no exact match found, DON'T skip any tokens
                    // Better to have some duplication than to lose content
                    // The tokenizer will handle repeated words in post-processing
                }

                let tokensToAdd = Array(chunkTokens.dropFirst(skipCount))

                if enableDebug {
                    print(
                        "Chunk \(chunkIndex): \(chunkTokens.count) tokens, overlap detected: \(skipCount) tokens, adding: \(tokensToAdd.count) new tokens"
                    )
                    if chunkIndex == 1 {
                        // Debug the first overlap
                        print("  Previous chunk ended with: \(previousChunkTokens.suffix(10))")
                        print("  Current chunk starts with: \(chunkTokens.prefix(10))")
                        print("  Skipping \(skipCount) tokens")
                    }
                }

                allTokens.append(contentsOf: tokensToAdd)
            } else {
                // First chunk or no overlap - add all tokens
                allTokens.append(contentsOf: chunkTokens)

                if enableDebug && chunkIndex == 0 {
                    print("Chunk 0: Adding all \(chunkTokens.count) tokens")
                }
            }

            previousChunkTokens = chunkTokens
            // Decoder state is now updated directly via inout parameter
            position += stepSize
            chunkIndex += 1
        }

        let (text, _) = manager.convertTokensWithExistingTimings(allTokens, timings: [])

        // Update the passed-in decoder state for next call
        decoderState = workingDecoderState

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
        decoderState: inout DecoderState
    ) async throws -> [Int] {
        if enableDebug {
            print("DEBUG: processChunk \(chunkIndex) - decoderState.lastToken = \(decoderState.lastToken ?? -1)")
        }
        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)

        if enableDebug {
            print(
                "  Audio chunk: position=\(position), endPosition=\(endPosition), samples=\(chunkSamples.count), padded=\(paddedChunk.count)"
            )
            // Check if audio has content
            let maxAmplitude = chunkSamples.map { abs($0) }.max() ?? 0
            print("  Max amplitude: \(maxAmplitude)")
        }

        let (tokenIds, _) = try await manager.executeMLInference(
            paddedChunk, originalLength: chunkSamples.count, enableDebug: enableDebug, decoderState: &decoderState)

        if enableDebug && chunkIndex > 0 {
            print(
                "DEBUG: Chunk \(chunkIndex) state - lastToken: \(decoderState.lastToken ?? -1), tokens generated: \(tokenIds.count)"
            )
        }

        return tokenIds
    }
}
