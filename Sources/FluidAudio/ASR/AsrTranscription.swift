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

    // Punctuation tokens for smarter boundary matching
    private let punctuationTokens: Set<Int> = [
        7877,  // comma ,
        7883,  // period .
        7956,  // question mark ?
        8020,  // exclamation !
        8033,  // colon :
    ]

    init(audioSamples: [Float], chunkSize: Int, enableDebug: Bool) {
        self.audioSamples = audioSamples
        self.chunkSize = chunkSize
        self.enableDebug = enableDebug
        // Add 2 seconds overlap (32000 samples at 16kHz) to avoid boundary issues
        self.overlapSize = min(32000, chunkSize / 5)  // 20% overlap or 2 seconds, whichever is smaller
        self.stepSize = chunkSize - overlapSize
    }

    func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        let audioLength = Double(audioSamples.count) / 16000.0

        // For audio shorter than chunk size, process directly

        var allTokens: [Int] = []
        var position = 0
        var chunkIndex = 0
        var decoderState = try DecoderState()

        while position < audioSamples.count {
            let (chunkTokens, newDecoderState) = try await processChunk(
                at: position, chunkIndex: chunkIndex, using: manager, decoderState: decoderState)

            // Handle overlapping tokens from overlapping audio chunks
            if chunkIndex > 0 && !allTokens.isEmpty && !chunkTokens.isEmpty {
                // Filter trailing punctuation from previous chunk for better matching
                let cleanedAllTokens = stripTrailingPunctuation(from: allTokens)
                let cleanedChunkTokens = stripLeadingPunctuation(from: chunkTokens)

                // Skip matching if either cleaned array is empty
                if !cleanedAllTokens.isEmpty && !cleanedChunkTokens.isEmpty {
                    // With audio overlap, we need to find and remove duplicate tokens
                    // The overlap produces duplicate tokens that need to be merged

                    // Estimate how many tokens might be in the overlap region
                    // (this is approximate - actual token count depends on speech rate)
                    let overlapRatio = Float(overlapSize) / Float(chunkSize)
                    let estimatedOverlapTokens = max(1, Int(Float(cleanedChunkTokens.count) * overlapRatio * 0.5))  // Conservative estimate

                    // Look for matching sequences at the boundary (fuzzy matching ignoring punctuation)
                    var bestMatch = 0
                    let searchRange = min(
                        estimatedOverlapTokens + 10, min(cleanedAllTokens.count, cleanedChunkTokens.count))

                    if searchRange > 0 {
                        for overlapLen in 1...searchRange {
                            let tailTokens = Array(cleanedAllTokens.suffix(overlapLen))
                            let headTokens = Array(cleanedChunkTokens.prefix(overlapLen))

                            if fuzzyTokenMatch(tailTokens, headTokens) {
                                bestMatch = overlapLen
                            }
                        }
                    }

                    // Skip the overlapping tokens from the new chunk
                    if bestMatch > 0 {
                        // Use original tokens but skip the matched portion
                        let skipCount = findOriginalSkipCount(
                            cleanedTokens: Array(cleanedChunkTokens.prefix(bestMatch)),
                            originalTokens: chunkTokens
                        )
                        allTokens.append(contentsOf: Array(chunkTokens.dropFirst(skipCount)))
                        if enableDebug {
                            print(
                                "DEBUG: Chunk \(chunkIndex) - skipped \(skipCount) overlapping tokens (fuzzy matched \(bestMatch))"
                            )
                        }
                    } else {
                        // No clear overlap found, append all tokens but filter duplicate punctuation
                        let mergedTokens = smartMergeTokens(allTokens, chunkTokens)
                        allTokens = mergedTokens
                    }
                } else {
                    // If cleaned arrays are empty, just merge with smart punctuation handling
                    let mergedTokens = smartMergeTokens(allTokens, chunkTokens)
                    allTokens = mergedTokens
                }
            } else {
                allTokens.append(contentsOf: chunkTokens)
            }

            decoderState = newDecoderState  // Propagate decoder state as per NeMo's GreedyBatchedTDTInfer

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

        if enableDebug {
            print(
                "DEBUG: Chunk \(chunkIndex) - position: \(position)/\(audioSamples.count), "
                    + "tokens generated: \(tokenIds.count), " + "state: lastToken=\(workingState.lastToken ?? -1), "
                    + "decoderState propagated: \(decoderState.lastToken ?? -1) -> \(workingState.lastToken ?? -1)"
            )

            // Debug suspicious tokens
            if !tokenIds.isEmpty {
                let suspiciousTokens = tokenIds.filter { $0 > 8000 || $0 == 7883 }  // 7883 is "."
                if !suspiciousTokens.isEmpty {
                    print("  WARNING: High/suspicious token IDs found: \(suspiciousTokens)")
                }
            }
        }

        return (tokenIds, workingState)
    }

    // Helper functions for smarter token boundary matching

    private func stripTrailingPunctuation(from tokens: [Int]) -> [Int] {
        var cleaned = tokens
        while !cleaned.isEmpty && punctuationTokens.contains(cleaned.last!) {
            cleaned.removeLast()
        }
        return cleaned
    }

    private func stripLeadingPunctuation(from tokens: [Int]) -> [Int] {
        var cleaned = tokens
        while !cleaned.isEmpty && punctuationTokens.contains(cleaned.first!) {
            cleaned.removeFirst()
        }
        return cleaned
    }

    private func fuzzyTokenMatch(_ tokens1: [Int], _ tokens2: [Int]) -> Bool {
        // Strip punctuation from both sequences for comparison
        let clean1 = tokens1.filter { !punctuationTokens.contains($0) }
        let clean2 = tokens2.filter { !punctuationTokens.contains($0) }

        // If content tokens match, consider it a match
        return clean1 == clean2 && !clean1.isEmpty
    }

    private func findOriginalSkipCount(cleanedTokens: [Int], originalTokens: [Int]) -> Int {
        // Find how many original tokens correspond to the cleaned tokens
        var skipCount = 0
        var cleanedIndex = 0

        for token in originalTokens {
            if cleanedIndex >= cleanedTokens.count {
                break
            }
            skipCount += 1
            if !punctuationTokens.contains(token) {
                cleanedIndex += 1
            }
        }

        return skipCount
    }

    private func smartMergeTokens(_ existing: [Int], _ new: [Int]) -> [Int] {
        // Remove trailing punctuation from existing if new starts with punctuation
        var result = existing

        // If last token of existing and first token of new are both punctuation,
        // keep only one
        if !result.isEmpty && !new.isEmpty && punctuationTokens.contains(result.last!)
            && punctuationTokens.contains(new.first!)
        {
            // Keep the more significant punctuation (period over comma, etc.)
            let lastPunc = result.last!
            let firstPunc = new.first!

            // Priority: . > ! > ? > : > ,
            let priority: [Int: Int] = [7883: 5, 8020: 4, 7956: 3, 8033: 2, 7877: 1]

            if (priority[firstPunc] ?? 0) > (priority[lastPunc] ?? 0) {
                result.removeLast()
            }
            result.append(contentsOf: new)
        } else {
            result.append(contentsOf: new)
        }

        return result
    }
}
