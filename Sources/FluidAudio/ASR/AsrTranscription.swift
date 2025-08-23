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

        // Use overlap processor if overlap is configured
        let result: ASRResult
        if config.overlapSeconds > 0 {
            result = try await ChunkProcessorWithOverlap(
                audioSamples: audioSamples,
                chunkSize: 160_000,
                enableDebug: config.enableDebug,
                overlapSeconds: config.overlapSeconds
            ).process(using: self, startTime: startTime)
        } else {
            var processor = ChunkProcessor(
                audioSamples: audioSamples,
                chunkSize: 160_000,
                enableDebug: config.enableDebug
            )
            result = try await processor.process(using: self, startTime: startTime)
        }

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

        // Encoder_v2 already outputs in the correct format (B, T, D)
        let encoderHiddenStates = rawEncoderOutput
        let encoderSequenceLength = encoderLength[0].intValue

        // Diagnostic: Check encoder output characteristics
        // if enableDebug || ProcessInfo.processInfo.environment["CHUNK_DEBUG"] != nil {
        //     print("🔬 Encoder Analysis:")
        //     print("  - Sequence length: \(encoderSequenceLength) frames")
        //     logger.info("🔬 Encoder Analysis:")
        //     logger.info("  - Sequence length: \(encoderSequenceLength) frames")

        //     // Check if encoder output is mostly zeros (indicating silence processing)
        //     let encoderShape = encoderHiddenStates.shape
        //     var sumAbsValues: Float = 0
        //     var maxValue: Float = 0
        //     var minValue: Float = Float.infinity
        //     var zeroCount = 0
        //     let elementCount = encoderShape.reduce(1, { $0 * $1.intValue })
        //     let sampleSize = min(elementCount, 1000)

        //     for i in 0..<sampleSize {  // Sample first 1000 values
        //         let value = encoderHiddenStates[i].floatValue
        //         sumAbsValues += abs(value)
        //         maxValue = max(maxValue, abs(value))
        //         minValue = min(minValue, abs(value))
        //         if abs(value) < 0.0001 {
        //             zeroCount += 1
        //         }
        //     }

        //     let avgMagnitude = sumAbsValues / Float(sampleSize)
        //     logger.info("  - Shape: \(encoderShape)")
        //     logger.info("  - Avg magnitude: \(String(format: "%.6f", avgMagnitude))")
        //     logger.info("  - Max magnitude: \(String(format: "%.6f", maxValue))")
        //     logger.info("  - Min magnitude: \(String(format: "%.6f", minValue))")
        //     logger.info(
        //         "  - Near-zero values: \(zeroCount)/\(sampleSize) (\(String(format: "%.1f%%", Double(zeroCount)/Double(sampleSize)*100)))"
        //     )

        //     if avgMagnitude < 0.001 {
        //         print("  ❌ Encoder output is nearly zero! (avg < 0.001)")
        //         logger.error("  ❌ Encoder output is nearly zero! (avg < 0.001)")
        //     } else if avgMagnitude < 0.01 {
        //         print("  ⚠️ Encoder output has very low magnitude (avg < 0.01)")
        //         logger.warning("  ⚠️ Encoder output has very low magnitude (avg < 0.01)")
        //     } else if avgMagnitude < 0.1 {
        //         print("  ⚠️ Encoder output has low magnitude (avg < 0.1)")
        //         logger.info("  ⚠️ Encoder output has low magnitude (avg < 0.1)")
        //     } else {
        //         print("  ✅ Encoder output has normal magnitude")
        //         logger.info("  ✅ Encoder output has normal magnitude")
        //     }

        //     // Check the actual length vs expected
        //     let expectedFrames = originalLength ?? paddedAudio.count / 160  // Assuming 160 samples per frame
        //     if encoderSequenceLength < expectedFrames / 2 {
        //         logger.warning(
        //             "  ⚠️ Encoder sequence much shorter than expected (\(encoderSequenceLength) vs ~\(expectedFrames))")
        //     }
        // }

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
    var chunkDetails: [ChunkDetail] = []

    mutating func process(using manager: AsrManager, startTime: Date) async throws -> ASRResult {
        var allTexts: [String] = []
        let audioLength = Double(audioSamples.count) / 16000.0

        var position = 0
        var chunkIndex = 0
        var decoderState = try DecoderState()

        while position < audioSamples.count {
            let chunkStartTime = Double(position) / 16000.0
            let endPosition = min(position + chunkSize, audioSamples.count)
            let chunkEndTime = Double(endPosition) / 16000.0
            let chunkSamples = endPosition - position
            let paddingSamples = chunkSize - chunkSamples

            let text = try await processChunk(
                at: position, chunkIndex: chunkIndex, using: manager, decoderState: &decoderState)
            allTexts.append(text)

            // Store chunk details
            chunkDetails.append(
                ChunkDetail(
                    chunkIndex: chunkIndex,
                    startTime: chunkStartTime,
                    endTime: chunkEndTime,
                    text: text,
                    audioSamples: chunkSamples,
                    paddingSamples: paddingSamples
                ))

            position += chunkSize
            chunkIndex += 1
        }

        return ASRResult(
            text: allTexts.joined(separator: " ").trimmingCharacters(in: .whitespacesAndNewlines),
            confidence: 1.0,
            duration: audioLength,
            processingTime: Date().timeIntervalSince(startTime),
            tokenTimings: nil,
            performanceMetrics: nil,
            chunkDetails: chunkDetails
        )
    }

    private func processChunk(
        at position: Int, chunkIndex: Int, using manager: AsrManager,
        decoderState: inout DecoderState
    ) async throws -> String {
        let enableDiagnostics = manager.config.enableDebug || ProcessInfo.processInfo.environment["CHUNK_DEBUG"] != nil

        // Variables for state comparison (used when diagnostics enabled)
        var hiddenSum: Float = 0
        var cellSum: Float = 0

        if enableDiagnostics {
            print("\n========== CHUNK \(chunkIndex) PROCESSING ==========")
            print("Position: \(position) / \(audioSamples.count) samples")
            manager.logger.info("\n========== CHUNK \(chunkIndex) PROCESSING ==========")
            manager.logger.info("Position: \(position) / \(audioSamples.count) samples")
        }

        let endPosition = min(position + chunkSize, audioSamples.count)
        let chunkSamples = Array(audioSamples[position..<endPosition])
        let paddedChunk = manager.padAudioIfNeeded(chunkSamples, targetLength: chunkSize)

        // Diagnostic logging for chunk processing
        if enableDiagnostics {
            let paddingRatio = Double(chunkSize - chunkSamples.count) / Double(chunkSize) * 100
            print("📊 Chunk Stats:")
            print("  - Samples: \(chunkSamples.count) / \(chunkSize)")
            print("  - Padding: \(String(format: "%.1f", paddingRatio))%")
            print("  - Duration: \(String(format: "%.2f", Double(chunkSamples.count) / 16000.0))s")
            manager.logger.info("📊 Chunk Stats:")
            manager.logger.info("  - Samples: \(chunkSamples.count) / \(chunkSize)")
            manager.logger.info("  - Padding: \(String(format: "%.1f", paddingRatio))%")
            manager.logger.info("  - Duration: \(String(format: "%.2f", Double(chunkSamples.count) / 16000.0))s")

            // Check audio signal characteristics
            let audioEnergy = chunkSamples.reduce(0.0) { $0 + abs($1) } / Float(chunkSamples.count)
            let maxAmplitude = chunkSamples.map { abs($0) }.max() ?? 0
            let minAmplitude = chunkSamples.min() ?? 0
            let maxAmplitudeAbs = chunkSamples.max() ?? 0

            print("🎵 Audio Signal:")
            print("  - Avg energy: \(String(format: "%.6f", audioEnergy))")
            print("  - Max amplitude: \(String(format: "%.6f", maxAmplitude))")
            print("  - Range: [\(String(format: "%.6f", minAmplitude)), \(String(format: "%.6f", maxAmplitudeAbs))]")
            manager.logger.info("🎵 Audio Signal:")
            manager.logger.info("  - Avg energy: \(String(format: "%.6f", audioEnergy))")
            manager.logger.info("  - Max amplitude: \(String(format: "%.6f", maxAmplitude))")
            manager.logger.info(
                "  - Range: [\(String(format: "%.6f", minAmplitude)), \(String(format: "%.6f", maxAmplitudeAbs))]")

            // Check if it's mostly silence
            if audioEnergy < 0.0001 {
                print("  ⚠️ Very low energy - likely silence or near-silence")
                manager.logger.warning("  ⚠️ Very low energy - likely silence or near-silence")
            } else if audioEnergy < 0.001 {
                print("  ⚠️ Low energy audio signal")
                manager.logger.warning("  ⚠️ Low energy audio signal")
            }

            // Log decoder state before processing
            print("🧠 Decoder State (before):")
            hiddenSum = (0..<decoderState.hiddenState.count).reduce(Float(0)) { sum, i in
                sum + abs(decoderState.hiddenState[i].floatValue)
            }
            cellSum = (0..<decoderState.cellState.count).reduce(Float(0)) { sum, i in
                sum + abs(decoderState.cellState[i].floatValue)
            }
            print("  - Hidden state sum: \(String(format: "%.4f", hiddenSum))")
            print("  - Cell state sum: \(String(format: "%.4f", cellSum))")
            manager.logger.info("🧠 Decoder State (before):")
            manager.logger.info("  - Hidden state sum: \(String(format: "%.4f", hiddenSum))")
            manager.logger.info("  - Cell state sum: \(String(format: "%.4f", cellSum))")
        }

        // Reset decoder state between chunks if configured
        if manager.config.resetDecoderBetweenChunks && chunkIndex > 0 {
            if enableDiagnostics {
                print("🔄 Resetting decoder state for chunk \(chunkIndex)")
            }
            try await manager.initializeDecoderState(decoderState: &decoderState)
            if enableDiagnostics {
                print("  ✅ Decoder state reset complete")
                // Log state after reset
                let hiddenAfterReset = (0..<decoderState.hiddenState.count).reduce(Float(0)) { sum, i in
                    sum + abs(decoderState.hiddenState[i].floatValue)
                }
                let cellAfterReset = (0..<decoderState.cellState.count).reduce(Float(0)) { sum, i in
                    sum + abs(decoderState.cellState[i].floatValue)
                }
                print("  - Hidden after reset: \(String(format: "%.4f", hiddenAfterReset))")
                print("  - Cell after reset: \(String(format: "%.4f", cellAfterReset))")
            }
        }

        let (tokenIds, encoderSeqLen) = try await manager.executeMLInference(
            paddedChunk, originalLength: chunkSamples.count, enableDebug: enableDiagnostics, decoderState: &decoderState
        )

        // Diagnostic: Check token generation
        if enableDiagnostics {
            manager.logger.info("🔤 Token Generation:")
            manager.logger.info("  - Encoder sequence length: \(encoderSeqLen)")
            manager.logger.info("  - Tokens generated: \(tokenIds.count)")

            // Check for period tokens (7883)
            let periodCount = tokenIds.filter { $0 == 7883 }.count
            let blankCount = tokenIds.filter { $0 == 8192 }.count

            if periodCount > 0 {
                print("  ⚠️ Period tokens (.): \(periodCount) occurrences")
                manager.logger.warning("  ⚠️ Period tokens (.): \(periodCount) occurrences")
            }
            if blankCount > 0 {
                manager.logger.info("  - Blank tokens: \(blankCount)")
            }

            // Show token distribution
            if !tokenIds.isEmpty {
                let first5 = tokenIds.prefix(5).map { String($0) }.joined(separator: ", ")
                let last5 = tokenIds.suffix(5).map { String($0) }.joined(separator: ", ")
                manager.logger.info("  - First 5 tokens: [\(first5)]")
                if tokenIds.count > 5 {
                    manager.logger.info("  - Last 5 tokens: [\(last5)]")
                }

                // Count unique tokens
                let uniqueTokens = Set(tokenIds)
                manager.logger.info("  - Unique tokens: \(uniqueTokens.count)")

                // If only periods, show warning
                if uniqueTokens.count == 1 && uniqueTokens.contains(7883) {
                    print("  ❌ ONLY PERIOD TOKENS GENERATED!")
                    manager.logger.error("  ❌ ONLY PERIOD TOKENS GENERATED!")
                } else if uniqueTokens.count <= 3 {
                    print("  ⚠️ Very few unique tokens: \(uniqueTokens)")
                    manager.logger.warning("  ⚠️ Very few unique tokens: \(uniqueTokens)")
                }
            }

            // Log decoder state after processing
            manager.logger.info("🧠 Decoder State (after):")
            let hiddenSumAfter = (0..<decoderState.hiddenState.count).reduce(Float(0)) { sum, i in
                sum + abs(decoderState.hiddenState[i].floatValue)
            }
            let cellSumAfter = (0..<decoderState.cellState.count).reduce(Float(0)) { sum, i in
                sum + abs(decoderState.cellState[i].floatValue)
            }
            manager.logger.info("  - Hidden state sum: \(String(format: "%.4f", hiddenSumAfter))")
            manager.logger.info("  - Cell state sum: \(String(format: "%.4f", cellSumAfter))")

            // Check for decoder state explosion or collapse
            if hiddenSumAfter > hiddenSum * 10 {
                manager.logger.warning("  ⚠️ Decoder hidden state grew significantly (10x)")
            } else if hiddenSumAfter < hiddenSum * 0.1 && hiddenSum > 0.01 {
                manager.logger.warning("  ⚠️ Decoder hidden state collapsed (<10% of original)")
            }
        }

        let (text, _) = manager.convertTokensWithExistingTimings(tokenIds, timings: [])

        if enableDiagnostics {
            print("📝 Chunk Output:")
            print("  Text: \"\(text)\"")
            print("  Length: \(text.count) characters")
            manager.logger.info("📝 Chunk Output:")
            manager.logger.info("  Text: \"\(text)\"")
            manager.logger.info("  Length: \(text.count) characters")
            if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                manager.logger.warning("  ⚠️ Empty transcription!")
            } else if text == "." || text == ".." || text == "..." || text == "...." || text.starts(with: "....") {
                print("  ❌ Only periods generated!")
                manager.logger.error("  ❌ Only periods generated!")
            }
            manager.logger.info("=========================================\n")
        }

        return text
    }
}
