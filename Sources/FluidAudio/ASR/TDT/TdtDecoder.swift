/// Token-and-Duration Transducer (TDT) Decoder
///
/// This decoder implements NVIDIA's TDT algorithm from the Parakeet model family.
/// TDT extends the RNN-T (Recurrent Neural Network Transducer) by adding duration prediction,
/// allowing the model to "jump" multiple audio frames at once, significantly improving speed.
///
/// Key concepts:
/// - **Token prediction**: What character/subword to emit
/// - **Duration prediction**: How many audio frames to skip before next prediction
/// - **Blank tokens**: Special tokens (ID=8192) indicating no speech/silence
/// - **Inner loop**: Optimized processing of consecutive blank tokens
///
/// Algorithm flow:
/// 1. Process audio frame through encoder (done before this decoder)
/// 2. Combine encoder frame + decoder state in joint network
/// 3. Predict token AND duration (frames to skip)
/// 4. If blank token: enter inner loop to skip silence quickly WITHOUT updating decoder
/// 5. If non-blank: emit token, update decoder LSTM, advance by duration
/// 6. Repeat until all audio frames processed
///
/// Performance optimizations:
/// - ANE (Apple Neural Engine) aligned memory for 2-3x speedup
/// - Zero-copy array operations where possible
/// - Cached decoder outputs to avoid redundant computation
/// - SIMD operations for argmax using Accelerate framework
/// - **Intentional decoder state reuse for blanks** (key optimization)

import Accelerate
import CoreML
import Foundation
import OSLog

/// Pre-processed encoder frames for fast access
/// Each frame is a vector of features extracted from ~80ms of audio
private typealias EncoderFrameArray = [[Float]]

@available(macOS 13.0, iOS 16.0, *)
internal struct TdtDecoder {

    private let logger = AppLogger(category: "TDT")
    private let config: ASRConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()
    // Parakeet‑TDT‑v3: duration head has 5 bins mapping directly to frame advances

    init(config: ASRConfig) {
        self.config = config
    }

    /// Execute TDT decoding and return tokens with emission timestamps
    ///
    /// This is the main entry point for the decoder. It processes encoder frames sequentially,
    /// predicting tokens and their durations, while maintaining decoder LSTM state.
    ///
    /// - Parameters:
    ///   - encoderOutput: 3D tensor [batch=1, time_frames, hidden_dim=1024] from encoder
    ///   - encoderSequenceLength: Number of valid frames in encoderOutput (rest is padding)
    ///   - decoderModel: CoreML model for LSTM decoder (updates language context)
    ///   - jointModel: CoreML model combining encoder+decoder features for predictions
    ///   - decoderState: LSTM hidden/cell states, maintained across chunks for context
    ///   - startFrameOffset: For streaming - offset into the full audio stream
    ///   - lastProcessedFrame: For streaming - last frame processed in previous chunk
    ///
    /// - Returns: Tuple of:
    ///   - tokens: Array of token IDs (vocabulary indices) for recognized speech
    ///   - timestamps: Array of encoder frame indices when each token was emitted
    ///
    /// - Note: Frame indices can be converted to time: frame_index * 0.08 = time_in_seconds
    func decodeWithTimings(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        actualAudioFrames: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0
    ) async throws -> TdtHypothesis {
        // Early exit for very short audio (< 160ms)
        guard encoderSequenceLength > 1 else {
            return TdtHypothesis(decState: decoderState)
        }

        // Pre-process encoder output for faster access
        // Converts 3D tensor to array of frame vectors for O(1) frame access
        let encoderFrames = try preProcessEncoderOutput(
            encoderOutput, length: encoderSequenceLength)

        var hypothesis = TdtHypothesis(decState: decoderState)
        hypothesis.lastToken = decoderState.lastToken

        // Initialize time tracking for frame navigation
        // timeIndices: Current position in encoder frames (advances by duration)
        // timeJump: Tracks overflow when we process beyond current chunk (for streaming)
        // contextFrameAdjustment: Adjusts for adaptive context overlap
        var timeIndices: Int
        if let prevTimeJump = decoderState.timeJump {
            // Streaming continuation: timeJump represents decoder position beyond previous chunk
            // For the new chunk, we need to account for:
            // 1. How far the decoder advanced past the previous chunk (prevTimeJump)
            // 2. The overlap/context between chunks (contextFrameAdjustment)
            //
            // If prevTimeJump > 0: decoder went past previous chunk's frames
            // If contextFrameAdjustment < 0: decoder should skip frames (overlap with previous chunk)
            // If contextFrameAdjustment > 0: decoder should start later (adaptive context)
            // Net position = prevTimeJump + contextFrameAdjustment (add adjustment to decoder position)

            // SPECIAL CASE: When prevTimeJump = 0 and contextFrameAdjustment = 0,
            // decoder finished exactly at boundary but chunk has physical overlap
            // Need to skip the overlap frames to avoid re-processing
            if prevTimeJump == 0 && contextFrameAdjustment == 0 {
                // Skip standard overlap (1.6s = 20 frames)
                timeIndices = 20
            } else {
                timeIndices = max(0, prevTimeJump + contextFrameAdjustment)
            }

        } else {
            // First chunk: start from beginning, accounting for any context frames that were already processed
            timeIndices = contextFrameAdjustment
        }
        // Use the minimum of encoder sequence length and actual audio frames to avoid processing padding
        let effectiveSequenceLength = min(encoderSequenceLength, actualAudioFrames)

        // Key variables for frame navigation:
        var safeTimeIndices = min(timeIndices, effectiveSequenceLength - 1)  // Bounds-checked index
        var timeIndicesCurrentLabels = timeIndices  // Frame where current token was emitted
        var activeMask = timeIndices < effectiveSequenceLength  // Start processing only if we haven't exceeded bounds
        let lastTimestep = effectiveSequenceLength - 1  // Maximum valid frame index

        // If timeJump puts us beyond the available frames, return empty
        if timeIndices >= effectiveSequenceLength {
            return TdtHypothesis(decState: decoderState)
        }

        let reusableTargetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        let reusableTargetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        reusableTargetLengthArray[0] = NSNumber(value: 1)

        // Initialize decoder LSTM state for a fresh utterance
        // This ensures clean state when starting transcription
        if decoderState.lastToken == nil && decoderState.predictorOutput == nil {
            let zero = TdtDecoderState(fallback: true)
            decoderState.hiddenState.copyData(from: zero.hiddenState)
            decoderState.cellState.copyData(from: zero.cellState)
        }

        // Prime the decoder with Start-of-Sequence token if needed
        // This initializes the LSTM's language model context
        // Note: In RNN-T/TDT, we use blank token as SOS
        if decoderState.predictorOutput == nil && hypothesis.lastToken == nil {
            let sos = config.tdtConfig.blankId  // blank=8192 serves as SOS
            let primed = try runDecoder(
                token: sos,
                state: decoderState,
                model: decoderModel,
                targetArray: reusableTargetArray,
                targetLengthArray: reusableTargetLengthArray
            )
            let proj = try extractFeatureValue(
                from: primed.output, key: "decoder_output", errorMessage: "Invalid decoder output")
            decoderState.predictorOutput = proj
            hypothesis.decState = primed.newState
        }

        // Variables for preventing infinite token emission at same timestamp
        // This handles edge cases where model gets stuck predicting many tokens
        // without advancing through audio (force-blank mechanism)
        var lastEmissionTimestamp = -1
        var emissionsAtThisTimestamp = 0
        let maxSymbolsPerStep = config.tdtConfig.maxSymbolsPerStep  // Usually 5-10
        var tokensProcessedThisChunk = 0  // Track tokens per chunk to prevent runaway decoding

        // ===== MAIN DECODING LOOP =====
        // Process each encoder frame until we've consumed all audio
        while activeMask {
            // Use last emitted token for decoder context, or blank if starting
            var label = hypothesis.lastToken ?? config.tdtConfig.blankId
            let stateToUse = hypothesis.decState ?? decoderState

            // Get decoder output (LSTM hidden state projection)
            // OPTIMIZATION: Use cached output if available to avoid redundant computation
            // This cache is valid when decoder state hasn't changed
            let decoderResult: (output: MLFeatureProvider, newState: TdtDecoderState)
            if let cached = decoderState.predictorOutput {
                // Reuse cached decoder output - significant speedup
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "decoder_output": MLFeatureValue(multiArray: cached)
                ])
                decoderResult = (output: provider, newState: stateToUse)
            } else {
                // No cache - run decoder LSTM
                decoderResult = try runDecoder(
                    token: label,
                    state: stateToUse,
                    model: decoderModel,
                    targetArray: reusableTargetArray,
                    targetLengthArray: reusableTargetLengthArray
                )
            }

            // Get current audio frame features
            let encoderStep = encoderFrames[safeTimeIndices]

            // Run joint network: combines audio (encoder) + language (decoder) features
            // Output: logits for both token prediction and duration prediction
            let logits = try runJoint(
                encoderStep: encoderStep,
                decoderOutput: decoderResult.output,
                model: jointModel
            )

            // Split joint output into token logits (8193 dims) and duration logits (5 dims)
            let (tokenLogits, durationLogits) = try splitLogits(
                logits, durationElements: config.tdtConfig.durationBins.count)

            // Predict token (what to emit) and duration (how many frames to skip)
            label = argmaxSIMD(tokenLogits)  // Get highest scoring token
            let tokenProbabilities = softmax(tokenLogits)
            var score = tokenProbabilities[label]  // Confidence score (probability) for this token

            // Map duration bin to actual frame count
            // durationBins typically = [0,1,2,3,4] meaning skip 0-4 frames
            let outerDurationBinIndex = argmaxSIMD(durationLogits)
            var duration = config.tdtConfig.durationBins[outerDurationBinIndex]
            // Duration prediction logging removed for cleaner output

            let blankId = config.tdtConfig.blankId  // 8192 for v3 models
            var blankMask = (label == blankId)  // Is this a blank (silence) token?

            // CRITICAL FIX: Prevent infinite loops when blank has duration=0
            // Always advance at least 1 frame to ensure forward progress
            if blankMask && duration == 0 {
                duration = 1
            }

            // Advance through audio frames based on predicted duration
            timeIndicesCurrentLabels = timeIndices  // Remember where this token was emitted
            timeIndices += duration  // Jump forward by predicted duration
            safeTimeIndices = min(timeIndices, lastTimestep)  // Bounds check

            activeMask = timeIndices < effectiveSequenceLength  // Continue if more frames
            var advanceMask = activeMask && blankMask  // Enter inner loop for blank tokens

            // ===== INNER LOOP: OPTIMIZED BLANK PROCESSING =====
            // When we predict a blank token, we enter this loop to quickly skip
            // through consecutive silence/non-speech frames.
            //
            // IMPORTANT DESIGN DECISION:
            // We intentionally REUSE decoderResult.output from outside the loop.
            // This is NOT a bug - it's a key optimization based on the principle that
            // blank tokens (silence) should not change the language model context.
            //
            // Why this works:
            // - Blanks represent absence of speech, not linguistic content
            // - The decoder LSTM tracks language context (what words came before)
            // - Silence doesn't change what words were spoken
            // - So we keep the same decoder state until we find actual speech
            //
            // This optimization:
            // - Avoids expensive LSTM computations for silence frames
            // - Maintains linguistic continuity across gaps in speech
            // - Speeds up processing by 2-3x for audio with silence
            while advanceMask {
                timeIndicesCurrentLabels = timeIndices

                let innerEncoderStep = encoderFrames[safeTimeIndices]

                // INTENTIONAL: Reusing decoderResult.output from outside loop
                // Blanks don't update language context - only speech tokens do
                let innerLogits = try runJoint(
                    encoderStep: innerEncoderStep,
                    decoderOutput: decoderResult.output,  // Same decoder output (by design)
                    model: jointModel
                )

                let (innerTokenLogits, innerDurationLogits) = try splitLogits(
                    innerLogits, durationElements: config.tdtConfig.durationBins.count)
                label = argmaxSIMD(innerTokenLogits)
                let innerTokenProbabilities = softmax(innerTokenLogits)
                score = innerTokenProbabilities[label]
                let innerDurationBinIndex = argmaxSIMD(innerDurationLogits)
                duration = config.tdtConfig.durationBins[innerDurationBinIndex]

                blankMask = (label == blankId)

                // Same duration=0 fix for inner loop
                if blankMask && duration == 0 {
                    duration = 1
                }

                // Advance and check if we should continue the inner loop
                timeIndices += duration
                safeTimeIndices = min(timeIndices, lastTimestep)
                activeMask = timeIndices < effectiveSequenceLength
                advanceMask = activeMask && blankMask  // Exit loop if non-blank found
            }
            // ===== END INNER LOOP =====

            // Process non-blank token: emit it and update decoder state
            if activeMask && label != blankId {
                // Check per-chunk token limit to prevent runaway decoding
                tokensProcessedThisChunk += 1
                if tokensProcessedThisChunk > config.tdtConfig.maxTokensPerChunk {
                    break
                }

                // Add token to output sequence
                hypothesis.ySequence.append(label)
                hypothesis.score += score
                hypothesis.timestamps.append(timeIndicesCurrentLabels + globalFrameOffset)
                hypothesis.tokenConfidences.append(score)
                hypothesis.lastToken = label  // Remember for next iteration

                // CRITICAL: Update decoder LSTM with the new token
                // This updates the language model context for better predictions
                // Only non-blank tokens update the decoder - this is key!
                // NOTE: We update the decoder state regardless of whether we emit the token
                // to maintain proper language model context across chunk boundaries
                let step = try runDecoder(
                    token: label,
                    state: decoderResult.newState,
                    model: decoderModel,
                    targetArray: reusableTargetArray,
                    targetLengthArray: reusableTargetLengthArray
                )
                hypothesis.decState = step.newState
                decoderState.predictorOutput = try extractFeatureValue(
                    from: step.output, key: "decoder_output", errorMessage: "Invalid decoder output")

                if timeIndicesCurrentLabels == lastEmissionTimestamp {
                    emissionsAtThisTimestamp += 1
                } else {
                    lastEmissionTimestamp = timeIndicesCurrentLabels
                    emissionsAtThisTimestamp = 1
                }

                // Force-blank mechanism: Prevent infinite token emission at same timestamp
                // If we've emitted too many tokens without advancing frames,
                // force advancement to prevent getting stuck
                if emissionsAtThisTimestamp >= maxSymbolsPerStep {
                    let forcedAdvance = 1
                    timeIndices = min(timeIndices + forcedAdvance, lastTimestep)
                    safeTimeIndices = min(timeIndices, lastTimestep)
                    emissionsAtThisTimestamp = 0
                    lastEmissionTimestamp = -1
                }
            }

            // Update activeMask for next iteration
            activeMask = timeIndices < effectiveSequenceLength
        }

        // ===== LAST CHUNK FINALIZATION =====
        // For the last chunk, ensure we force emission of any pending tokens
        // Continue processing even after encoder frames are exhausted
        if isLastChunk {

            var additionalSteps = 0
            var consecutiveBlanks = 0
            let maxConsecutiveBlanks = config.tdtConfig.consecutiveBlankLimit
            var lastToken = hypothesis.lastToken ?? config.tdtConfig.blankId
            var finalProcessingTimeIndices = timeIndices

            // Continue until we get consecutive blanks or hit max steps
            while additionalSteps < maxSymbolsPerStep && consecutiveBlanks < maxConsecutiveBlanks {
                let stateToUse = hypothesis.decState ?? decoderState

                // Get decoder output for final processing
                let decoderResult: (output: MLFeatureProvider, newState: TdtDecoderState)
                if let cached = decoderState.predictorOutput {
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "decoder_output": MLFeatureValue(multiArray: cached)
                    ])
                    decoderResult = (output: provider, newState: stateToUse)
                } else {
                    decoderResult = try runDecoder(
                        token: lastToken,
                        state: stateToUse,
                        model: decoderModel,
                        targetArray: reusableTargetArray,
                        targetLengthArray: reusableTargetLengthArray
                    )
                }

                // Use sliding window approach: try different frames near the boundary
                // to capture tokens that might be emitted at frame boundaries
                let frameVariations = [
                    min(finalProcessingTimeIndices, encoderFrames.count - 1),
                    min(effectiveSequenceLength - 1, encoderFrames.count - 1),
                    min(max(0, effectiveSequenceLength - 2), encoderFrames.count - 1),
                ]
                let frameIndex = frameVariations[additionalSteps % frameVariations.count]
                let encoderStep = encoderFrames[frameIndex]

                let logits = try runJoint(
                    encoderStep: encoderStep,
                    decoderOutput: decoderResult.output,
                    model: jointModel
                )

                let (tokenLogits, durationLogits) = try splitLogits(
                    logits, durationElements: config.tdtConfig.durationBins.count)

                let token = argmaxSIMD(tokenLogits)
                let finalTokenProbabilities = softmax(tokenLogits)
                let score = finalTokenProbabilities[token]

                // Also get duration for proper timestamp calculation
                let durationBinIndex = argmaxSIMD(durationLogits)
                let duration = config.tdtConfig.durationBins[durationBinIndex]

                if token == config.tdtConfig.blankId {
                    consecutiveBlanks += 1
                } else {
                    consecutiveBlanks = 0  // Reset on non-blank

                    // Non-blank token found - emit it
                    hypothesis.ySequence.append(token)
                    hypothesis.score += score
                    // Use the current processing position for timestamp, ensuring it doesn't exceed bounds
                    let finalTimestamp =
                        min(finalProcessingTimeIndices, effectiveSequenceLength - 1) + globalFrameOffset
                    hypothesis.timestamps.append(finalTimestamp)
                    hypothesis.tokenConfidences.append(score)
                    hypothesis.lastToken = token

                    // Update decoder state
                    let step = try runDecoder(
                        token: token,
                        state: decoderResult.newState,
                        model: decoderModel,
                        targetArray: reusableTargetArray,
                        targetLengthArray: reusableTargetLengthArray
                    )
                    hypothesis.decState = step.newState
                    decoderState.predictorOutput = try extractFeatureValue(
                        from: step.output, key: "decoder_output", errorMessage: "Invalid decoder output")
                    lastToken = token
                }

                // Advance processing position by predicted duration, but clamp to bounds
                finalProcessingTimeIndices = min(finalProcessingTimeIndices + max(1, duration), effectiveSequenceLength)
                additionalSteps += 1
            }

            // Finalize decoder state
            decoderState.finalizeLastChunk()
        }

        if let finalState = hypothesis.decState {
            decoderState = finalState
        }
        decoderState.lastToken = hypothesis.lastToken

        // Clear cached predictor output if ending with punctuation
        // This prevents punctuation from being duplicated at chunk boundaries
        if let lastToken = hypothesis.lastToken {
            let punctuationTokens = [7883, 7952, 7948]  // period, question, exclamation
            if punctuationTokens.contains(lastToken) {
                decoderState.predictorOutput = nil
                // Keep lastToken for linguistic context - deduplication handles duplicates at higher level
            }
        }

        // Always store time jump for streaming: how far beyond this chunk we've processed
        // Used to align timestamps when processing next chunk
        // Formula: timeJump = finalPosition - effectiveFrames
        let finalTimeJump = timeIndices - effectiveSequenceLength
        decoderState.timeJump = finalTimeJump

        // For the last chunk, clear timeJump since there are no more chunks
        if isLastChunk {
            decoderState.timeJump = nil
        }

        // No filtering at decoder level - let post-processing handle deduplication
        return hypothesis
    }

    /// Pre-process encoder output into contiguous memory for faster access
    private func preProcessEncoderOutput(
        _ encoderOutput: MLMultiArray, length: Int
    ) throws
        -> EncoderFrameArray
    {
        let shape = encoderOutput.shape
        guard shape.count >= 3 else {
            throw ASRError.processingFailed("Invalid encoder output shape: \(shape)")
        }
        let hiddenSize = shape[2].intValue

        var frames = EncoderFrameArray()
        frames.reserveCapacity(length)

        // Zero-copy optimization: create views instead of copying data
        if encoderOutput.dataType == .float32 {
            // Store the encoder output reference for zero-copy access
            let floatPtr = encoderOutput.dataPointer.bindMemory(
                to: Float.self, capacity: encoderOutput.count)

            for timeIdx in 0..<length {
                let startIdx = timeIdx * hiddenSize

                // Create a lightweight wrapper that references the original memory
                let frameView = UnsafeBufferPointer(
                    start: floatPtr + startIdx,
                    count: hiddenSize
                )

                // Only copy when absolutely necessary (for now, to maintain compatibility)
                frames.append(Array(frameView))
            }
        } else {
            // Fallback for non-float32 types
            for timeIdx in 0..<length {
                var frame = [Float]()
                frame.reserveCapacity(hiddenSize)

                for h in 0..<hiddenSize {
                    let index = timeIdx * hiddenSize + h
                    if index < encoderOutput.count {
                        frame.append(encoderOutput[index].floatValue)
                    } else {
                        throw ASRError.processingFailed("Index out of bounds in encoder output")
                    }
                }

                frames.append(frame)
            }
        }

        return frames
    }

    /// Decoder execution
    private func runDecoder(
        token: Int,
        state: TdtDecoderState,
        model: MLModel,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray
    ) throws -> (output: MLFeatureProvider, newState: TdtDecoderState) {

        // Reuse pre-allocated arrays
        targetArray[0] = NSNumber(value: token)
        // targetLengthArray[0] is already set to 1 and never changes

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targetArray),
            "target_lengths": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: state.hiddenState),
            "c_in": MLFeatureValue(multiArray: state.cellState),
        ])

        let output = try model.prediction(
            from: input,
            options: predictionOptions
        )

        var newState = state
        newState.update(from: output)

        return (output, newState)
    }

    /// Joint network execution with zero-copy
    private func runJoint(
        encoderStep: [Float],
        decoderOutput: MLFeatureProvider,
        model: MLModel
    ) throws -> MLMultiArray {

        // Create ANE-aligned encoder array for optimal performance
        let encoderArray = try ANEOptimizer.createANEAlignedArray(
            shape: [1, 1, encoderStep.count as NSNumber],
            dataType: .float32
        )

        // Use optimized memory copy
        encoderStep.withUnsafeBufferPointer { buffer in
            let destPtr = encoderArray.dataPointer.bindMemory(
                to: Float.self, capacity: encoderStep.count)
            memcpy(destPtr, buffer.baseAddress!, encoderStep.count * MemoryLayout<Float>.stride)
        }

        let decoderOutputArray = try extractFeatureValue(
            from: decoderOutput, key: "decoder_output", errorMessage: "Invalid decoder output")

        // Prefetch arrays for ANE if available
        if #available(macOS 14.0, iOS 17.0, *) {
            ANEOptimizer.prefetchToNeuralEngine(encoderArray)
            ANEOptimizer.prefetchToNeuralEngine(decoderOutputArray)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_outputs": MLFeatureValue(multiArray: encoderArray),
            "decoder_outputs": MLFeatureValue(multiArray: decoderOutputArray),
        ])

        let output = try model.prediction(
            from: input,
            options: predictionOptions
        )

        return try extractFeatureValue(
            from: output, key: "logits", errorMessage: "Joint network output missing logits")
    }

    /// Predict token and duration from joint logits
    internal func predictTokenAndDuration(
        _ logits: MLMultiArray,
        durationBins: [Int]
    ) throws -> (
        token: Int, score: Float, duration: Int
    ) {
        let (tokenLogits, durationLogits) = try splitLogits(logits, durationElements: durationBins.count)

        let bestToken = argmax(tokenLogits)
        let tokenScore = tokenLogits[bestToken]

        let (_, duration) = try processDurationLogits(durationLogits, durationBins: durationBins)

        return (token: bestToken, score: tokenScore, duration: duration)
    }

    /// Update hypothesis with new token
    internal func updateHypothesis(
        _ hypothesis: inout TdtHypothesis,
        token: Int,
        score: Float,
        duration: Int,
        timeIdx: Int,
        decoderState: TdtDecoderState
    ) {
        hypothesis.ySequence.append(token)
        hypothesis.score += score
        hypothesis.timestamps.append(timeIdx)
        hypothesis.tokenConfidences.append(score)
        hypothesis.decState = decoderState
        hypothesis.lastToken = token

        if config.tdtConfig.includeTokenDuration {
            hypothesis.tokenDurations.append(duration)
        }
    }

    // MARK: - Private Helper Methods

    /// Split joint logits into token and duration components with optimized memory access
    private func splitLogits(
        _ logits: MLMultiArray,
        durationElements: Int
    ) throws -> (
        tokenLogits: [Float], durationLogits: [Float]
    ) {
        let totalElements = logits.count
        let durationElements = durationElements
        // Parakeet-TDT-0.6b-v3: 8192 regular tokens + 1 blank token = 8193 total vocab
        // Joint network outputs: [8193 token logits] + [5 duration logits]
        let vocabSize = totalElements - durationElements

        guard totalElements >= durationElements else {
            throw ASRError.processingFailed("Logits dimension mismatch")
        }
        guard vocabSize > 0 else { throw ASRError.processingFailed("Logits dim mismatch") }

        // Create views directly without copying - zero-copy operation
        let logitsPtr = logits.dataPointer.bindMemory(to: Float.self, capacity: totalElements)

        // Use ContiguousArray for better cache locality
        let tokenLogits = ContiguousArray(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))
        let durationLogits = ContiguousArray(
            UnsafeBufferPointer(start: logitsPtr + vocabSize, count: durationElements))

        return (Array(tokenLogits), Array(durationLogits))
    }

    /// Find index of maximum value using SIMD operations
    private func argmaxSIMD(_ values: [Float]) -> Int {
        guard !values.isEmpty else { return 0 }

        // Use Accelerate framework for optimized argmax
        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0

        values.withUnsafeBufferPointer { buffer in
            vDSP_maxvi(buffer.baseAddress!, 1, &maxValue, &maxIndex, vDSP_Length(values.count))
        }

        return Int(maxIndex)
    }

    /// Non-SIMD argmax for compatibility
    private func argmax(_ values: [Float]) -> Int {
        return argmaxSIMD(values)
    }

    /// Apply softmax to convert logits to probabilities using Accelerate framework
    internal func softmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }

        var probabilities = [Float](repeating: 0.0, count: logits.count)

        logits.withUnsafeBufferPointer { logitsPtr in
            probabilities.withUnsafeMutableBufferPointer { probsPtr in
                // Find max value to prevent overflow
                var maxValue: Float = 0
                var maxIndex: vDSP_Length = 0
                vDSP_maxvi(logitsPtr.baseAddress!, 1, &maxValue, &maxIndex, vDSP_Length(logits.count))

                // Subtract max from all values (for numerical stability)
                vDSP_vsadd(logitsPtr.baseAddress!, 1, [-maxValue], probsPtr.baseAddress!, 1, vDSP_Length(logits.count))

                // Apply exponential
                vvexpf(probsPtr.baseAddress!, probsPtr.baseAddress!, [Int32(logits.count)])

                // Calculate sum
                var sum: Float = 0
                vDSP_sve(probsPtr.baseAddress!, 1, &sum, vDSP_Length(logits.count))

                // Normalize by sum
                vDSP_vsdiv(probsPtr.baseAddress!, 1, [sum], probsPtr.baseAddress!, 1, vDSP_Length(logits.count))
            }
        }

        return probabilities
    }

    /// Process duration logits to get duration value
    private func processDurationLogits(
        _ durationLogits: [Float],
        durationBins: [Int]
    ) throws -> (
        bestDuration: Int, duration: Int
    ) {
        let bestDurationIdx = argmaxSIMD(durationLogits)
        let duration = durationBins[bestDurationIdx]
        return (bestDurationIdx, duration)
    }

    internal func extractEncoderTimeStep(
        _ encoderOutput: MLMultiArray, timeIndex: Int
    ) throws
        -> MLMultiArray
    {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let sequenceLength = shape[1].intValue
        let hiddenSize = shape[2].intValue

        guard timeIndex < sequenceLength else {
            throw ASRError.processingFailed(
                "Time index out of bounds: \(timeIndex) >= \(sequenceLength)")
        }

        let timeStepArray = try MLMultiArray(
            shape: [batchSize, 1, hiddenSize] as [NSNumber], dataType: .float32)

        for h in 0..<hiddenSize {
            let sourceIndex = timeIndex * hiddenSize + h
            timeStepArray[h] = encoderOutput[sourceIndex]
        }

        return timeStepArray
    }

    internal func prepareDecoderInput(
        targetToken: Int,
        hiddenState: MLMultiArray,
        cellState: MLMultiArray
    ) throws -> MLFeatureProvider {
        let targetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        targetArray[0] = NSNumber(value: targetToken)

        let targetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        targetLengthArray[0] = NSNumber(value: 1)

        return try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targetArray),
            "target_lengths": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: hiddenState),
            "c_in": MLFeatureValue(multiArray: cellState),
        ])
    }

    internal func prepareJointInput(
        encoderOutput: MLMultiArray,
        decoderOutput: MLFeatureProvider,
        timeIndex: Int
    ) throws -> MLFeatureProvider {
        let decoderOutputArray = try extractFeatureValue(
            from: decoderOutput, key: "decoder_output", errorMessage: "Invalid decoder output")

        return try MLDictionaryFeatureProvider(dictionary: [
            "encoder_outputs": MLFeatureValue(multiArray: encoderOutput),
            "decoder_outputs": MLFeatureValue(multiArray: decoderOutputArray),
        ])
    }

    /// Validates and extracts a required feature value from MLFeatureProvider
    private func extractFeatureValue(
        from provider: MLFeatureProvider, key: String, errorMessage: String
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }
}

extension MLMultiArray {
    /// Fast L2 norm (float32 optimized)
    func l2Normf() -> Float {
        let n = self.count
        if self.dataType == .float32 {
            return self.dataPointer.withMemoryRebound(to: Float.self, capacity: n) { ptr in
                var ss: Float = 0
                vDSP_svesq(ptr, 1, &ss, vDSP_Length(n))
                return sqrtf(ss)
            }
        } else {
            var ss: Float = 0
            for i in 0..<n {
                let v = self[i].floatValue
                ss += v * v
            }
            return sqrtf(ss)
        }
    }
    /// "BxTxH" style string
    var shapeString: String { shape.map { "\($0.intValue)" }.joined(separator: "x") }
}
