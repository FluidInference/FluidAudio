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

internal struct TdtDecoderV3 {

    /// Joint model decision for a single encoder/decoder step.
    private struct JointDecision {
        let token: Int
        let probability: Float
        let durationBin: Int
    }

    private let logger = AppLogger(category: "TDT")
    private let config: ASRConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()
    // Parakeet‑TDT‑v3: duration head has 5 bins mapping directly to frame advances

    init(config: ASRConfig) {
        self.config = config
    }

    /// Reusable input provider that holds references to preallocated
    /// encoder and decoder step tensors for the joint model.
    private final class ReusableJointInput: NSObject, MLFeatureProvider {
        let encoderStep: MLMultiArray
        let decoderStep: MLMultiArray

        init(encoderStep: MLMultiArray, decoderStep: MLMultiArray) {
            self.encoderStep = encoderStep
            self.decoderStep = decoderStep
            super.init()
        }

        var featureNames: Set<String> {
            ["encoder_step", "decoder_step"]
        }

        func featureValue(for featureName: String) -> MLFeatureValue? {
            switch featureName {
            case "encoder_step":
                return MLFeatureValue(multiArray: encoderStep)
            case "decoder_step":
                return MLFeatureValue(multiArray: decoderStep)
            default:
                return nil
            }
        }
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

        // Build a stride-aware view so we can access encoder frames without extra copies
        let encoderFrames = try EncoderFrameView(
            encoderOutput: encoderOutput, validLength: encoderSequenceLength)

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
                // Skip standard overlap (2.0s = 25 frames at 0.08s per frame)
                timeIndices = 25
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

        // Preallocate joint input tensors and a reusable provider to avoid per-step allocations.
        let encoderHidden = encoderFrames.hiddenSize
        let decoderHidden = ASRConstants.decoderHiddenSize
        let reusableEncoderStep = try ANEOptimizer.createANEAlignedArray(
            shape: [1, NSNumber(value: encoderHidden), 1],
            dataType: .float32
        )
        let reusableDecoderStep = try ANEOptimizer.createANEAlignedArray(
            shape: [1, NSNumber(value: decoderHidden), 1],
            dataType: .float32
        )
        let jointInput = ReusableJointInput(encoderStep: reusableEncoderStep, decoderStep: reusableDecoderStep)
        // Cache frequently used stride for copying encoder frames
        let encDestStride = reusableEncoderStep.strides.map { $0.intValue }[1]
        let encDestPtr = reusableEncoderStep.dataPointer.bindMemory(to: Float.self, capacity: encoderHidden)

        // Preallocate small output backings for joint outputs (token_id, token_prob, duration)
        // Joint model scalar outputs are shaped [1 x 1 x 1] in the model description
        let tokenIdBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .int32)
        let tokenProbBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .float32)
        let durationBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .int32)

        // Initialize decoder LSTM state for a fresh utterance
        // This ensures clean state when starting transcription
        if decoderState.lastToken == nil && decoderState.predictorOutput == nil {
            let zero = TdtDecoderState.make()
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
                from: primed.output, key: "decoder", errorMessage: "Invalid decoder output")
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
                    "decoder": MLFeatureValue(multiArray: cached)
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

            // Prepare decoder projection once and reuse for inner blank loop
            let decoderProjection = try extractFeatureValue(
                from: decoderResult.output, key: "decoder", errorMessage: "Invalid decoder output")
            try populatePreparedDecoderProjection(decoderProjection, into: reusableDecoderStep)

            // Run joint network with preallocated inputs
            let decision = try runJointPrepared(
                encoderFrames: encoderFrames,
                timeIndex: safeTimeIndices,
                preparedDecoderStep: reusableDecoderStep,
                model: jointModel,
                encoderStep: reusableEncoderStep,
                encoderDestPtr: encDestPtr,
                encoderDestStride: encDestStride,
                inputProvider: jointInput,
                tokenIdBacking: tokenIdBacking,
                tokenProbBacking: tokenProbBacking,
                durationBacking: durationBacking
            )

            // Predict token (what to emit) and duration (how many frames to skip)
            label = decision.token
            var score = clampProbability(decision.probability)

            // Map duration bin to actual frame count
            // durationBins typically = [0,1,2,3,4] meaning skip 0-4 frames
            var duration = try mapDurationBin(
                decision.durationBin, durationBins: config.tdtConfig.durationBins)
            // Duration prediction logging removed for cleaner output

            let blankId = config.tdtConfig.blankId  // 8192 for v3 models
            var blankMask = (label == blankId)  // Is this a blank (silence) token?

            let currentTimeIndex = timeIndices
            // Prevent repeated non-blank emissions at the same frame when duration=0.
            if !blankMask && duration == 0
                && currentTimeIndex == lastEmissionTimestamp
                && emissionsAtThisTimestamp >= 1
            {
                duration = 1
            }

            // Prevent infinite loops when blank has duration=0.
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

                // INTENTIONAL: Reusing prepared decoder step from outside loop
                let innerDecision = try runJointPrepared(
                    encoderFrames: encoderFrames,
                    timeIndex: safeTimeIndices,
                    preparedDecoderStep: reusableDecoderStep,
                    model: jointModel,
                    encoderStep: reusableEncoderStep,
                    encoderDestPtr: encDestPtr,
                    encoderDestStride: encDestStride,
                    inputProvider: jointInput,
                    tokenIdBacking: tokenIdBacking,
                    tokenProbBacking: tokenProbBacking,
                    durationBacking: durationBacking
                )

                label = innerDecision.token
                score = clampProbability(innerDecision.probability)
                duration = try mapDurationBin(
                    innerDecision.durationBin, durationBins: config.tdtConfig.durationBins)

                blankMask = (label == blankId)

                // Same duration=0 fix for inner loop.
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
                    from: step.output, key: "decoder", errorMessage: "Invalid decoder output")

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
                        "decoder": MLFeatureValue(multiArray: cached)
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
                // Prepare decoder projection into reusable buffer (if not already)
                let finalProjection = try extractFeatureValue(
                    from: decoderResult.output, key: "decoder", errorMessage: "Invalid decoder output")
                try populatePreparedDecoderProjection(finalProjection, into: reusableDecoderStep)

                let decision = try runJointPrepared(
                    encoderFrames: encoderFrames,
                    timeIndex: frameIndex,
                    preparedDecoderStep: reusableDecoderStep,
                    model: jointModel,
                    encoderStep: reusableEncoderStep,
                    encoderDestPtr: encDestPtr,
                    encoderDestStride: encDestStride,
                    inputProvider: jointInput,
                    tokenIdBacking: tokenIdBacking,
                    tokenProbBacking: tokenProbBacking,
                    durationBacking: durationBacking
                )

                let token = decision.token
                let score = clampProbability(decision.probability)

                // Also get duration for proper timestamp calculation
                let duration = try mapDurationBin(
                    decision.durationBin, durationBins: config.tdtConfig.durationBins)

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
                        from: step.output, key: "decoder", errorMessage: "Invalid decoder output")
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
            "target_length": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: state.hiddenState),
            "c_in": MLFeatureValue(multiArray: state.cellState),
        ])

        // Reuse decoder state output buffers to avoid CoreML allocating new ones
        // Note: outputBackings expects raw backing objects (MLMultiArray / CVPixelBuffer)
        predictionOptions.outputBackings = [
            "h_out": state.hiddenState,
            "c_out": state.cellState,
        ]

        let output = try model.prediction(
            from: input,
            options: predictionOptions
        )

        var newState = state
        newState.update(from: output)

        return (output, newState)
    }

    /// Joint network execution with zero-copy
    /// Joint network execution using preallocated input arrays and a reusable provider.
    private func runJointPrepared(
        encoderFrames: EncoderFrameView,
        timeIndex: Int,
        preparedDecoderStep: MLMultiArray,
        model: MLModel,
        encoderStep: MLMultiArray,
        encoderDestPtr: UnsafeMutablePointer<Float>,
        encoderDestStride: Int,
        inputProvider: MLFeatureProvider,
        tokenIdBacking: MLMultiArray,
        tokenProbBacking: MLMultiArray,
        durationBacking: MLMultiArray
    ) throws -> JointDecision {

        // Fill encoder step with the requested frame
        try encoderFrames.copyFrame(at: timeIndex, into: encoderDestPtr, destinationStride: encoderDestStride)

        // Prefetch arrays for ANE
        ANEOptimizer.prefetchToNeuralEngine(encoderStep)
        ANEOptimizer.prefetchToNeuralEngine(preparedDecoderStep)

        // Reuse tiny output tensors for joint prediction (provide raw MLMultiArray backings)
        predictionOptions.outputBackings = [
            "token_id": tokenIdBacking,
            "token_prob": tokenProbBacking,
            "duration": durationBacking,
        ]

        // Execute joint network using the reusable provider
        let output = try model.prediction(
            from: inputProvider,
            options: predictionOptions
        )

        let tokenIdArray = try extractFeatureValue(
            from: output, key: "token_id", errorMessage: "Joint decision output missing token_id")
        let tokenProbArray = try extractFeatureValue(
            from: output, key: "token_prob", errorMessage: "Joint decision output missing token_prob")
        let durationArray = try extractFeatureValue(
            from: output, key: "duration", errorMessage: "Joint decision output missing duration")

        guard tokenIdArray.count == 1,
            tokenProbArray.count == 1,
            durationArray.count == 1
        else {
            throw ASRError.processingFailed("Joint decision returned unexpected tensor shapes")
        }

        let tokenPointer = tokenIdArray.dataPointer.bindMemory(to: Int32.self, capacity: tokenIdArray.count)
        let token = Int(tokenPointer[0])
        let probPointer = tokenProbArray.dataPointer.bindMemory(to: Float.self, capacity: tokenProbArray.count)
        let probability = probPointer[0]
        let durationPointer = durationArray.dataPointer.bindMemory(to: Int32.self, capacity: durationArray.count)
        let durationBin = Int(durationPointer[0])

        return JointDecision(token: token, probability: probability, durationBin: durationBin)
    }

    private func prepareDecoderProjection(_ projection: MLMultiArray) throws -> MLMultiArray {
        let hiddenSize = ASRConstants.decoderHiddenSize
        let shape = projection.shape.map { $0.intValue }

        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid decoder projection rank: \(shape)")
        }
        guard shape[0] == 1 else {
            throw ASRError.processingFailed("Unsupported decoder batch dimension: \(shape[0])")
        }
        guard projection.dataType == .float32 else {
            throw ASRError.processingFailed("Unsupported decoder projection type: \(projection.dataType)")
        }

        let hiddenAxis: Int
        if shape[2] == hiddenSize {
            hiddenAxis = 2
        } else if shape[1] == hiddenSize {
            hiddenAxis = 1
        } else {
            throw ASRError.processingFailed("Decoder projection hidden size mismatch: \(shape)")
        }

        let timeAxis = (0...2).first { $0 != hiddenAxis && $0 != 0 } ?? 1
        guard shape[timeAxis] == 1 else {
            throw ASRError.processingFailed("Decoder projection time axis must be 1: \(shape)")
        }

        let normalized = try ANEOptimizer.createANEAlignedArray(
            shape: [1, NSNumber(value: hiddenSize), 1],
            dataType: .float32
        )

        let destPtr = normalized.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)
        let destStrides = normalized.strides.map { $0.intValue }
        let destHiddenStride = destStrides[1]
        let destStrideCblas = try makeBlasIndex(destHiddenStride, label: "Decoder destination stride")
        let sourcePtr = projection.dataPointer.bindMemory(to: Float.self, capacity: projection.count)
        let strides = projection.strides.map { $0.intValue }

        let hiddenStride = strides[hiddenAxis]
        let timeStride = strides[timeAxis]
        let batchStride = strides[0]

        var baseOffset = 0
        if batchStride < 0 {
            baseOffset += (shape[0] - 1) * batchStride
        }
        if timeStride < 0 {
            baseOffset += (shape[timeAxis] - 1) * timeStride
        }

        let minOffset = hiddenStride < 0 ? hiddenStride * (hiddenSize - 1) : 0
        let maxOffset = hiddenStride > 0 ? hiddenStride * (hiddenSize - 1) : 0
        let lowerBound = baseOffset + minOffset
        let upperBound = baseOffset + maxOffset
        guard lowerBound >= 0 && upperBound < projection.count else {
            throw ASRError.processingFailed("Decoder projection stride exceeds buffer bounds")
        }

        let startPtr = sourcePtr.advanced(by: baseOffset)
        if hiddenStride == 1 && destHiddenStride == 1 {
            destPtr.update(from: startPtr, count: hiddenSize)
        } else {
            let count = try makeBlasIndex(hiddenSize, label: "Decoder projection length")
            let stride = try makeBlasIndex(hiddenStride, label: "Decoder projection stride")
            cblas_scopy(count, startPtr, stride, destPtr, destStrideCblas)
        }

        return normalized
    }

    /// Populate a preallocated decoder-step array with the normalized projection data.
    private func populatePreparedDecoderProjection(
        _ projection: MLMultiArray,
        into out: MLMultiArray
    ) throws {
        let hiddenSize = ASRConstants.decoderHiddenSize
        let shape = projection.shape.map { $0.intValue }

        guard shape.count == 3 else {
            throw ASRError.processingFailed("Invalid decoder projection rank: \(shape)")
        }
        guard shape[0] == 1 else {
            throw ASRError.processingFailed("Unsupported decoder batch dimension: \(shape[0])")
        }
        guard projection.dataType == .float32 else {
            throw ASRError.processingFailed("Unsupported decoder projection type: \(projection.dataType)")
        }

        // Validate destination shape
        let outShape = out.shape.map { $0.intValue }
        guard out.dataType == .float32, outShape.count == 3, outShape[0] == 1, outShape[2] == 1,
            outShape[1] == hiddenSize
        else {
            throw ASRError.processingFailed("Prepared decoder step shape mismatch: \(out.shapeString)")
        }

        let hiddenAxis: Int
        if shape[2] == hiddenSize {
            hiddenAxis = 2
        } else if shape[1] == hiddenSize {
            hiddenAxis = 1
        } else {
            throw ASRError.processingFailed("Decoder projection hidden size mismatch: \(shape)")
        }

        let timeAxis = (0...2).first { $0 != hiddenAxis && $0 != 0 } ?? 1
        guard shape[timeAxis] == 1 else {
            throw ASRError.processingFailed("Decoder projection time axis must be 1: \(shape)")
        }

        let destPtr = out.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)
        let destStrides = out.strides.map { $0.intValue }
        let destHiddenStride = destStrides[1]
        let destStrideCblas = try makeBlasIndex(destHiddenStride, label: "Decoder destination stride")

        let sourcePtr = projection.dataPointer.bindMemory(to: Float.self, capacity: projection.count)
        let strides = projection.strides.map { $0.intValue }
        let hiddenStride = strides[hiddenAxis]
        let timeStride = strides[timeAxis]
        let batchStride = strides[0]

        var baseOffset = 0
        if batchStride < 0 { baseOffset += (shape[0] - 1) * batchStride }
        if timeStride < 0 { baseOffset += (shape[timeAxis] - 1) * timeStride }

        let minOffset = hiddenStride < 0 ? hiddenStride * (hiddenSize - 1) : 0
        let maxOffset = hiddenStride > 0 ? hiddenStride * (hiddenSize - 1) : 0
        let lowerBound = baseOffset + minOffset
        let upperBound = baseOffset + maxOffset
        guard lowerBound >= 0 && upperBound < projection.count else {
            throw ASRError.processingFailed("Decoder projection stride exceeds buffer bounds")
        }

        let startPtr = sourcePtr.advanced(by: baseOffset)
        if hiddenStride == 1 && destHiddenStride == 1 {
            destPtr.update(from: startPtr, count: hiddenSize)
        } else {
            let count = try makeBlasIndex(hiddenSize, label: "Decoder projection length")
            let stride = try makeBlasIndex(hiddenStride, label: "Decoder projection stride")
            cblas_scopy(count, startPtr, stride, destPtr, destStrideCblas)
        }
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
    private func mapDurationBin(_ binIndex: Int, durationBins: [Int]) throws -> Int {
        guard binIndex >= 0 && binIndex < durationBins.count else {
            throw ASRError.processingFailed("Duration bin index out of range: \(binIndex)")
        }
        return durationBins[binIndex]
    }

    private func clampProbability(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return min(max(value, 0), 1)
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
            "target_length": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: hiddenState),
            "c_in": MLFeatureValue(multiArray: cellState),
        ])
    }

    internal func prepareJointInput(
        encoderOutput: MLMultiArray,
        decoderOutput: MLFeatureProvider,
        timeIndex: Int
    ) throws -> MLFeatureProvider {
        let encoderFrames = try EncoderFrameView(
            encoderOutput: encoderOutput,
            validLength: encoderOutput.count)
        let encoderStep = try ANEOptimizer.createANEAlignedArray(
            shape: [1, NSNumber(value: encoderFrames.hiddenSize), 1],
            dataType: .float32)
        let encoderPtr = encoderStep.dataPointer.bindMemory(to: Float.self, capacity: encoderFrames.hiddenSize)
        let destStrides = encoderStep.strides.map { $0.intValue }
        try encoderFrames.copyFrame(at: timeIndex, into: encoderPtr, destinationStride: destStrides[1])

        let decoderProjection = try extractFeatureValue(
            from: decoderOutput, key: "decoder", errorMessage: "Invalid decoder output")
        let normalizedDecoder = try prepareDecoderProjection(decoderProjection)

        return try MLDictionaryFeatureProvider(dictionary: [
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
            "decoder_step": MLFeatureValue(multiArray: normalizedDecoder),
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
