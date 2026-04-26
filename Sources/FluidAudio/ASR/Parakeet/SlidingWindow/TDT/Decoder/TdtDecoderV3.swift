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

internal struct TdtDecoderV3: Sendable {

    private let logger = AppLogger(category: "TDT")
    private let config: ASRConfig
    private let modelInference = TdtModelInference()
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
        globalFrameOffset: Int = 0,
        language: Language? = nil,
        vocabulary: [Int: String]? = nil
    ) async throws -> TdtHypothesis {
        // Early exit for very short audio (< 160ms)
        guard encoderSequenceLength > 1 else {
            return TdtHypothesis(decState: decoderState)
        }

        // Use encoder hidden size from config (512 for 110m, 1024 for 0.6B)
        let expectedEncoderHidden = config.encoderHiddenSize

        // Script-filtering consumes top-K; skip the extraction when the caller
        // didn't provide a language (default path), so v3 joint runs don't pay
        // for K-length array allocations they'll never use.
        let needsTopK = language != nil

        // Build a stride-aware view so we can access encoder frames without extra copies
        let encoderFrames = try EncoderFrameView(
            encoderOutput: encoderOutput,
            validLength: encoderSequenceLength,
            expectedHiddenSize: expectedEncoderHidden
        )

        var hypothesis = TdtHypothesis(decState: decoderState)
        hypothesis.lastToken = decoderState.lastToken

        // Initialize time tracking for frame navigation
        // timeIndices: Current position in encoder frames (advances by duration)
        // timeJump: Tracks overflow when we process beyond current chunk (for streaming)
        // contextFrameAdjustment: Adjusts for adaptive context overlap
        var timeIndices = TdtFrameNavigation.calculateInitialTimeIndices(
            timeJump: decoderState.timeJump,
            contextFrameAdjustment: contextFrameAdjustment
        )

        let navigationState = TdtFrameNavigation.initializeNavigationState(
            timeIndices: timeIndices,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualAudioFrames
        )
        let effectiveSequenceLength = navigationState.effectiveSequenceLength
        var safeTimeIndices = navigationState.safeTimeIndices
        let lastTimestep = navigationState.lastTimestep
        var activeMask = navigationState.activeMask

        var timeIndicesCurrentLabels = timeIndices  // Frame where current token was emitted

        // If timeJump puts us beyond the available frames, return empty
        if timeIndices >= effectiveSequenceLength {
            return TdtHypothesis(decState: decoderState)
        }

        let reusableTargetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        let reusableTargetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        reusableTargetLengthArray[0] = NSNumber(value: 1)

        // Preallocate joint input tensors and a reusable provider to avoid per-step allocations.
        let encoderHidden = expectedEncoderHidden
        let decoderHidden = ASRConstants.decoderHiddenSize
        let reusableEncoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: encoderHidden), 1],
            dataType: .float32
        )
        let reusableDecoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: decoderHidden), 1],
            dataType: .float32
        )
        let jointInput = ReusableJointInputProvider(encoderStep: reusableEncoderStep, decoderStep: reusableDecoderStep)
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
            decoderState.hiddenState.resetData(to: 0)
            decoderState.cellState.resetData(to: 0)
        }

        // Prime the decoder with Start-of-Sequence token if needed
        // This initializes the LSTM's language model context
        // Note: In RNN-T/TDT, we use blank token as SOS
        if decoderState.predictorOutput == nil && hypothesis.lastToken == nil {
            let sos = config.tdtConfig.blankId  // blank=8192 serves as SOS
            let primed = try modelInference.runDecoder(
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
            try Task.checkCancellation()
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
                decoderResult = try modelInference.runDecoder(
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
            try modelInference.normalizeDecoderProjection(decoderProjection, into: reusableDecoderStep)

            // Run joint network with preallocated inputs
            let decision = try modelInference.runJointPrepared(
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
                durationBacking: durationBacking,
                needsTopK: needsTopK
            )

            // Predict token (what to emit) and duration (how many frames to skip)
            label = decision.token
            var score = TdtDurationMapping.clampProbability(decision.probability)

            let blankId = config.tdtConfig.blankId  // 8192 for v3 models

            Self.tokenLanguageFilter(
                label: &label,
                score: &score,
                topKIds: decision.topKIds,
                topKLogits: decision.topKLogits,
                language: language,
                vocabulary: vocabulary,
                blankId: blankId
            )

            // Map duration bin to actual frame count
            // durationBins typically = [0,1,2,3,4] meaning skip 0-4 frames
            var duration = try TdtDurationMapping.mapDurationBin(
                decision.durationBin, durationBins: config.tdtConfig.durationBins)
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
                try Task.checkCancellation()
                timeIndicesCurrentLabels = timeIndices

                // INTENTIONAL: Reusing prepared decoder step from outside loop
                let innerDecision = try modelInference.runJointPrepared(
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
                    durationBacking: durationBacking,
                    needsTopK: needsTopK
                )

                label = innerDecision.token
                score = TdtDurationMapping.clampProbability(innerDecision.probability)

                Self.tokenLanguageFilter(
                    label: &label,
                    score: &score,
                    topKIds: innerDecision.topKIds,
                    topKLogits: innerDecision.topKLogits,
                    language: language,
                    vocabulary: vocabulary,
                    blankId: blankId
                )

                duration = try TdtDurationMapping.mapDurationBin(
                    innerDecision.durationBin, durationBins: config.tdtConfig.durationBins)

                blankMask = (label == blankId)

                // Same duration=0 fix for inner loop.
                if blankMask && duration == 0 {
                    duration = 1
                }

                // Advance by duration regardless of blank/non-blank
                // This is the ORIGINAL and CORRECT logic
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
                hypothesis.tokenDurations.append(duration)
                hypothesis.lastToken = label  // Remember for next iteration

                // CRITICAL: Update decoder LSTM with the new token
                // This updates the language model context for better predictions
                // Only non-blank tokens update the decoder - this is key!
                // NOTE: We update the decoder state regardless of whether we emit the token
                // to maintain proper language model context across chunk boundaries
                let step = try modelInference.runDecoder(
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
                try Task.checkCancellation()
                let stateToUse = hypothesis.decState ?? decoderState

                // Get decoder output for final processing
                let decoderResult: (output: MLFeatureProvider, newState: TdtDecoderState)
                if let cached = decoderState.predictorOutput {
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "decoder": MLFeatureValue(multiArray: cached)
                    ])
                    decoderResult = (output: provider, newState: stateToUse)
                } else {
                    decoderResult = try modelInference.runDecoder(
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
                try modelInference.normalizeDecoderProjection(finalProjection, into: reusableDecoderStep)

                let decision = try modelInference.runJointPrepared(
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
                    durationBacking: durationBacking,
                    needsTopK: needsTopK
                )

                var token = decision.token
                var score = TdtDurationMapping.clampProbability(decision.probability)

                // Also get duration for proper timestamp calculation
                let duration = try TdtDurationMapping.mapDurationBin(
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
                    hypothesis.tokenDurations.append(duration)
                    hypothesis.lastToken = token

                    // Update decoder state
                    let step = try modelInference.runDecoder(
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
        if let lastToken = hypothesis.lastToken,
            ASRConstants.punctuationTokens.contains(lastToken)
        {
            decoderState.predictorOutput = nil
            // Keep lastToken for linguistic context - deduplication handles duplicates at higher level
        }

        // Calculate final timeJump for streaming continuation
        decoderState.timeJump = TdtFrameNavigation.calculateFinalTimeJump(
            currentTimeIndices: timeIndices,
            effectiveSequenceLength: effectiveSequenceLength,
            isLastChunk: isLastChunk
        )

        // Script filtering runs per step in the main and inner decode loops.
        // The last-chunk flush loop was empirically blank/punct-dominated on
        // the issue #512 Polish samples (0 filter swaps across 7 clips), so no
        // filter call is needed here; post-processing handles deduplication.
        return hypothesis
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

        hypothesis.tokenDurations.append(duration)
    }

    // MARK: - Private Helper Methods

    /// Replace `label`/`score` with the best right-language top-K candidate
    /// when the joint's top-1 token is in the wrong language for `language`.
    /// No-op when inputs are missing or the prediction is already right.
    ///
    /// Blanks are excluded from replacement — substituting silence via top-K
    /// would hallucinate speech, and some vocabs map blankId to an empty string
    /// which would otherwise slip through the `!matches(...)` guard.
    private static func tokenLanguageFilter(
        label: inout Int,
        score: inout Float,
        topKIds: [Int]?,
        topKLogits: [Float]?,
        language: Language?,
        vocabulary: [Int: String]?,
        blankId: Int
    ) {
        guard label != blankId,
            let language = language,
            let vocab = vocabulary,
            let topKIds = topKIds,
            let topKLogits = topKLogits,
            !topKIds.isEmpty,
            let tokenText = vocab[label],
            !TokenLanguageFilter.matches(tokenText, script: language.script),
            let filtered = TokenLanguageFilter.filterTopK(
                topKIds: topKIds,
                topKLogits: topKLogits,
                vocabulary: vocab,
                preferredScript: language.script
            )
        else { return }

        label = filtered.tokenId
        score = TdtDurationMapping.clampProbability(filtered.probability)
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
            validLength: encoderOutput.count,
            expectedHiddenSize: config.encoderHiddenSize)
        let encoderStep = try ANEMemoryUtils.createAlignedArray(
            shape: [1, NSNumber(value: encoderFrames.hiddenSize), 1],
            dataType: .float32)
        let encoderPtr = encoderStep.dataPointer.bindMemory(to: Float.self, capacity: encoderFrames.hiddenSize)
        let destStrides = encoderStep.strides.map { $0.intValue }
        try encoderFrames.copyFrame(at: timeIndex, into: encoderPtr, destinationStride: destStrides[1])

        let decoderProjection = try extractFeatureValue(
            from: decoderOutput, key: "decoder", errorMessage: "Invalid decoder output")
        let normalizedDecoder = try modelInference.normalizeDecoderProjection(decoderProjection)

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
