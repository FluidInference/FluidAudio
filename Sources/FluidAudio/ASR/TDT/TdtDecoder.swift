/// TDT decoder with hybrid CoreML + Metal acceleration
import Accelerate
import CoreML
import Foundation
import OSLog

/// Pre-processed encoder frames for fast access
private typealias EncoderFrameArray = [[Float]]

@available(macOS 13.0, iOS 16.0, *)
internal struct TdtDecoder {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "TDT")
    private let config: ASRConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()
    // Parakeet‑TDT‑v3: duration head has 5 bins mapping directly to frame advances

    init(config: ASRConfig) {
        self.config = config
    }

    /// Execute optimized TDT decoding
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout TdtDecoderState
    ) async throws -> [Int] {
        // print("TDT decode: encoderSequenceLength=\(encoderSequenceLength)")

        guard encoderSequenceLength > 1 else {
            print("TDT: Encoder sequence too short (\(encoderSequenceLength))")
            return []
        }

        // Pre-process encoder output for faster access
        let encoderFrames = try preProcessEncoderOutput(
            encoderOutput, length: encoderSequenceLength)

        var hypothesis = TdtHypothesis(decState: decoderState)
        hypothesis.lastToken = decoderState.lastToken  // Preserve last token from previous chunk
        var timeIndices = 0
        var safeTimeIndices = 0
        var timeIndicesCurrentLabels = 0
        var activeMask = true
        let lastTimestep = encoderSequenceLength - 1

        // Gate loop-labels by blank margin (logit units). Typical good range: 1.5–3.0.
        // We apply it most aggressively before the first real emission.
        let LOOP_LABEL_MARGIN_BEFORE_FIRST: Float = 4.0
        let LOOP_LABEL_MARGIN_AFTER_FIRST:  Float = 0.5

        // Variables removed - no longer needed with simplified max_symbols logic

        // Pre-allocate reusable MLMultiArrays for decoder to avoid repeated allocations
        let reusableTargetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        let reusableTargetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        reusableTargetLengthArray[0] = NSNumber(value: 1)  // This never changes

        // --- DEBUG: chunk start context ---
        let chunkTag = String(UUID().uuidString.prefix(6))
        print("[\(chunkTag)] chunk start len=\(encoderSequenceLength), lastToken=\(String(describing: decoderState.lastToken))")

        print("[\(chunkTag)] state.h L2=\(decoderState.hiddenState.l2Normf()), state.c L2=\(decoderState.cellState.l2Normf())")
        // Ensure a clean predictor state for a NEW utterance (NeMo parity).
        // NeMo calls decoder with state=None at start; do the same by zeroing ours.
        if decoderState.lastToken == nil && decoderState.predictorOutput == nil {
            let zero = TdtDecoderState(fallback: true) // allocates zeroed h/c
            decoderState.hiddenState.copyData(from: zero.hiddenState)
            decoderState.cellState.copyData(from: zero.cellState)
            print("[\(chunkTag)] RESET predictor LSTM at start (zero state)")
        }

        // --- SOS (blank-as-pad) predictor priming (one-time at utterance start) ---
        if decoderState.predictorOutput == nil && hypothesis.lastToken == nil {
            let sos = config.tdtConfig.blankId
            let primed = try runDecoder(
                token: sos,
                state: decoderState,
                model: decoderModel,
                targetArray: reusableTargetArray,
                targetLengthArray: reusableTargetLengthArray
            )
            let proj = try extractFeatureValue(from: primed.output, key: "decoder_output", errorMessage: "Invalid decoder output")
            decoderState.predictorOutput = proj
            hypothesis.decState = primed.newState

            print("[\(chunkTag)] SOS primed (blank-as-pad=\(sos)); decoder_output shape=\(proj.shapeString) L2=\(proj.l2Normf())")
            print("[\(chunkTag)] primed.h L2=\(hypothesis.decState?.hiddenState.l2Normf() ?? -1), primed.c L2=\(hypothesis.decState?.cellState.l2Normf() ?? -1)")
        }

        // OUTER loop - advance time index by predicted duration
        while activeMask {
            var label = hypothesis.lastToken ?? config.tdtConfig.blankId
            var finalDuration = 0  // Track the final duration to apply

            // After SOS priming we MUST use the primed predictor state (hypothesis.decState), not zeros.
            // Fall back to decoderState only if we truly have no primed state yet.
            let stateToUse = hypothesis.decState ?? decoderState

            // Use cached predictor output if available; DO NOT advance predictor until a non-blank is emitted.
            let decoderResult: (output: MLFeatureProvider, newState: TdtDecoderState)
            if let cached = decoderState.predictorOutput {
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "decoder_output": MLFeatureValue(multiArray: cached)
                ])
                decoderResult = (output: provider, newState: stateToUse)
                // IMPORTANT: do NOT clear the cache here. It is reused on every frame until a non-blank is emitted.
                print("[\(chunkTag)] Joint uses cached predictor_output; t=\(timeIndices)")
            } else {
                // Fallback: only happens after we've emitted a non-blank and refreshed the cache,
                // or if SOS priming was somehow skipped.
                decoderResult = try runDecoder(
                    token: label,
                    state: stateToUse,
                    model: decoderModel,
                    targetArray: reusableTargetArray,
                    targetLengthArray: reusableTargetLengthArray
                )
                if hypothesis.lastToken == nil {
                    print("[\(chunkTag)] WARNING: no cached predictor_output but no token emitted yet; decoder ran with label=\(label)")
                }
            }
            // Fast encoder frame access
            let encoderStep = encoderFrames[safeTimeIndices]

            // Batch process joint network if possible\
            print("[\(chunkTag)] pre-Joint t=\(timeIndices) safe=\(safeTimeIndices) lastToken=\(String(describing: hypothesis.lastToken))")

            let logits = try runJoint(
                encoderStep: encoderStep,
                decoderOutput: decoderResult.output,
                model: jointModel
            )

            // Token/duration prediction
            let (tokenLogits, durationLogits) = try splitLogits(
                logits, durationElements: config.tdtConfig.durationBins.count)

            label = argmaxSIMD(tokenLogits)
            var score = tokenLogits[label]
            let duration = config.tdtConfig.durationBins[argmaxSIMD(durationLogits)]
            print("[\(chunkTag)] Joint@t=\(timeIndices) -> label=\(label) score=\(score) dur=\(duration) blank=\(label == config.tdtConfig.blankId) needLoop=\((duration == 0) && (label != config.tdtConfig.blankId))")
            // --- loop-label gating vs blank (OUTER) ---
            let blankLogit = tokenLogits[config.tdtConfig.blankId]
            let marginOuter = score - blankLogit
            let reqOuter = (hypothesis.lastToken == nil) ? LOOP_LABEL_MARGIN_BEFORE_FIRST : LOOP_LABEL_MARGIN_AFTER_FIRST

            if (duration == 0) && (label != config.tdtConfig.blankId) && (marginOuter < reqOuter) {
                // Treat this as a BLANK step to avoid early hallucinations.
                print("[\(chunkTag)] gate@outer: token \(label) margin=\(marginOuter) < \(reqOuter) → treat as BLANK; advance by max(1,dur)")
                // Convert to blank behavior:
                label = config.tdtConfig.blankId
                score = blankLogit
            }

            let blankMask = label == config.tdtConfig.blankId

            // Determine if we need inner loop (loop labels) - only for non-blank tokens with duration=0
            let needLoop = (duration == 0) && !blankMask
            let originalNeedLoop = needLoop

            // Apply NeMo safeguard for outer loop blank as well
            // Prevent frame skipping for blanks until first token is emitted
            var actualDuration = duration
            if blankMask && duration == 0 {
                actualDuration = 1
                // print("TDT: Outer loop blank with duration=0, forcing duration=1")
            } else if blankMask && hypothesis.lastToken == nil {
                actualDuration = 1  // Force duration=1 for all blanks before first real token
                print("[\(chunkTag)] BLANK before first token: forcing duration=1 (was \(duration))")
            }

            // Set the final duration initially
            finalDuration = actualDuration

            // Track if we've advanced the time index yet
            var hasAdvanced = false

            // Save current time index for token timestamps
            timeIndicesCurrentLabels = timeIndices

            // Only advance time if we have a non-zero duration OR blank
            if !needLoop {
                timeIndices += actualDuration
                safeTimeIndices = min(timeIndices, lastTimestep)
                hasAdvanced = true
            }

            activeMask = timeIndices < encoderSequenceLength
            var advanceMask = activeMask && needLoop

            // Track the current decoder result for state updates
            var currentDecoderResult = decoderResult

            // Track tokens emitted at this frame (duration==0 case)
            var innerLoopCount = 0
            var innerLoopTokensEmitted = 0

            // INNER LOOP - continue predicting tokens at the same time index when duration == 0
            while advanceMask {
                print("[\(chunkTag)] innerLoop#\(innerLoopCount) t=\(timeIndices) (no time advance yet)")
                // NeMo parity: if we've emitted max symbols on this frame, force t += 1 and break
                if innerLoopTokensEmitted >= config.tdtConfig.maxSymbolsPerStep {
                    print("[\(chunkTag)] hit maxSymbols=\(config.tdtConfig.maxSymbolsPerStep) at t=\(timeIndices) → t+=1 and break")
                    timeIndices += 1
                    safeTimeIndices = min(timeIndices, lastTimestep)
                    hasAdvanced = true
                    break
                }

                innerLoopCount += 1

                // Use the SAME encoder frame (don't advance safeTimeIndices)
                let innerEncoderStep = encoderFrames[safeTimeIndices]
                let innerLogits = try runJoint(
                    encoderStep: innerEncoderStep,
                    decoderOutput: currentDecoderResult.output,
                    model: jointModel
                )

                let (innerTokenLogits, innerDurationLogits) = try splitLogits(
                    innerLogits, durationElements: config.tdtConfig.durationBins.count)
                let moreLabel = argmaxSIMD(innerTokenLogits)
                let moreScore = innerTokenLogits[moreLabel]
                let moreDuration = config.tdtConfig.durationBins[argmaxSIMD(innerDurationLogits)]

                label = moreLabel
                score = moreScore
                print("[\(chunkTag)] inner Joint -> label=\(moreLabel) score=\(moreScore) dur=\(moreDuration) blank=\(moreLabel == config.tdtConfig.blankId)")
                
                // --- loop-label gating vs blank (INNER) ---
                let innerBlankLogit = innerTokenLogits[config.tdtConfig.blankId]
                let marginInner = moreScore - innerBlankLogit
                let reqInner = (hypothesis.lastToken == nil) ? LOOP_LABEL_MARGIN_BEFORE_FIRST : LOOP_LABEL_MARGIN_AFTER_FIRST
                var gatedToBlank = false
                if (moreDuration == 0) && (moreLabel != config.tdtConfig.blankId) && (marginInner < reqInner) {
                    print("[\(chunkTag)] gate@inner: token \(moreLabel) margin=\(marginInner) < \(reqInner) → BLANK@+1")
                    // Force a blank with duration 1 to get more acoustic context and exit the loop cleanly.
                    label = config.tdtConfig.blankId
                    score = innerBlankLogit
                    gatedToBlank = true
                }

                // Apply NeMo rule IMMEDIATELY for blank tokens with duration=0
                var actualDuration = moreDuration
                let innerBlankMask = (label == config.tdtConfig.blankId) || gatedToBlank

                if innerBlankMask && (moreDuration == 0 || gatedToBlank) {
                    actualDuration = 1
                    print("TDT: Applying blank duration=0 safeguard, forcing duration=1")
                }
                
                // Update final duration with the last predicted duration
                finalDuration = actualDuration

                print(
                    "TDT inner: token=\(moreLabel) duration=\(moreDuration) actualDuration=\(actualDuration) blank=\(innerBlankMask)"
                )

                // Update hypothesis for non-blank tokens immediately
                if !innerBlankMask {
                    hypothesis.ySequence.append(moreLabel)
                    hypothesis.score += moreScore
                    hypothesis.timestamps.append(timeIndicesCurrentLabels)
                    hypothesis.lastToken = moreLabel
                    innerLoopTokensEmitted += 1
                    print("[\(chunkTag)] EMIT token=\(moreLabel) at t=\(timeIndicesCurrentLabels) (inner)")

                    // Advance predictor ONLY after a non-blank; refresh the cache for subsequent frames.
                    let step = try runDecoder(
                        token: moreLabel,
                        state: currentDecoderResult.newState,
                        model: decoderModel,
                        targetArray: reusableTargetArray,
                        targetLengthArray: reusableTargetLengthArray
                    )
                    currentDecoderResult = step
                    hypothesis.decState = step.newState
                    decoderState.predictorOutput = try extractFeatureValue(
                        from: step.output,
                        key: "decoder_output",
                        errorMessage: "Invalid decoder output"
                    )
                    print("[\(chunkTag)] EMIT \(moreLabel) → refreshed predictor cache (inner)")
                }

                // Exit conditions: blank token OR non-zero duration
                if innerBlankMask || actualDuration > 0 {
                    print("[\(chunkTag)] advance due to blank/dur or maxSymbols → t += \(actualDuration == 0 ? 1 : actualDuration)")

                    // Advance time index by actualDuration (which may be 1 for blank+duration=0)
                    if !hasAdvanced {
                        timeIndices += actualDuration
                        safeTimeIndices = min(timeIndices, lastTimestep)
                        hasAdvanced = true
                    }
                    break
                }

                // Continue loop only for non-blank tokens with duration=0
                activeMask = timeIndices < encoderSequenceLength
                advanceMask = activeMask && (actualDuration == 0) && !innerBlankMask
            }  // End of inner loop
            print("[\(chunkTag)] step end: finalDuration=\(finalDuration) hasAdvanced=\(hasAdvanced) next t=\(timeIndices)")

            // Safety check matching NeMo - if we still have duration=0, force advance by 1
            // This prevents infinite loops
            if finalDuration == 0 && !hasAdvanced {
                finalDuration = 1
                timeIndices += 1
                safeTimeIndices = min(timeIndices, lastTimestep)
                // print("TDT: Safety check - forcing duration=1 to prevent infinite loop")
            }

            // Update predictor state
            if originalNeedLoop {
                // Inner loop already advanced predictor on each emission; capture final state.
                hypothesis.decState = currentDecoderResult.newState
            } else if label != config.tdtConfig.blankId {
                // Handle single token without inner loop: emit token and use existing decoder result
                hypothesis.ySequence.append(label)
                hypothesis.score += score
                hypothesis.timestamps.append(timeIndicesCurrentLabels)
                hypothesis.lastToken = label

                // Advance predictor ONLY NOW: run the decoder with the just-emitted label
                // to get the next predictor state + projected output for subsequent frames.
                let step = try runDecoder(
                    token: label,
                    state: decoderResult.newState,            // state that produced this Joint
                    model: decoderModel,
                    targetArray: reusableTargetArray,
                    targetLengthArray: reusableTargetLengthArray
                )
                hypothesis.decState = step.newState
                decoderState.predictorOutput = try extractFeatureValue(
                    from: step.output,
                    key: "decoder_output",
                    errorMessage: "Invalid decoder output"
                )
                print("[\(chunkTag)] EMIT \(label) → runDecoder+refresh predictor cache (outer)")
            } else if label == config.tdtConfig.blankId {
                // BLANK: do NOT update predictor state or predictorOutput cache.
                print("[\(chunkTag)] BLANK at t=\(timeIndices); predictor state/output unchanged")
            }

        }
        print("[\(chunkTag)] chunk end: final t=\(timeIndices) len=\(encoderSequenceLength) overrun=\(max(0, timeIndices - encoderSequenceLength)) emitted=\(hypothesis.ySequence.count) lastToken=\(String(describing: hypothesis.lastToken))")

        if let finalState = hypothesis.decState {
            decoderState = finalState
        }

        // Save the last token for the next chunk
        decoderState.lastToken = hypothesis.lastToken

        return hypothesis.ySequence
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
            for i in 0..<n { let v = self[i].floatValue; ss += v * v }
            return sqrtf(ss)
        }
    }
    /// "BxTxH" style string
    var shapeString: String { shape.map { "\($0.intValue)" }.joined(separator: "x") }
}
