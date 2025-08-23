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
        decoderState: inout DecoderState
    ) async throws -> [Int] {
        logger.debug("TDT decode: encoderSequenceLength=\(encoderSequenceLength)")

        guard encoderSequenceLength > 1 else {
            logger.warning("TDT: Encoder sequence too short (\(encoderSequenceLength))")
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

        // Variables removed - no longer needed with simplified max_symbols logic

        // Pre-allocate reusable MLMultiArrays for decoder to avoid repeated allocations
        let reusableTargetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        let reusableTargetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        reusableTargetLengthArray[0] = NSNumber(value: 1)  // This never changes

        while activeMask {
            var label = hypothesis.lastToken ?? config.tdtConfig.blankId
            var finalDuration = 0  // Track the final duration to apply

            // Use cached decoder inputs
            let decoderResult = try runDecoder(
                token: label,
                state: hypothesis.decState ?? decoderState,
                model: decoderModel,
                targetArray: reusableTargetArray,
                targetLengthArray: reusableTargetLengthArray
            )

            // Fast encoder frame access
            let encoderStep = encoderFrames[safeTimeIndices]

            // Batch process joint network if possible
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

            let blankMask = label == config.tdtConfig.blankId

            // Determine if we need inner loop (loop labels) - only for non-blank tokens with duration=0
            let needLoop = (duration == 0) && !blankMask
            let originalNeedLoop = needLoop

            // Apply NeMo safeguard for outer loop blank as well
            var actualDuration = duration
            if blankMask && duration == 0 {
                actualDuration = 1
                logger.debug("TDT: Outer loop blank with duration=0, forcing duration=1")
            }

            // Set the final duration initially
            finalDuration = actualDuration

            // Debug logging for TDT algorithm
            logger.debug(
                "TDT: t=\(timeIndices) token=\(label) duration=\(duration) actualDuration=\(actualDuration) blank=\(blankMask) needLoop=\(needLoop)"
            )

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

            // Inner loop - continue predicting tokens at the same time index when duration == 0
            while advanceMask {
                // NeMo parity: if we've emitted max symbols on this frame, force t += 1 and break
                if innerLoopTokensEmitted >= config.tdtConfig.maxSymbolsPerStep {
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

                // Apply NeMo rule IMMEDIATELY for blank tokens with duration=0
                var actualDuration = moreDuration
                let innerBlankMask = moreLabel == config.tdtConfig.blankId

                if innerBlankMask && moreDuration == 0 {
                    actualDuration = 1
                    logger.debug("TDT: Applying blank duration=0 safeguard, forcing duration=1")
                }

                // Update final duration with the last predicted duration
                finalDuration = actualDuration

                logger.debug(
                    "TDT inner: token=\(moreLabel) duration=\(moreDuration) actualDuration=\(actualDuration) blank=\(innerBlankMask)"
                )

                // Update hypothesis for non-blank tokens immediately
                if !innerBlankMask {
                    hypothesis.ySequence.append(moreLabel)
                    hypothesis.score += moreScore
                    hypothesis.timestamps.append(timeIndicesCurrentLabels)
                    hypothesis.lastToken = moreLabel
                    innerLoopTokensEmitted += 1

                    // Update decoder state with the new token
                    currentDecoderResult = try runDecoder(
                        token: moreLabel,
                        state: currentDecoderResult.newState,
                        model: decoderModel,
                        targetArray: reusableTargetArray,
                        targetLengthArray: reusableTargetLengthArray
                    )
                    hypothesis.decState = currentDecoderResult.newState
                }

                // Exit conditions: blank token OR non-zero duration
                if innerBlankMask || actualDuration > 0 {
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

            // Safety check matching NeMo - if we still have duration=0, force advance by 1
            // This prevents infinite loops
            if finalDuration == 0 && !hasAdvanced {
                finalDuration = 1
                timeIndices += 1
                safeTimeIndices = min(timeIndices, lastTimestep)
                logger.debug("TDT: Safety check - forcing duration=1 to prevent infinite loop")
            }

            // Update predictor state
            if originalNeedLoop {
                // Inner loop already advanced predictor on each emission; capture final state.
                hypothesis.decState = currentDecoderResult.newState
            } else if label != config.tdtConfig.blankId {
                // Handle single token without inner loop: emit + immediately advance predictor (NeMo parity).
                hypothesis.ySequence.append(label)
                hypothesis.score += score
                hypothesis.timestamps.append(timeIndicesCurrentLabels)
                hypothesis.lastToken = label

                // Advance predictor immediately so state mirrors NeMo greedy behavior
                let step = try runDecoder(
                    token: label,
                    state: currentDecoderResult.newState,
                    model: decoderModel,
                    targetArray: reusableTargetArray,
                    targetLengthArray: reusableTargetLengthArray
                )
                hypothesis.decState = step.newState
                currentDecoderResult = step  // keep in sync (optional, but nice)
            }

        }

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
        state: DecoderState,
        model: MLModel,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray
    ) throws -> (output: MLFeatureProvider, newState: DecoderState) {

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
        decoderState: DecoderState
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

        // Sanity check for expected logits size (strict for Parakeet‑TDT‑v3)
        let expectedTotal = 8193 + durationElements
        guard totalElements == expectedTotal else {
            throw ASRError.processingFailed(
                "Unexpected logits size: \(totalElements), expected \(expectedTotal)")
        }

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
