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

        // --- DEBUG: chunk start context ---
        let chunkTag = String(UUID().uuidString.prefix(6))
        print(
            "[\(chunkTag)] chunk start len=\(encoderSequenceLength), lastToken=\(String(describing: decoderState.lastToken)) timestamp=\(Date())"
        )

        print(
            "[\(chunkTag)] state.h L2=\(decoderState.hiddenState.l2Normf()), state.c L2=\(decoderState.cellState.l2Normf())"
        )
        // Ensure a clean predictor state for a NEW utterance (NeMo parity).
        // NeMo calls decoder with state=None at start; do the same by zeroing ours.
        if decoderState.lastToken == nil && decoderState.predictorOutput == nil {
            let zero = TdtDecoderState(fallback: true)  // allocates zeroed h/c
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
            let proj = try extractFeatureValue(
                from: primed.output, key: "decoder_output", errorMessage: "Invalid decoder output")
            decoderState.predictorOutput = proj
            hypothesis.decState = primed.newState

            print(
                "[\(chunkTag)] SOS primed (blank-as-pad=\(sos)); decoder_output shape=\(proj.shapeString) L2=\(proj.l2Normf())"
            )
            print(
                "[\(chunkTag)] primed.h L2=\(hypothesis.decState?.hiddenState.l2Normf() ?? -1), primed.c L2=\(hypothesis.decState?.cellState.l2Normf() ?? -1)"
            )
        }

        // Prevent infinite loops with max_symbols_per_step
        var lastEmissionTimestamp = -1
        var emissionsAtThisTimestamp = 0
        let maxSymbolsPerStep = config.tdtConfig.maxSymbolsPerStep

        // OUTER loop - advance time index by predicted duration (NeMo-parity: blank-looping)
        while activeMask {
            var label = hypothesis.lastToken ?? config.tdtConfig.blankId

            // After SOS priming, always use the primed predictor state until we emit a non-blank.
            let stateToUse = hypothesis.decState ?? decoderState

            // Use cached predictor output if available (don’t advance predictor until a non-blank is emitted).
            let decoderResult: (output: MLFeatureProvider, newState: TdtDecoderState)
            if let cached = decoderState.predictorOutput {
                let provider = try MLDictionaryFeatureProvider(dictionary: [
                    "decoder_output": MLFeatureValue(multiArray: cached)
                ])
                decoderResult = (output: provider, newState: stateToUse)
            } else {
                decoderResult = try runDecoder(
                    token: label,
                    state: stateToUse,
                    model: decoderModel,
                    targetArray: reusableTargetArray,
                    targetLengthArray: reusableTargetLengthArray
                )
            }

            // 1) Joint at current time index
            let encoderStep = encoderFrames[safeTimeIndices]
            let logits = try runJoint(
                encoderStep: encoderStep,
                decoderOutput: decoderResult.output,
                model: jointModel
            )
            let (tokenLogits, durationLogits) = try splitLogits(
                logits, durationElements: config.tdtConfig.durationBins.count)

            label = argmaxSIMD(tokenLogits)
            var score = tokenLogits[label]
            var duration = config.tdtConfig.durationBins[argmaxSIMD(durationLogits)]

            // NeMo safeguard: for BLANK, force duration >= 1
            let blankId = config.tdtConfig.blankId
            var blankMask = (label == blankId)
            if blankMask && duration == 0 { duration = 1 }

            // Save timestamp (t) where this label was decided
            timeIndicesCurrentLabels = timeIndices

            // Pre-advance time by predicted duration (NeMo does this before the blank loop)
            timeIndices += duration
            safeTimeIndices = min(timeIndices, lastTimestep)

            // Determine loop condition: keep probing while we are BLANK and still active
            activeMask = timeIndices < encoderSequenceLength
            var advanceMask = activeMask && blankMask

            // 2) INNER LOOP (NeMo-style): advance through BLANKs, re-running Joint at new t each time
            while advanceMask {
                // Store timestamp for this probe (used if/when we find a non-blank)
                timeIndicesCurrentLabels = timeIndices

                // Joint again at the NEW time index (same predictor output; we haven’t emitted yet)
                let innerEncoderStep = encoderFrames[safeTimeIndices]
                let innerLogits = try runJoint(
                    encoderStep: innerEncoderStep,
                    decoderOutput: decoderResult.output,
                    model: jointModel
                )

                let (innerTokenLogits, innerDurationLogits) = try splitLogits(
                    innerLogits, durationElements: config.tdtConfig.durationBins.count)
                label = argmaxSIMD(innerTokenLogits)
                score = innerTokenLogits[label]
                duration = config.tdtConfig.durationBins[argmaxSIMD(innerDurationLogits)]

                blankMask = (label == blankId)
                if blankMask && duration == 0 { duration = 1 }  // blank safeguard

                // Advance time by this (blank) duration and check if we should continue blank-looping
                timeIndices += duration
                safeTimeIndices = min(timeIndices, lastTimestep)
                activeMask = timeIndices < encoderSequenceLength
                advanceMask = activeMask && blankMask
            }

            // 3) If we found a NON-BLANK while still active, emit it and advance predictor ONCE
            if activeMask && label != blankId {
                hypothesis.ySequence.append(label)
                hypothesis.score += score
                hypothesis.timestamps.append(timeIndicesCurrentLabels)
                hypothesis.lastToken = label

                // Advance predictor using the emitted label; refresh predictor cache for subsequent frames
                let step = try runDecoder(
                    token: label,
                    state: decoderResult.newState,  // state that produced this Joint
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

                // --- ADD BELOW: track emissions at the same timestamp and force a 1-frame advance if needed ---
                if timeIndicesCurrentLabels == lastEmissionTimestamp {
                    emissionsAtThisTimestamp += 1
                } else {
                    lastEmissionTimestamp = timeIndicesCurrentLabels
                    emissionsAtThisTimestamp = 1
                }

                if emissionsAtThisTimestamp >= maxSymbolsPerStep {
                    // Force a blank advance of 1 frame (do NOT touch predictor state/output)
                    let forcedAdvance = 1
                    timeIndices = min(timeIndices + forcedAdvance, lastTimestep)
                    safeTimeIndices = min(timeIndices, lastTimestep)

                    // Optional: log for debug
                    print(
                        "[\(chunkTag)] hit maxSymbols=\(maxSymbolsPerStep) at t=\(timeIndicesCurrentLabels) → forced t+=1"
                    )

                    // Reset the counter for the new frame
                    emissionsAtThisTimestamp = 0
                    lastEmissionTimestamp = -1
                }
            }
        }

        print(
            "[\(chunkTag)] chunk end: final t=\(timeIndices) len=\(encoderSequenceLength) overrun=\(max(0, timeIndices - encoderSequenceLength)) emitted=\(hypothesis.ySequence.count) lastToken=\(String(describing: hypothesis.lastToken))"
        )

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
