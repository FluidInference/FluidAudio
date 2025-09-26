//
//  TdtDecoderV2.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

/// Optimized TDT decoder with hybrid CoreML + Metal acceleration
import Accelerate
import CoreML
import Foundation

@available(macOS 13.0, iOS 16.0, *)
internal struct TdtDecoderV2 {

    private struct JointDecision {
        let token: Int
        let probability: Float
        let duration: Int
    }

    private let logger = AppLogger(category: "TDT")
    private let config: ASRConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()

    init(config: ASRConfig) {
        self.config = config
    }

    // Special token Indexes matching Parakeet TDT model's vocabulary (1024 word tokens)
    // OUTPUT from joint network during decoding
    // 0-1023 represents characters, numbers, punctuations
    // 1024 represents, BLANK or nonexistent
    private let blankId = 1024

    // sosId (Start-of-Sequence)
    // sosId is INPUT when there's no real previous token
    private let sosId = 1024

    /// Execute optimized TDT decoding
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout TdtDecoderState
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

        var lastTimestamp = -1
        var lastTimestampCount = 0

        // Main decoding loop with optimizations
        while activeMask {
            var label = hypothesis.lastToken ?? sosId

            // Use cached decoder inputs
            let decoderResult = try runDecoderOptimized(
                token: label,
                state: hypothesis.decState ?? decoderState,
                model: decoderModel
            )

            // Fast encoder frame access handled inside the joint helper
            // Batch process joint network if possible
            var decision = try runJointOptimized(
                encoderFrames: encoderFrames,
                timeIndex: safeTimeIndices,
                decoderOutput: decoderResult.output,
                model: jointModel
            )

            label = decision.token
            var score = clampProbability(decision.probability)
            var actualDuration = decision.duration
            var blankMask = label == blankId

            if blankMask && actualDuration == 0 {
                actualDuration = 1
            }

            timeIndicesCurrentLabels = timeIndices
            timeIndices += actualDuration
            safeTimeIndices = min(timeIndices, lastTimestep)
            activeMask = timeIndices < encoderSequenceLength
            var advanceMask = activeMask && blankMask

            // Optimized inner loop
            while advanceMask {
                timeIndicesCurrentLabels = timeIndices

                decision = try runJointOptimized(
                    encoderFrames: encoderFrames,
                    timeIndex: safeTimeIndices,
                    decoderOutput: decoderResult.output,
                    model: jointModel
                )

                label = decision.token
                score = clampProbability(decision.probability)
                actualDuration = decision.duration

                blankMask = label == blankId
                if blankMask && actualDuration == 0 {
                    actualDuration = 1
                }

                timeIndices += actualDuration
                safeTimeIndices = min(timeIndices, lastTimestep)
                activeMask = timeIndices < encoderSequenceLength
                advanceMask = activeMask && blankMask
            }

            // Update hypothesis
            if label != blankId {
                hypothesis.ySequence.append(label)
                hypothesis.score += score
                hypothesis.timestamps.append(timeIndicesCurrentLabels)
                hypothesis.decState = decoderResult.newState
                hypothesis.lastToken = label
            }

            // Force blank logic
            let maxSymbols = config.tdtConfig.maxSymbolsPerStep
            if maxSymbols > 0 {
                if label != blankId && lastTimestamp == timeIndices
                    && lastTimestampCount >= maxSymbols
                {
                    timeIndices += 1
                    safeTimeIndices = min(timeIndices, lastTimestep)
                    activeMask = timeIndices < encoderSequenceLength
                }
            }

            if lastTimestamp == timeIndices {
                lastTimestampCount += 1
            } else {
                lastTimestamp = timeIndices
                lastTimestampCount = 1
            }
        }

        if let finalState = hypothesis.decState {
            decoderState = finalState
        }

        // Save the last token for the next chunk
        decoderState.lastToken = hypothesis.lastToken

        return hypothesis.ySequence
    }

    /// Pre-process encoder output into a stride-aware view for faster access
    private func preProcessEncoderOutput(
        _ encoderOutput: MLMultiArray, length: Int
    ) throws -> EncoderFrameView {
        try EncoderFrameView(encoderOutput: encoderOutput, validLength: length)
    }

    /// Optimized decoder execution
    private func runDecoderOptimized(
        token: Int,
        state: TdtDecoderState,
        model: MLModel
    ) throws -> (output: MLFeatureProvider, newState: TdtDecoderState) {

        // Create input arrays
        let targetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        targetArray[0] = NSNumber(value: token)

        let targetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        targetLengthArray[0] = NSNumber(value: 1)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targetArray),
            "target_length": MLFeatureValue(multiArray: targetLengthArray),
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

    /// Optimized joint network execution with zero-copy
    private func runJointOptimized(
        encoderFrames: EncoderFrameView,
        timeIndex: Int,
        decoderOutput: MLFeatureProvider,
        model: MLModel
    ) throws -> JointDecision {

        // Create ANE-aligned encoder array for optimal performance
        let hiddenSize = encoderFrames.hiddenSize
        let encoderArray = try ANEOptimizer.createANEAlignedArray(
            shape: [1, NSNumber(value: hiddenSize), 1],
            dataType: .float32
        )

        // Use optimized memory copy
        let encoderDestPtr = encoderArray.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)
        let encoderDestStride = encoderArray.strides.map { $0.intValue }[1]
        try encoderFrames.copyFrame(at: timeIndex, into: encoderDestPtr, destinationStride: encoderDestStride)

        let decoderProjection = try extractFeatureValue(
            from: decoderOutput, key: "decoder", errorMessage: "Invalid decoder output")
        let decoderArray = try prepareDecoderProjection(decoderProjection)

        // Prefetch arrays for ANE if available
        if #available(macOS 14.0, iOS 17.0, *) {
            ANEOptimizer.prefetchToNeuralEngine(encoderArray)
            ANEOptimizer.prefetchToNeuralEngine(decoderArray)
        }

        let tokenIdBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .int32)
        let tokenProbBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .float32)
        let durationBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .int32)

        let previousBackings = predictionOptions.outputBackings
        predictionOptions.outputBackings = [
            "token_id": tokenIdBacking,
            "token_prob": tokenProbBacking,
            "duration": durationBacking,
        ]
        defer { predictionOptions.outputBackings = previousBackings }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_step": MLFeatureValue(multiArray: encoderArray),
            "decoder_step": MLFeatureValue(multiArray: decoderArray),
        ])

        let output = try model.prediction(
            from: input,
            options: predictionOptions
        )

        let tokenIdArray = try extractFeatureValue(
            from: output, key: "token_id", errorMessage: "Joint network output missing token_id")
        let tokenProbArray = try extractFeatureValue(
            from: output, key: "token_prob", errorMessage: "Joint network output missing token_prob")
        let durationArray = try extractFeatureValue(
            from: output, key: "duration", errorMessage: "Joint network output missing duration")

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
        let duration = try mapDurationBin(durationBin)

        return JointDecision(token: token, probability: probability, duration: duration)
    }
    /// Predict token and duration from joint logits
    internal func predictTokenAndDuration(
        _ logits: MLMultiArray
    ) throws -> (
        token: Int, score: Float, duration: Int
    ) {
        let (tokenLogits, durationLogits) = try splitLogits(logits)

        let bestToken = argmax(tokenLogits)
        let tokenScore = tokenLogits[bestToken]

        let (_, duration) = try processDurationLogits(durationLogits)

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

    /// Calculate next time index based on duration prediction
    ///
    /// This implementation is based on NVIDIA's NeMo Parakeet TDT decoder optimization.
    /// The adaptive skip logic ensures stability for both short and long utterances.
    /// Source: Adapted from NVIDIA NeMo's TDT decoding strategy for production use.
    ///
    /// - Parameters:
    ///   - currentIdx: Current position in the audio sequence
    ///   - skip: Number of frames to skip (predicted by the model)
    ///   - sequenceLength: Total number of frames in the audio
    /// - Returns: The next frame index to process
    internal func calculateNextTimeIndex(currentIdx: Int, skip: Int, sequenceLength: Int) -> Int {
        // Determine the actual number of frames to skip
        let actualSkip: Int

        if sequenceLength < 10 && skip > 2 {
            // For very short audio (< 10 frames), limit skip to 2 frames max
            // This ensures we don't miss important tokens in brief utterances
            actualSkip = 2
        } else {
            // For normal audio, allow up to 4 frames skip
            // Even if model predicts more, cap at 4 for stability
            actualSkip = min(skip, 4)
        }

        // Move forward by actualSkip frames, but don't exceed sequence bounds
        return min(currentIdx + actualSkip, sequenceLength)
    }

    // MARK: - Private Helper Methods

    private func prepareDecoderProjection(
        _ projection: MLMultiArray
    ) throws -> MLMultiArray {
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
        let destStrideCblas: Int32
        if destHiddenStride == 1 {
            destStrideCblas = 1
        } else if let stride = Int32(exactly: destHiddenStride) {
            destStrideCblas = stride
        } else {
            throw ASRError.processingFailed("Decoder destination stride out of range")
        }

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
            guard let count = Int32(exactly: hiddenSize),
                let stride = Int32(exactly: hiddenStride)
            else {
                throw ASRError.processingFailed("Decoder projection stride out of range")
            }
            cblas_scopy(count, startPtr, stride, destPtr, destStrideCblas)
        }

        return normalized
    }

    /// Split joint logits into token and duration components with optimized memory access
    private func splitLogits(
        _ logits: MLMultiArray
    ) throws -> (
        tokenLogits: [Float], durationLogits: [Float]
    ) {
        let totalElements = logits.count
        let durationElements = config.tdtConfig.durationBins.count
        let vocabSize = totalElements - durationElements

        guard totalElements >= durationElements else {
            throw ASRError.processingFailed("Logits dimension mismatch")
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
        _ durationLogits: [Float]
    ) throws -> (
        bestDuration: Int, duration: Int
    ) {
        let bestDurationIdx = argmaxSIMD(durationLogits)
        let duration = config.tdtConfig.durationBins[bestDurationIdx]
        return (bestDurationIdx, duration)
    }

    private func clampProbability(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return min(max(value, 0), 1)
    }

    private func mapDurationBin(_ binIndex: Int) throws -> Int {
        let bins = config.tdtConfig.durationBins
        guard binIndex >= 0 && binIndex < bins.count else {
            throw ASRError.processingFailed("Duration bin index out of range: \(binIndex)")
        }
        return bins[binIndex]
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
        let frames = try EncoderFrameView(encoderOutput: encoderOutput, validLength: encoderOutput.count)
        let encoderStep = try ANEOptimizer.createANEAlignedArray(
            shape: [1, NSNumber(value: frames.hiddenSize), 1],
            dataType: .float32
        )
        let encoderPtr = encoderStep.dataPointer.bindMemory(to: Float.self, capacity: frames.hiddenSize)
        let encoderStride = encoderStep.strides.map { $0.intValue }[1]
        try frames.copyFrame(at: timeIndex, into: encoderPtr, destinationStride: encoderStride)

        let decoderProjection = try extractFeatureValue(
            from: decoderOutput, key: "decoder", errorMessage: "Invalid decoder output")
        let decoderStep = try prepareDecoderProjection(decoderProjection)

        if #available(macOS 14.0, iOS 17.0, *) {
            ANEOptimizer.prefetchToNeuralEngine(encoderStep)
            ANEOptimizer.prefetchToNeuralEngine(decoderStep)
        }

        return try MLDictionaryFeatureProvider(dictionary: [
            "encoder_step": MLFeatureValue(multiArray: encoderStep),
            "decoder_step": MLFeatureValue(multiArray: decoderStep),
        ])
    }

    // MARK: - Error Handling Helper

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
