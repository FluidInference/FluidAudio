//
//  TdtDecoder.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

/// Optimized TDT decoder with hybrid CoreML + Metal acceleration
import Accelerate
import CoreML
import Foundation
import OSLog

public struct TdtConfig: Sendable {
    public let durations: [Int]
    public let includeTokenDuration: Bool
    public let maxSymbolsPerStep: Int?

    public static let `default` = TdtConfig()

    public init(
        durations: [Int] = [0, 1, 2, 3, 4],
        includeTokenDuration: Bool = true,
        maxSymbolsPerStep: Int? = nil
    ) {
        self.durations = durations
        self.includeTokenDuration = includeTokenDuration
        self.maxSymbolsPerStep = maxSymbolsPerStep
    }
}

/// Hypothesis for TDT beam search decoding
struct TdtHypothesis: Sendable {
    var score: Float = 0.0
    var ySequence: [Int] = []
    var decState: DecoderState?
    var timestamps: [Int] = []
    var tokenDurations: [Int] = []
    /// Last non-blank token decoded in this hypothesis.
    /// Used to initialize the decoder for the next chunk, maintaining context across chunk boundaries.
    var lastToken: Int?
}

@available(macOS 13.0, iOS 16.0, *)
internal struct TdtDecoder {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "TDT")
    private let config: ASRConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()

    init(config: ASRConfig) {
        self.config = config
    }

    // Special token Indexes for v3 models
    // The joint network outputs 8193 vocab tokens (0-8192) + 5 duration tokens
    // Token 8192 is the blank token (vocabulary size boundary)
    // But the decoder embedding only accepts tokens 0-8192, so we need to handle this carefully
    private let blankId = 8192  // v3 models use 8192 as blank token

    // sosId (Start-of-Sequence)
    // Python uses token 2 (<pad>) and it works, so let's match that
    // Token 2 = <pad> (what Python uses successfully)
    // Token 64 = <|en|> for English (didn't help)
    // Token 4 = <|startoftranscript|> (alternative)
    private let sosId = 2  // Use pad token like Python does
    private let startTranscriptId = 4  // Alternative: generic start token

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

        var lastTimestamp = -1
        var lastTimestampCount = 0

        // Main decoding loop with optimizations
        while activeMask {
            var label = hypothesis.lastToken ?? sosId

            // SAFETY: Never feed token >= 8192 to decoder to avoid embedding issues
            // Map blank token (8192) to pad token (2) for decoder input
            let decoderInputToken = (label >= 8192) ? 2 : label  // Use pad token if out of bounds

            // Use cached decoder inputs
            let decoderResult = try runDecoderOptimized(
                token: decoderInputToken,
                state: hypothesis.decState ?? decoderState,
                model: decoderModel
            )

            // Fast encoder frame access
            let encoderStep = encoderFrames[safeTimeIndices]

            // Batch process joint network if possible
            let logits = try runJointOptimized(
                encoderStep: encoderStep,
                decoderOutput: decoderResult.output,
                model: jointModel
            )

            // Use raw logits directly, like Python does
            let (tokenLogits, durationLogits) = try splitLogits(logits)

            // Simple argmax on raw logits
            label = argmaxSIMD(tokenLogits)

            // Debug: Log decoded tokens
            if config.enableDebug {
                if label >= 8192 {
                    print("DEBUG: Got blank token \(label)")
                } else if hypothesis.ySequence.count < 10 {
                    // Log first 10 tokens for debugging
                    let topScores = tokenLogits.enumerated()
                        .sorted { $0.element > $1.element }
                        .prefix(3)
                        .map { "(\($0.offset): \(String(format: "%.2f", $0.element)))" }
                        .joined(separator: ", ")
                    print("DEBUG: Token \(hypothesis.ySequence.count): selected \(label), top scores: \(topScores)")
                }
            }

            var score = tokenLogits[label]
            let duration = config.tdtConfig.durations[argmaxSIMD(durationLogits)]

            var blankMask = label == blankId
            var actualDuration = duration

            if blankMask && duration == 0 {
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

                let innerEncoderStep = encoderFrames[safeTimeIndices]
                let innerLogits = try runJointOptimized(
                    encoderStep: innerEncoderStep,
                    decoderOutput: decoderResult.output,
                    model: jointModel
                )

                // Use raw logits like Python does
                let (innerTokenLogits, innerDurationLogits) = try splitLogits(innerLogits)

                let moreLabel = argmaxSIMD(innerTokenLogits)
                let moreScore = innerTokenLogits[moreLabel]
                let moreDuration = config.tdtConfig.durations[argmaxSIMD(innerDurationLogits)]

                label = moreLabel
                score = moreScore
                actualDuration = moreDuration

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
            if let maxSymbols = config.tdtConfig.maxSymbolsPerStep {
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

        // Shape is [batch, time, hidden] = [1, 126, 1024]
        let batchSize = shape[0].intValue
        let sequenceLength = shape[1].intValue
        let hiddenSize = shape[2].intValue

        guard batchSize == 1 else {
            throw ASRError.processingFailed("Expected batch size 1, got \(batchSize)")
        }

        guard hiddenSize == 1024 else {
            throw ASRError.processingFailed("Expected hidden size 1024, got \(hiddenSize)")
        }

        guard length <= sequenceLength else {
            throw ASRError.processingFailed("Requested length \(length) exceeds sequence length \(sequenceLength)")
        }

        var frames = EncoderFrameArray()
        frames.reserveCapacity(length)

        // Zero-copy optimization: create views instead of copying data
        if encoderOutput.dataType == .float32 {
            // Store the encoder output reference for zero-copy access
            let floatPtr = encoderOutput.dataPointer.bindMemory(
                to: Float.self, capacity: encoderOutput.count)

            // The data is laid out as [batch, time, hidden]
            // So for batch 0, time t, the data starts at: t * hiddenSize
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
                    // For [batch, time, hidden] layout with batch=0
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

    /// Optimized decoder execution
    private func runDecoderOptimized(
        token: Int,
        state: DecoderState,
        model: MLModel
    ) throws -> (output: MLFeatureProvider, newState: DecoderState) {

        // Create input arrays
        let targetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        targetArray[0] = NSNumber(value: token)

        let targetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        targetLengthArray[0] = NSNumber(value: 1)

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

    /// Optimized joint network execution with zero-copy
    private func runJointOptimized(
        encoderStep: [Float],
        decoderOutput: MLFeatureProvider,
        model: MLModel
    ) throws -> MLMultiArray {

        // Create ANE-aligned encoder array for optimal performance
        // The encoder step is a single frame of size 1024 (hidden dimension)
        let encoderArray = try ANEOptimizer.createANEAlignedArray(
            shape: [1, 1, 1024],
            dataType: .float32
        )

        // Use optimized memory copy
        // Ensure we copy exactly 1024 values (the encoder hidden dimension)
        guard encoderStep.count == 1024 else {
            throw ASRError.processingFailed("Invalid encoder frame size: \(encoderStep.count), expected 1024")
        }
        encoderStep.withUnsafeBufferPointer { buffer in
            let destPtr = encoderArray.dataPointer.bindMemory(
                to: Float.self, capacity: 1024)
            memcpy(destPtr, buffer.baseAddress!, 1024 * MemoryLayout<Float>.stride)
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

    /// Split joint logits into token and duration components with optimized memory access
    private func splitLogits(
        _ logits: MLMultiArray
    ) throws -> (
        tokenLogits: [Float], durationLogits: [Float]
    ) {
        let totalElements = logits.count
        let durationElements = config.tdtConfig.durations.count
        let vocabSize = totalElements - durationElements

        // Debug: Log dimensions
        if config.enableDebug {
            print("DEBUG: Joint output - total: \(totalElements), vocab: \(vocabSize), durations: \(durationElements)")
        }

        guard totalElements >= durationElements else {
            throw ASRError.processingFailed(
                "Logits dimension mismatch: got \(totalElements), need at least \(durationElements)")
        }

        // Create views directly without copying - zero-copy operation
        let logitsPtr = logits.dataPointer.bindMemory(to: Float.self, capacity: totalElements)

        // Use ContiguousArray for better cache locality
        let tokenLogits = ContiguousArray(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))
        let durationLogits = ContiguousArray(
            UnsafeBufferPointer(start: logitsPtr + vocabSize, count: durationElements))

        // Debug: Check for valid values
        if config.enableDebug {
            let tokenArray = Array(tokenLogits)
            let maxToken = tokenArray.max() ?? -Float.infinity
            let minToken = tokenArray.min() ?? Float.infinity
            let argMaxToken = tokenArray.firstIndex(of: maxToken) ?? -1
            print("DEBUG: Token logits range: [\(minToken), \(maxToken)], argmax: \(argMaxToken)")

            // Check if all values are the same (indicating a problem)
            if maxToken == minToken {
                print("WARNING: All token logits have the same value: \(maxToken)")
            }
        }

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
        let duration = config.tdtConfig.durations[bestDurationIdx]
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

/// Pre-processed encoder frames for fast access
private typealias EncoderFrameArray = [[Float]]
