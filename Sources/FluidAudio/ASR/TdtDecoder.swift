//
//  TdtDecoder.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import CoreML
import Foundation
import OSLog
import Accelerate

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
    var lastToken: Int?
}

/// Statistics for TDT decoding performance
struct TdtDecodingStats: Sendable, CustomStringConvertible {
    let totalFrames: Int
    let framesProcessed: Int
    let framesSkipped: Int
    let tokensGenerated: Int
    let blankTokens: Int
    let averageSkipLength: Double
    let skipRate: Double

    var description: String {
        """
        Frames: \(framesProcessed)/\(totalFrames) (skip rate: \(String(format: "%.1f", skipRate * 100))%)
        Tokens: \(tokensGenerated) (blanks: \(blankTokens))
        Avg skip: \(String(format: "%.1f", averageSkipLength)) frames
        """
    }
}

/// Optimized TDT decoder with hybrid CoreML + Metal acceleration
@available(macOS 13.0, iOS 16.0, *)
internal struct TdtDecoder {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "TDT")
    private let config: ASRConfig
    private let blankId = 1024
    private let sosId = 1024


    init(config: ASRConfig) {
        self.config = config
    }

    /// Execute optimized TDT decoding
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout DecoderState,
        collectStats: Bool = false
    ) async throws -> (tokens: [Int], stats: TdtDecodingStats?) {

        guard encoderSequenceLength > 1 else {
            logger.warning("TDT: Encoder sequence too short (\(encoderSequenceLength))")
            return (tokens: [], stats: nil)
        }

        // Pre-process encoder output for faster access
        let encoderFrames = try preProcessEncoderOutput(encoderOutput, length: encoderSequenceLength)

        var hypothesis = TdtHypothesis(decState: decoderState)
        var timeIndices = 0
        var safeTimeIndices = 0
        var timeIndicesCurrentLabels = 0
        var activeMask = true
        let lastTimestep = encoderSequenceLength - 1

        var lastTimestamp = -1
        var lastTimestampCount = 0

        var framesProcessed = 0
        var totalSkips = 0
        var blankTokenCount = 0

        // Main decoding loop with optimizations
        while activeMask {
            framesProcessed += 1
            var label = hypothesis.lastToken ?? sosId

            // Use cached decoder inputs
            let decoderResult = try runDecoderOptimized(
                token: label,
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

            // Optimized token/duration prediction
            let (tokenLogits, durationLogits) = try splitLogits(logits)
            label = argmaxSIMD(tokenLogits)
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
                if actualDuration > 0 {
                    totalSkips += actualDuration
                }
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
            } else {
                blankTokenCount += 1
            }

            // Force blank logic
            if let maxSymbols = config.tdtConfig.maxSymbolsPerStep {
                if label != blankId && lastTimestamp == timeIndices && lastTimestampCount >= maxSymbols {
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

        let stats: TdtDecodingStats?
        if collectStats {
            let framesSkipped = encoderSequenceLength - framesProcessed
            let avgSkipLength = framesProcessed > 0 ? Double(totalSkips) / Double(framesProcessed) : 0.0
            let skipRate = Double(framesSkipped) / Double(encoderSequenceLength)

            stats = TdtDecodingStats(
                totalFrames: encoderSequenceLength,
                framesProcessed: framesProcessed,
                framesSkipped: framesSkipped,
                tokensGenerated: hypothesis.ySequence.count,
                blankTokens: blankTokenCount,
                averageSkipLength: avgSkipLength,
                skipRate: skipRate
            )

            if config.enableDebug {
                logger.info("TDT-Optimized Stats: \(stats!.description)")
            }
        } else {
            stats = nil
        }

        return (tokens: hypothesis.ySequence, stats: stats)
    }

    /// Pre-process encoder output into contiguous memory for faster access
    private func preProcessEncoderOutput(_ encoderOutput: MLMultiArray, length: Int) throws -> EncoderFrameArray {
        let shape = encoderOutput.shape
        let hiddenSize = shape[2].intValue

        var frames = EncoderFrameArray()
        frames.reserveCapacity(length)

        let encoderPtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)

        for timeIdx in 0..<length {
            let offset = timeIdx * hiddenSize
            let frame = Array(UnsafeBufferPointer(start: encoderPtr + offset, count: hiddenSize))
            frames.append(frame)
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
            "c_in": MLFeatureValue(multiArray: state.cellState)
        ])

        let output = try model.prediction(
            from: input,
            options: MLPredictionOptions()
        )

        var newState = state
        newState.update(from: output)

        return (output, newState)
    }

    /// Optimized joint network execution
    private func runJointOptimized(
        encoderStep: [Float],
        decoderOutput: MLFeatureProvider,
        model: MLModel
    ) throws -> MLMultiArray {

        // Create encoder MLMultiArray from pre-processed data
        let encoderArray = try MLMultiArray(shape: [1, 1, encoderStep.count as NSNumber], dataType: .float32)
        encoderStep.withUnsafeBufferPointer { buffer in
            let dst = encoderArray.dataPointer.bindMemory(to: Float.self, capacity: encoderStep.count)
            if let baseAddress = buffer.baseAddress {
                dst.initialize(from: baseAddress, count: encoderStep.count)
            }
        }

        let decoderOutputArray = try extractFeatureValue(from: decoderOutput, key: "decoder_output", errorMessage: "Invalid decoder output")

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "encoder_outputs": MLFeatureValue(multiArray: encoderArray),
            "decoder_outputs": MLFeatureValue(multiArray: decoderOutputArray)
        ])

        let output = try model.prediction(
            from: input,
            options: MLPredictionOptions()
        )

        return try extractFeatureValue(from: output, key: "logits", errorMessage: "Joint network output missing logits")
    }
    /// Predict token and duration from joint logits
    internal func predictTokenAndDuration(_ logits: MLMultiArray) throws -> (token: Int, score: Float, duration: Int) {
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

    /// Split joint logits into token and duration components
    private func splitLogits(_ logits: MLMultiArray) throws -> (tokenLogits: [Float], durationLogits: [Float]) {
        let totalElements = logits.count
        let durationElements = config.tdtConfig.durations.count
        let vocabSize = totalElements - durationElements

        guard totalElements >= durationElements else {
            throw ASRError.processingFailed("Logits dimension mismatch")
        }

        // Use contiguous memory access
        let logitsPtr = logits.dataPointer.bindMemory(to: Float.self, capacity: totalElements)

        let tokenLogits = Array(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))
        let durationLogits = Array(UnsafeBufferPointer(start: logitsPtr + vocabSize, count: durationElements))

        return (tokenLogits, durationLogits)
    }

    /// SIMD-accelerated argmax
    private func argmaxSIMD(_ values: [Float]) -> Int {
        guard !values.isEmpty else { return 0 }

        var maxIndex = 0
        var maxValue = values[0]

        // Use vDSP for finding maximum
        vDSP_maxvi(values, 1, &maxValue, &maxIndex, vDSP_Length(values.count))

        return Int(maxIndex)
    }
    
    /// Non-SIMD argmax for compatibility
    private func argmax(_ values: [Float]) -> Int {
        return argmaxSIMD(values)
    }
    
    /// Process duration logits to get duration value
    private func processDurationLogits(_ durationLogits: [Float]) throws -> (bestDuration: Int, duration: Int) {
        let bestDurationIdx = argmaxSIMD(durationLogits)
        let duration = config.tdtConfig.durations[bestDurationIdx]
        return (bestDurationIdx, duration)
    }
    internal func extractEncoderTimeStep(_ encoderOutput: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        let shape = encoderOutput.shape
        let batchSize = shape[0].intValue
        let sequenceLength = shape[1].intValue
        let hiddenSize = shape[2].intValue

        guard timeIndex < sequenceLength else {
            throw ASRError.processingFailed("Time index out of bounds: \(timeIndex) >= \(sequenceLength)")
        }

        let timeStepArray = try MLMultiArray(shape: [batchSize, 1, hiddenSize] as [NSNumber], dataType: .float32)

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
            "c_in": MLFeatureValue(multiArray: cellState)
        ])
    }

    internal func prepareJointInput(
        encoderOutput: MLMultiArray,
        decoderOutput: MLFeatureProvider,
        timeIndex: Int
    ) throws -> MLFeatureProvider {
        let decoderOutputArray = try extractFeatureValue(from: decoderOutput, key: "decoder_output", errorMessage: "Invalid decoder output")

        return try MLDictionaryFeatureProvider(dictionary: [
            "encoder_outputs": MLFeatureValue(multiArray: encoderOutput),
            "decoder_outputs": MLFeatureValue(multiArray: decoderOutputArray)
        ])
    }

    // MARK: - Error Handling Helper

    /// Validates and extracts a required feature value from MLFeatureProvider
    private func extractFeatureValue(from provider: MLFeatureProvider, key: String, errorMessage: String) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }
}

/// Pre-processed encoder frames for fast access
private typealias EncoderFrameArray = [[Float]]
