//
//  TdtDecoder.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import CoreML
import Foundation
import OSLog

/// Token-and-Duration Transducer (TDT) configuration
public struct TdtConfig: Sendable {
    public let durations: [Int]
    public let includeTokenDuration: Bool
    public let includeDurationConfidence: Bool
    public let maxSymbolsPerStep: Int?

    public static let `default` = TdtConfig()

    public init(
        durations: [Int] = [0, 1, 2, 3, 4],
        includeTokenDuration: Bool = true,
        includeDurationConfidence: Bool = false,
        maxSymbolsPerStep: Int? = nil
    ) {
        self.durations = durations
        self.includeTokenDuration = includeTokenDuration
        self.includeDurationConfidence = includeDurationConfidence
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
    var lastToken: Int?
}

/// Token-and-Duration Transducer (TDT) decoder implementation
/// This decoder jointly predicts both tokens and their durations, enabling accurate
/// transcription of speech with varying speaking rates.
@available(macOS 13.0, iOS 16.0, *)
internal struct TdtDecoder {
    
    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "TDT")
    private let config: ASRConfig
    private let blankId = 1024
    private let sosId = 1024
    
    init(config: ASRConfig) {
        self.config = config
    }
    
    /// Execute TDT decoding on encoder output
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout DecoderState,
        predictionOptions: MLPredictionOptions?
    ) async throws -> [Int] {
        
        guard encoderSequenceLength > 1 else {
            logger.warning("TDT: Encoder sequence too short (\(encoderSequenceLength))")
            return []
        }
        
        var hypothesis = TdtHypothesis()
        hypothesis.decState = decoderState
        hypothesis.lastToken = nil
        
        var timeIdx = 0
        
        while timeIdx < encoderSequenceLength {
            let result = try await processTimeStep(
                timeIdx: timeIdx,
                encoderOutput: encoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                decoderModel: decoderModel,
                jointModel: jointModel,
                hypothesis: &hypothesis,
                predictionOptions: predictionOptions
            )
            
            timeIdx = result
        }
        
        return hypothesis.ySequence
    }
    
    /// Process a single time step in the TDT decoding
    private func processTimeStep(
        timeIdx: Int,
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        hypothesis: inout TdtHypothesis,
        predictionOptions: MLPredictionOptions?
    ) async throws -> Int {
        
        let encoderStep = try extractEncoderTimeStep(encoderOutput, timeIndex: timeIdx)
        let maxSymbolsPerFrame = config.tdtConfig.maxSymbolsPerStep ?? config.maxSymbolsPerFrame
        
        var symbolsAdded = 0
        var nextTimeIdx = timeIdx
        
        while symbolsAdded < maxSymbolsPerFrame {
            let result = try await processSymbol(
                encoderStep: encoderStep,
                timeIdx: timeIdx,
                decoderModel: decoderModel,
                jointModel: jointModel,
                hypothesis: &hypothesis,
                predictionOptions: predictionOptions
            )
            
            symbolsAdded += 1
            
            // Handle time advancement based on duration prediction
            if let skip = result {
                nextTimeIdx = calculateNextTimeIndex(
                    currentIdx: timeIdx,
                    skip: skip,
                    sequenceLength: encoderSequenceLength
                )
                break
            }
            
            // No skip, check if we should continue processing symbols
            if symbolsAdded >= maxSymbolsPerFrame {
                nextTimeIdx = timeIdx + 1
            }
        }
        
        return nextTimeIdx
    }
    
    /// Process a single symbol prediction
    private func processSymbol(
        encoderStep: MLMultiArray,
        timeIdx: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        hypothesis: inout TdtHypothesis,
        predictionOptions: MLPredictionOptions?
    ) async throws -> Int? {
        
        // Run decoder with current token
        let targetToken = hypothesis.lastToken ?? sosId
        let decoderState = hypothesis.decState ?? DecoderState()
        
        let decoderOutput = try runDecoder(
            token: targetToken,
            state: decoderState,
            model: decoderModel,
            options: predictionOptions
        )
        
        // Run joint network
        let logits = try runJointNetwork(
            encoderStep: encoderStep,
            decoderOutput: decoderOutput.output,
            model: jointModel,
            options: predictionOptions
        )
        
        // Predict token and duration
        let prediction = try predictTokenAndDuration(logits)
        
        // Update hypothesis if non-blank token
        if prediction.token != blankId {
            updateHypothesis(
                &hypothesis,
                token: prediction.token,
                score: prediction.score,
                duration: prediction.duration,
                timeIdx: timeIdx,
                decoderState: decoderOutput.newState
            )
        }
        
        // Return skip frames if duration prediction indicates time advancement
        return prediction.duration > 0 ? prediction.duration : nil
    }
    
    /// Run decoder model
    private func runDecoder(
        token: Int,
        state: DecoderState,
        model: MLModel,
        options: MLPredictionOptions?
    ) throws -> (output: MLFeatureProvider, newState: DecoderState) {
        
        let input = try prepareDecoderInput(
            targetToken: token,
            hiddenState: state.hiddenState,
            cellState: state.cellState
        )
        
        let output = try model.prediction(
            from: input,
            options: options ?? MLPredictionOptions()
        )
        
        var newState = state
        newState.update(from: output)
        
        return (output, newState)
    }
    
    /// Run joint network
    private func runJointNetwork(
        encoderStep: MLMultiArray,
        decoderOutput: MLFeatureProvider,
        model: MLModel,
        options: MLPredictionOptions?
    ) throws -> MLMultiArray {
        
        let input = try prepareJointInput(
            encoderOutput: encoderStep,
            decoderOutput: decoderOutput,
            timeIndex: 0  // Already extracted time step
        )
        
        let output = try model.prediction(
            from: input,
            options: options ?? MLPredictionOptions()
        )
        
        guard let logits = output.featureValue(for: "logits")?.multiArrayValue else {
            throw ASRError.processingFailed("Joint network output missing logits")
        }
        
        return logits
    }
    
    /// Predict token and duration from joint logits
    private func predictTokenAndDuration(_ logits: MLMultiArray) throws -> (token: Int, score: Float, duration: Int) {
        let (tokenLogits, durationLogits) = try splitLogits(logits)
        
        let bestToken = argmax(tokenLogits)
        let tokenScore = tokenLogits[bestToken]
        
        let (_, duration) = try processDurationLogits(durationLogits)
        
        return (token: bestToken, score: tokenScore, duration: duration)
    }
    
    /// Update hypothesis with new token
    private func updateHypothesis(
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
    private func calculateNextTimeIndex(currentIdx: Int, skip: Int, sequenceLength: Int) -> Int {
        let actualSkip = min(skip, 4)
        
        // Special handling for short sequences
        if sequenceLength < 10 && actualSkip > 2 {
            return min(currentIdx + 2, sequenceLength)
        }
        
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
        
        let tokenLogits = (0..<vocabSize).map { logits[$0].floatValue }
        let durationLogits = (vocabSize..<totalElements).map { logits[$0].floatValue }
        
        return (tokenLogits, durationLogits)
    }
    
    /// Process duration logits and return duration index with skip value
    private func processDurationLogits(_ logits: [Float]) throws -> (index: Int, skip: Int) {
        let maxIndex = argmax(logits)
        let durations = config.tdtConfig.durations
        guard maxIndex < durations.count else {
            throw ASRError.processingFailed("Duration index out of bounds")
        }
        return (maxIndex, durations[maxIndex])
    }
    
    /// Find argmax in a float array
    private func argmax(_ values: [Float]) -> Int {
        guard !values.isEmpty else { return 0 }
        return values.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
    }
    
    private func extractEncoderTimeStep(_ encoderOutput: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
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
    
    private func prepareDecoderInput(
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
    
    private func prepareJointInput(
        encoderOutput: MLMultiArray,
        decoderOutput: MLFeatureProvider,
        timeIndex: Int
    ) throws -> MLFeatureProvider {
        guard let decoderOutputArray = decoderOutput.featureValue(for: "decoder_output")?.multiArrayValue else {
            throw ASRError.processingFailed("Invalid decoder output")
        }
        
        return try MLDictionaryFeatureProvider(dictionary: [
            "encoder_outputs": MLFeatureValue(multiArray: encoderOutput),
            "decoder_outputs": MLFeatureValue(multiArray: decoderOutputArray)
        ])
    }
}