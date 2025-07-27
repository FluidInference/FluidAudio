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
/// 
/// This decoder jointly predicts both tokens and their durations, enabling accurate
/// transcription of speech with varying speaking rates.
/// 
/// Based on NVIDIA's Parakeet TDT architecture from the NeMo toolkit.
/// The TDT model extends RNN-T by adding duration prediction, allowing
/// efficient frame-skipping during inference for faster decoding.
@available(macOS 13.0, iOS 16.0, *)
internal struct TdtDecoder {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "TDT")
    private let config: ASRConfig

    // Special token Indexes matching Parakeet TDT model's vocabulary (1024 word tokens)
    // OUTPUT from joint network during decoding
    // 0-1023 represents characters, numbers, punctuations
    // 1024 represents, BLANK or nonexistent
    private let blankId = 1024

    // sosId (Start-of-Sequence)
    // sosId is INPUT when there's no real previous token
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
        decoderState: inout DecoderState
    ) async throws -> [Int] {

        // TDT (Token-and-Duration Transducer) jointly predicts tokens AND their durations
        // This allows the decoder to "skip" frames when it predicts that a token spans multiple frames
        // For example: if "hello" spans 5 frames, instead of processing each frame individually,
        // TDT can predict the token once and skip ahead 5 frames, making decoding much faster
        
        // We need at least 2 frames because:
        // 1. Frame 0: Initial state, predicts first token/duration
        // 2. Frame 1+: Needed to validate duration predictions and continue decoding
        // With only 1 frame, there's no way to verify if the duration prediction makes sense
        guard encoderSequenceLength > 1 else {
            logger.warning("TDT: Encoder sequence too short (\(encoderSequenceLength))")
            return []
        }

        // Initialize hypothesis with the decoder's current state
        // This maintains context across multiple audio chunks in streaming scenarios
        var hypothesis = TdtHypothesis(decState: decoderState)

        // timeIdx tracks our position in the encoder output sequence
        // Unlike traditional decoders that process every frame sequentially,
        // TDT can jump forward based on duration predictions
        var timeIdx = 0

        // Main decoding loop - continues until we've processed the entire audio sequence
        while timeIdx < encoderSequenceLength {
            // Process the current time step, which may:
            // 1. Predict multiple tokens (up to maxSymbolsPerFrame)
            // 2. Predict blank tokens (no speech)
            // 3. Skip ahead multiple frames based on duration predictions
            let result = try await processTimeStep(
                timeIdx: timeIdx,
                encoderOutput: encoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                decoderModel: decoderModel,
                jointModel: jointModel,
                hypothesis: &hypothesis
            )

            // result contains the next timeIdx to process
            // This could be timeIdx+1 (normal advance) or timeIdx+N (duration skip)
            timeIdx = result
        }

        // Return the decoded token sequence (vocabulary indices)
        // These will be converted to text by the AsrManager using the vocabulary
        return hypothesis.ySequence
    }

    /// Process a single time step in the TDT decoding
    private func processTimeStep(
        timeIdx: Int,
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        hypothesis: inout TdtHypothesis
    ) async throws -> Int {

        let encoderStep = try extractEncoderTimeStep(encoderOutput, timeIndex: timeIdx)
        let maxSymbolsPerFrame = config.tdtConfig.maxSymbolsPerStep ?? config.maxSymbolsPerFrame

        var symbolsAdded = 0
        var nextTimeIdx = timeIdx

        while symbolsAdded < maxSymbolsPerFrame {
            // processSymbol returns an optional Int:
            // - nil: predicted a regular token (no frame skip)
            // - Some(n): predicted a blank token with duration n (skip n frames)
            let result = try await processSymbol(
                encoderStep: encoderStep,
                timeIdx: timeIdx,
                decoderModel: decoderModel,
                jointModel: jointModel,
                hypothesis: &hypothesis
            )

            symbolsAdded += 1

            // Swift's "if let" pattern for optional binding:
            // - If result is nil (no duration), this block doesn't execute
            // - If result has a value, it's unwrapped and assigned to 'skip'
            // This is NOT a variable declaration - it's pattern matching!
            if let skip = result {
                // We only get here if processSymbol returned a duration value
                // 'skip' is the unwrapped duration value from the optional
                nextTimeIdx = calculateNextTimeIndex(
                    currentIdx: timeIdx,
                    skip: skip,
                    sequenceLength: encoderSequenceLength
                )
                break
            }
            // If result was nil, we continue the loop to predict more tokens
        }

        // Default to next frame if no skip occurred
        return nextTimeIdx == timeIdx ? timeIdx + 1 : nextTimeIdx
    }

    /// Process a single symbol prediction
    private func processSymbol(
        encoderStep: MLMultiArray,
        timeIdx: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        hypothesis: inout TdtHypothesis
    ) async throws -> Int? {

        // Run decoder with current token
        let targetToken = hypothesis.lastToken ?? sosId
        let decoderState = hypothesis.decState ?? DecoderState()

        let decoderOutput = try runDecoder(
            token: targetToken,
            state: decoderState,
            model: decoderModel
        )

        // Run joint network
        let logits = try runJointNetwork(
            encoderStep: encoderStep,
            decoderOutput: decoderOutput.output,
            model: jointModel
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
        model: MLModel
    ) throws -> (output: MLFeatureProvider, newState: DecoderState) {

        let input = try prepareDecoderInput(
            targetToken: token,
            hiddenState: state.hiddenState,
            cellState: state.cellState
        )

        let output = try model.prediction(
            from: input,
            options: MLPredictionOptions()
        )

        var newState = state
        newState.update(from: output)

        return (output, newState)
    }

    /// Run joint network
    private func runJointNetwork(
        encoderStep: MLMultiArray,
        decoderOutput: MLFeatureProvider,
        model: MLModel
    ) throws -> MLMultiArray {

        let input = try prepareJointInput(
            encoderOutput: encoderStep,
            decoderOutput: decoderOutput,
            timeIndex: 0  // Already extracted time step
        )

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
