//
//  TdtBeamSearch.swift
//  FluidAudio
//
//  Beam search implementation for TDT decoder
//

import Accelerate
import CoreML
import Foundation
import OSLog

/// Beam hypothesis for tracking multiple decoding paths
struct BeamHypothesis: Sendable {
    var score: Float = 0.0
    var tokens: [Int] = []
    var decoderState: DecoderState?
    var timestamps: [Int] = []
    var timeIdx: Int = 0
    var lastToken: Int?

    /// Create a copy with updated values
    func copy(
        addToken: Int? = nil,
        addScore: Float = 0,
        newState: DecoderState? = nil,
        addTimestamp: Int? = nil,
        newTimeIdx: Int? = nil
    ) -> BeamHypothesis {
        var copy = self
        if let token = addToken {
            copy.tokens.append(token)
            copy.lastToken = token
        }
        copy.score += addScore
        if let state = newState {
            copy.decoderState = state
        }
        if let timestamp = addTimestamp {
            copy.timestamps.append(timestamp)
        }
        if let idx = newTimeIdx {
            copy.timeIdx = idx
        }
        return copy
    }
}

@available(macOS 13.0, iOS 16.0, *)
internal struct TdtBeamSearch {

    private let logger = Logger(subsystem: "com.fluidinfluence.asr", category: "BeamSearch")
    private let config: ASRConfig
    private let beamWidth: Int
    private let predictionOptions = AsrModels.optimizedPredictionOptions()

    // Special tokens (same as TdtDecoder)
    private let blankId = 8192
    private let sosId = 8192

    init(config: ASRConfig, beamWidth: Int = 5) {
        self.config = config
        self.beamWidth = beamWidth
    }

    /// Execute beam search decoding
    func decode(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout DecoderState
    ) async throws -> [Int] {

        logger.debug("Beam search decode: beamWidth=\(beamWidth), encoderLength=\(encoderSequenceLength)")

        guard encoderSequenceLength > 1 else {
            logger.warning("Encoder sequence too short (\(encoderSequenceLength))")
            return []
        }

        // Pre-process encoder output
        let encoderFrames = try preProcessEncoderOutput(encoderOutput, length: encoderSequenceLength)

        // Initialize beam with single hypothesis
        var beam: [BeamHypothesis] = [
            BeamHypothesis(
                score: 0.0,
                tokens: [],
                decoderState: decoderState,
                timestamps: [],
                timeIdx: 0,
                lastToken: decoderState.lastToken
            )
        ]

        let lastTimestep = encoderSequenceLength - 1
        var maxTokensGenerated = encoderSequenceLength * (config.tdtConfig.maxSymbolsPerStep ?? 3)
        var iterationCount = 0
        let maxIterations = encoderSequenceLength * 10  // Safety limit

        // Main beam search loop
        while !beam.isEmpty && beam[0].timeIdx < encoderSequenceLength && iterationCount < maxIterations {
            iterationCount += 1
            var nextBeam: [BeamHypothesis] = []

            for hyp in beam {
                // Skip if this hypothesis has reached the end
                if hyp.timeIdx >= encoderSequenceLength {
                    nextBeam.append(hyp)
                    continue
                }

                // Skip if generated too many tokens
                if hyp.tokens.count >= maxTokensGenerated {
                    nextBeam.append(hyp)
                    continue
                }

                let currentToken = hyp.lastToken ?? sosId

                // Run decoder
                let decoderResult = try runDecoderOptimized(
                    token: currentToken,
                    state: hyp.decoderState ?? decoderState,
                    model: decoderModel
                )

                // Get encoder frame
                let safeTimeIdx = min(hyp.timeIdx, lastTimestep)
                let encoderStep = encoderFrames[safeTimeIdx]

                // Run joint network
                let logits = try runJointOptimized(
                    encoderStep: encoderStep,
                    decoderOutput: decoderResult.output,
                    model: jointModel
                )

                // Split logits
                let (tokenLogits, durationLogits) = try splitLogits(logits)

                // Apply softmax
                let tokenProbs = softmax(tokenLogits)
                let durationProbs = softmax(durationLogits)

                // Apply confidence threshold (similar to greedy decoder)
                let maxTokenProb = tokenProbs.max() ?? 0
                let initialBestTokenIdx = argmax(tokenProbs)

                // If confidence is too low, force blank token
                var effectiveTokenProbs = tokenProbs
                if maxTokenProb < 0.1 && initialBestTokenIdx != blankId {
                    // Set blank token to high probability
                    effectiveTokenProbs = [Float](repeating: 0.0, count: tokenProbs.count)
                    effectiveTokenProbs[blankId] = 1.0
                }

                // Get most likely duration
                let bestDurationIdx = argmax(durationProbs)
                var duration = config.tdtConfig.durations[bestDurationIdx]

                // For beam search, use greedy-like approach but keep top alternatives
                let bestTokenIdx = argmax(effectiveTokenProbs)
                let maxProb = effectiveTokenProbs[bestTokenIdx]

                // Explore top token + alternatives if confidence is high
                var candidateTokens: [(Int, Float)] = [(bestTokenIdx, maxProb)]

                // Only add alternatives if we're confident about the best choice
                if maxProb > 0.5 {
                    let alternatives = getTopK(effectiveTokenProbs, k: min(2, beamWidth))
                        .filter { $0.0 != bestTokenIdx && $0.1 > 0.1 }
                    candidateTokens.append(contentsOf: alternatives)
                }

                // Explore each candidate token
                for (tokenId, tokenProb) in candidateTokens {
                    // Simple scoring: just accumulate log probabilities
                    let logProb = log(max(tokenProb, 1e-10))
                    let newScore = hyp.score + logProb

                    // Handle blank token
                    if tokenId == blankId {
                        // Blank token - advance time without adding token
                        if duration == 0 {
                            duration = 1  // Ensure we advance for blank
                        }

                        let newHyp = BeamHypothesis(
                            score: newScore,
                            tokens: hyp.tokens,
                            decoderState: decoderResult.newState,
                            timestamps: hyp.timestamps,
                            timeIdx: hyp.timeIdx + duration,
                            lastToken: tokenId
                        )
                        nextBeam.append(newHyp)
                    } else {
                        // Non-blank token - add to sequence
                        var newTokens = hyp.tokens
                        newTokens.append(tokenId)
                        var newTimestamps = hyp.timestamps
                        newTimestamps.append(hyp.timeIdx)

                        let newHyp = BeamHypothesis(
                            score: newScore,
                            tokens: newTokens,
                            decoderState: decoderResult.newState,
                            timestamps: newTimestamps,
                            timeIdx: hyp.timeIdx + duration,
                            lastToken: tokenId
                        )
                        nextBeam.append(newHyp)
                    }
                }
            }

            // Prune beam to top k hypotheses
            beam = pruneBeam(nextBeam, beamWidth: beamWidth)

            // Check for early termination if all hypotheses have finished
            if beam.allSatisfy({ $0.timeIdx >= encoderSequenceLength }) {
                break
            }

            // Debug logging
            if config.enableDebug && beam[0].tokens.count % 10 == 0 {
                logger.debug("Beam search: tokens=\(beam[0].tokens.count), score=\(beam[0].score)")
            }
        }

        // Get best hypothesis
        guard let bestHyp = beam.first else {
            return []
        }

        // Update decoder state
        if let finalState = bestHyp.decoderState {
            decoderState = finalState
        }
        decoderState.lastToken = bestHyp.lastToken

        logger.info("Beam search completed: tokens=\(bestHyp.tokens.count), score=\(bestHyp.score)")

        return bestHyp.tokens
    }

    /// Prune beam to top k hypotheses
    private func pruneBeam(_ hypotheses: [BeamHypothesis], beamWidth: Int) -> [BeamHypothesis] {
        // Sort by score (descending) - using log probability
        // Note: Scores are negative (log probabilities), so higher = better
        let sorted = hypotheses.sorted { $0.score > $1.score }

        // Keep top k
        return Array(sorted.prefix(beamWidth))
    }

    /// Get top k tokens with their probabilities
    private func getTopK(_ probs: [Float], k: Int) -> [(Int, Float)] {
        // Create index-value pairs
        let indexed = probs.enumerated().map { ($0.offset, $0.element) }

        // Sort by probability (descending)
        let sorted = indexed.sorted { $0.1 > $1.1 }

        // Return top k
        return Array(sorted.prefix(k))
    }

    /// Apply softmax to convert logits to probabilities
    private func softmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }

        // Find max for numerical stability
        let maxLogit = logits.max() ?? 0

        // Compute exp(x - max)
        var expValues = [Float](repeating: 0, count: logits.count)
        var sum: Float = 0

        for i in 0..<logits.count {
            expValues[i] = exp(logits[i] - maxLogit)
            sum += expValues[i]
        }

        // Normalize
        if sum > 0 {
            for i in 0..<expValues.count {
                expValues[i] /= sum
            }
        }

        return expValues
    }

    /// Find index of maximum value
    private func argmax(_ values: [Float]) -> Int {
        guard !values.isEmpty else { return 0 }

        var maxValue: Float = 0
        var maxIndex: vDSP_Length = 0

        values.withUnsafeBufferPointer { buffer in
            vDSP_maxvi(buffer.baseAddress!, 1, &maxValue, &maxIndex, vDSP_Length(values.count))
        }

        return Int(maxIndex)
    }

    // MARK: - Helper methods (similar to TdtDecoder)

    /// Pre-process encoder output
    private func preProcessEncoderOutput(
        _ encoderOutput: MLMultiArray, length: Int
    ) throws -> [[Float]] {
        let shape = encoderOutput.shape
        guard shape.count >= 3 else {
            throw ASRError.processingFailed("Invalid encoder output shape: \(shape)")
        }

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

        var frames: [[Float]] = []
        frames.reserveCapacity(length)

        if encoderOutput.dataType == .float32 {
            let floatPtr = encoderOutput.dataPointer.bindMemory(
                to: Float.self, capacity: encoderOutput.count)

            for timeIdx in 0..<length {
                let startIdx = timeIdx * hiddenSize
                let frameView = UnsafeBufferPointer(
                    start: floatPtr + startIdx,
                    count: hiddenSize
                )
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

    /// Run decoder model
    private func runDecoderOptimized(
        token: Int,
        state: DecoderState,
        model: MLModel
    ) throws -> (output: MLFeatureProvider, newState: DecoderState) {

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

    /// Run joint network
    private func runJointOptimized(
        encoderStep: [Float],
        decoderOutput: MLFeatureProvider,
        model: MLModel
    ) throws -> MLMultiArray {

        let encoderArray = try ANEOptimizer.createANEAlignedArray(
            shape: [1, 1, 1024],
            dataType: .float32
        )

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

    /// Split joint logits into token and duration components
    private func splitLogits(
        _ logits: MLMultiArray
    ) throws -> (tokenLogits: [Float], durationLogits: [Float]) {
        let totalElements = logits.count
        let durationElements = config.tdtConfig.durations.count
        let vocabSize = totalElements - durationElements

        guard totalElements >= durationElements else {
            throw ASRError.processingFailed(
                "Logits dimension mismatch: got \(totalElements), need at least \(durationElements)")
        }

        let logitsPtr = logits.dataPointer.bindMemory(to: Float.self, capacity: totalElements)

        let tokenLogits = ContiguousArray(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))
        let durationLogits = ContiguousArray(
            UnsafeBufferPointer(start: logitsPtr + vocabSize, count: durationElements))

        return (Array(tokenLogits), Array(durationLogits))
    }

    /// Extract feature value from provider
    private func extractFeatureValue(
        from provider: MLFeatureProvider, key: String, errorMessage: String
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }
}
