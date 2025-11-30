/// RNNT (Recurrent Neural Network Transducer) Decoder
///
/// This decoder implements standard RNNT decoding for Parakeet Realtime EOU 120M.
/// Unlike TDT, RNNT does not predict durations - it simply predicts tokens until blank.
///
/// Algorithm flow:
/// 1. Process audio frame through encoder (done before this decoder)
/// 2. For each encoder frame:
///    a. Run decoder to get language model context
///    b. Combine encoder frame + decoder state in joint network
///    c. If blank token: advance to next encoder frame
///    d. If non-blank: emit token, update decoder LSTM, stay on same frame
///    e. Check for <EOU> token to detect end-of-utterance
/// 3. Repeat until all encoder frames processed
///
/// Key differences from TDT:
/// - No duration prediction (always advances 1 frame on blank)
/// - <EOU> token detection for utterance boundaries
/// - Simpler joint network output (just token_id and token_prob)

import Accelerate
import CoreML
import Foundation
import OSLog

internal struct RnntDecoder {

    /// Joint model decision for a single encoder/decoder step
    private struct JointDecision {
        let token: Int
        let probability: Float
    }

    private let logger = AppLogger(category: "RNNT")
    private let config: RnntConfig
    private let predictionOptions = AsrModels.optimizedPredictionOptions()

    init(config: RnntConfig = .parakeetEOU) {
        self.config = config
    }

    /// Reusable input provider for joint model
    private final class ReusableJointInput: NSObject, MLFeatureProvider {
        let encoderStep: MLMultiArray
        let decoderStep: MLMultiArray

        init(encoderStep: MLMultiArray, decoderStep: MLMultiArray) {
            self.encoderStep = encoderStep
            self.decoderStep = decoderStep
            super.init()
        }

        var featureNames: Set<String> {
            ["encoder_step", "decoder_step"]
        }

        func featureValue(for featureName: String) -> MLFeatureValue? {
            switch featureName {
            case "encoder_step":
                return MLFeatureValue(multiArray: encoderStep)
            case "decoder_step":
                return MLFeatureValue(multiArray: decoderStep)
            default:
                return nil
            }
        }
    }

    /// Execute RNNT decoding and return tokens with emission timestamps
    ///
    /// - Parameters:
    ///   - encoderOutput: 3D tensor [batch=1, hidden_dim=512, time_frames] from encoder
    ///   - encoderSequenceLength: Number of valid frames in encoderOutput
    ///   - decoderModel: CoreML model for LSTM decoder
    ///   - jointModel: CoreML model combining encoder+decoder features
    ///   - decoderState: LSTM hidden/cell states
    ///   - globalFrameOffset: Offset for timestamp calculation in streaming
    ///
    /// - Returns: RnntHypothesis with tokens, timestamps, and updated state
    func decodeWithTimings(
        encoderOutput: MLMultiArray,
        encoderSequenceLength: Int,
        decoderModel: MLModel,
        jointModel: MLModel,
        decoderState: inout RnntDecoderState,
        globalFrameOffset: Int = 0,
        partialHypothesis: RnntHypothesis? = nil
    ) async throws -> RnntHypothesis {
        // Early exit for very short audio
        guard encoderSequenceLength > 0 else {
            return partialHypothesis ?? RnntHypothesis(decState: decoderState)
        }

        var hypothesis: RnntHypothesis
        
        if let partial = partialHypothesis {
            // Restore state from partial hypothesis (NeMo streaming pattern)
            hypothesis = partial
            // Ensure current decoder state is linked
            hypothesis.decState = decoderState
        } else {
            hypothesis = RnntHypothesis(decState: decoderState)
            hypothesis.lastToken = decoderState.lastToken
        }

        // Encoder output shape: [1, 512, T] - need to access frames along time axis
        let encoderShape = encoderOutput.shape.map { $0.intValue }
        let encoderHiddenSize = encoderShape[1]  // 512
        let encoderTimeFrames = min(encoderShape[2], encoderSequenceLength)

        // let encoderStrides = encoderOutput.strides.map { $0.intValue }
        // logger.debug(
        //     "RNNT decode: encoder shape=\(encoderShape), strides=\(encoderStrides), seqLen=\(encoderSequenceLength), frames=\(encoderTimeFrames)"
        // )

        // Debug: print encoder layout info
        // let encPtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: min(1000, encoderOutput.count))
        // var firstVals: [Float] = []
        // for i in 0..<min(5, encoderOutput.count) {
        //     firstVals.append(encPtr[i])
        // }
        // logger.debug("First 5 encoder values (raw memory): \(firstVals)")

        // Check what frame 0 looks like with different interpretations
        // If shape is [1, 512, 189] with strides [S0, S1, S2], then [0, h, t] = h*S1 + t*S2
        // var frame0Values: [Float] = []
        // for h in 0..<5 {
        //     // Shape [1, 512, 189] -> [batch, hidden, time]
        //     // For frame 0: h*strides[1] + 0*strides[2]
        //     let idx = h * 512 + 0 // Assuming stride
        //     // frame0Values.append(encPtr[idx])
        // }
        // logger.debug("Frame 0 hidden dims 0-4 (shape [1,512,189]): \(frame0Values)")

        // Preallocate arrays for decoder input
        let reusableTargetArray = try MLMultiArray(shape: [1, 1] as [NSNumber], dataType: .int32)
        let reusableTargetLengthArray = try MLMultiArray(shape: [1] as [NSNumber], dataType: .int32)
        reusableTargetLengthArray[0] = NSNumber(value: 1)

        // Preallocate joint input tensors
        // NOTE: Use standard MLMultiArray (not ANE-aligned) because the joint model
        // expects contiguous data without stride padding
        let reusableEncoderStep = try MLMultiArray(
            shape: [1, NSNumber(value: encoderHiddenSize), 1],
            dataType: .float32
        )
        let reusableDecoderStep = try MLMultiArray(
            shape: [1, NSNumber(value: config.decoderHiddenSize), 1],
            dataType: .float32
        )
        let jointInput = ReusableJointInput(
            encoderStep: reusableEncoderStep,
            decoderStep: reusableDecoderStep
        )

        // Preallocate output backings for joint model
        let tokenIdBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .int32)
        let tokenProbBacking = try MLMultiArray(shape: [1, 1, 1] as [NSNumber], dataType: .float32)

        // Initialize decoder state if needed
        if decoderState.lastToken == nil && decoderState.predictorOutput == nil {
            let zero = RnntDecoderState.make(hiddenSize: config.decoderHiddenSize)
            decoderState.hiddenState.copyData(from: zero.hiddenState)
            decoderState.cellState.copyData(from: zero.cellState)
        }

        // Initialize decoder projection if no prior context
        // NeMo's decoder.predict(y=None, add_sos=True) processes TWO zero embeddings:
        //   time=0: LSTM(zero_input, zero_state) -> t=0 output
        //   time=1: LSTM(zero_input, t=0_state) -> t=1 output (this is what we need)
        // CoreML model only outputs 1 timestep per call, so we call it twice
        if decoderState.predictorOutput == nil && hypothesis.lastToken == nil {
            // First call: process blank with zero state -> get t=0 output and updated state
            // NeMo's greedy decoding initializes by running the RNN once with a zero input (add_sos=False).
            // We match this by running the decoder once with the blank token.
            let firstCall = try runDecoder(
                token: config.blankId,
                state: decoderState,
                model: decoderModel,
                targetArray: reusableTargetArray,
                targetLengthArray: reusableTargetLengthArray
            )

            let proj = try extractFeatureValue(
                from: firstCall.output,
                key: "decoder",
                errorMessage: "Invalid decoder output"
            )
            
            // Now we have the t=0 output which matches NeMo's predict(None)
            let timeSteps = proj.shape[2].intValue
            let sosProjection = try extractTimeStep(proj, timeIndex: timeSteps - 1)
            decoderState.predictorOutput = sosProjection
            hypothesis.decState = firstCall.newState
        }

        var tokensProcessedThisChunk = 0
        var frameIdx = 0

        // Main RNNT decoding loop
        while frameIdx < encoderTimeFrames {
            // Copy encoder frame to reusable buffer
            try copyEncoderFrame(
                from: encoderOutput,
                frameIndex: frameIdx,
                hiddenSize: encoderHiddenSize,
                into: reusableEncoderStep
            )

            // Inner loop: emit tokens until blank
            var symbolsThisFrame = 0
            while symbolsThisFrame < config.maxSymbolsPerStep {
                let label = hypothesis.lastToken ?? config.blankId
                let stateToUse = hypothesis.decState ?? decoderState

                // Get decoder output
                let decoderResult: (output: MLFeatureProvider, newState: RnntDecoderState)
                if let cached = decoderState.predictorOutput {
                    // Use cached zero projection for first step only
                    let provider = try MLDictionaryFeatureProvider(dictionary: [
                        "decoder": MLFeatureValue(multiArray: cached)
                    ])
                    decoderResult = (output: provider, newState: stateToUse)
                    // Do NOT clear cache here - we reuse it until non-blank token
                    // decoderState.predictorOutput = nil
                } else {
                    decoderResult = try runDecoder(
                        token: label,
                        state: stateToUse,
                        model: decoderModel,
                        targetArray: reusableTargetArray,
                        targetLengthArray: reusableTargetLengthArray
                    )
                }

                // Prepare decoder projection
                let decoderProjection = try extractFeatureValue(
                    from: decoderResult.output,
                    key: "decoder",
                    errorMessage: "Invalid decoder output"
                )
                if frameIdx == 0 && symbolsThisFrame == 0 {
                    // logger.debug("Decoder projection shape: \(decoderProjection.shape.map { $0.intValue }), strides: \(decoderProjection.strides.map { $0.intValue })")
                }
                try copyDecoderProjection(decoderProjection, into: reusableDecoderStep)

                // Run joint network
                let decision = try runJoint(
                    model: jointModel,
                    inputProvider: jointInput,
                    tokenIdBacking: tokenIdBacking,
                    tokenProbBacking: tokenProbBacking,
                    debugFrame: nil  // Disable debug output
                )

                // Debug frames at various points (disabled)
                if false {
                    // Check encoder step values - ALL values for statistics
                    let encStepPtr = reusableEncoderStep.dataPointer.bindMemory(
                        to: Float.self, capacity: encoderHiddenSize)
                    var encVals: [Float] = []
                    var encMin: Float = .infinity
                    var encMax: Float = -.infinity
                    var encSum: Float = 0
                    for i in 0..<encoderHiddenSize {
                        let v = encStepPtr[i]
                        if i < 5 { encVals.append(v) }
                        encMin = min(encMin, v)
                        encMax = max(encMax, v)
                        encSum += v
                    }
                    let encMean = encSum / Float(encoderHiddenSize)

                    // Check decoder step values - ALL values for statistics
                    let decStepPtr = reusableDecoderStep.dataPointer.bindMemory(
                        to: Float.self, capacity: config.decoderHiddenSize)
                    var decVals: [Float] = []
                    var decMin: Float = .infinity
                    var decMax: Float = -.infinity
                    var decSum: Float = 0
                    for i in 0..<config.decoderHiddenSize {
                        let v = decStepPtr[i]
                        if i < 5 { decVals.append(v) }
                        decMin = min(decMin, v)
                        decMax = max(decMax, v)
                        decSum += v
                    }
                    let decMean = decSum / Float(config.decoderHiddenSize)

                    print(
                        "[DEBUG] Frame \(frameIdx): enc_step[:5]=\(encVals) | min=\(encMin), max=\(encMax), mean=\(encMean)"
                    )
                    print(
                        "[DEBUG] Frame \(frameIdx): dec_step[:5]=\(decVals) | min=\(decMin), max=\(decMax), mean=\(decMean)"
                    )
                    print(
                        "[DEBUG] Frame \(frameIdx): joint decision token=\(decision.token), prob=\(decision.probability), blank=\(config.blankId)"
                    )
                }

                // Log every frame to debug iteration
                if symbolsThisFrame == 0 {
                    // logger.debug("Frame \(frameIdx): token=\(decision.token), prob=\(decision.probability) \(decision.token == config.blankId ? "BLANK" : "NON-BLANK")")
                }
                
                // Check for blank token
                if decision.token == config.blankId {
                    // Blank = done with this frame, move to next
                    break
                }

                // Non-blank token: emit it
                tokensProcessedThisChunk += 1
                if tokensProcessedThisChunk > config.maxTokensPerChunk {
                    break
                }

                hypothesis.ySequence.append(decision.token)
                hypothesis.score += decision.probability
                hypothesis.timestamps.append(frameIdx + globalFrameOffset)
                hypothesis.tokenConfidences.append(decision.probability)
                hypothesis.lastToken = decision.token

                // Check for EOU token
                if decision.token == config.eouTokenId {
                    hypothesis.eouDetected = true
                    decoderState.markEndOfUtterance()
                }

                // Update decoder state for next iteration
                // NeMo: hypothesis.dec_state = hidden_prime (state after processing lastToken)
                // In next iteration, we will run decoder(token=decision.token, state=hypothesis.decState)
                hypothesis.decState = decoderResult.newState
                
                // Clear cached projection because we have a new token
                decoderState.predictorOutput = nil

                symbolsThisFrame += 1
            }

            // Move to next encoder frame
            frameIdx += 1

            // Safety check for max tokens
            if tokensProcessedThisChunk > config.maxTokensPerChunk {
                break
            }
        }

        // Update final state
        if let finalState = hypothesis.decState {
            decoderState = finalState
        }
        decoderState.lastToken = hypothesis.lastToken

        return hypothesis
    }

    // MARK: - Private Helper Methods

    private func runDecoder(
        token: Int,
        state: RnntDecoderState,
        model: MLModel,
        targetArray: MLMultiArray,
        targetLengthArray: MLMultiArray
    ) throws -> (output: MLFeatureProvider, newState: RnntDecoderState) {
        targetArray[0] = NSNumber(value: token)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targetArray),
            "target_length": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: state.hiddenState),
            "c_in": MLFeatureValue(multiArray: state.cellState),
        ])

        // DON'T reuse input state as output backing - this corrupts state when we have multiple hypotheses
        // The output arrays will be freshly allocated by CoreML
        predictionOptions.outputBackings = [:]

        let output = try model.prediction(from: input, options: predictionOptions)

        // Create a new state by deep-copying from output
        var newState = try RnntDecoderState(from: state)
        newState.update(from: output)

        return (output, newState)
    }

    private func runJoint(
        model: MLModel,
        inputProvider: MLFeatureProvider,
        tokenIdBacking: MLMultiArray,
        tokenProbBacking: MLMultiArray,
        debugFrame: Int? = nil
    ) throws -> JointDecision {
        // Run the joint model
        let output = try model.prediction(from: inputProvider)

        // Try to get token_id directly (joint_decision_single_step model)
        if let tokenIdValue = output.featureValue(for: "token_id"),
            let tokenIdArray = tokenIdValue.multiArrayValue
        {
            let tokenIdPtr = tokenIdArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)
            let token = Int(tokenIdPtr[0])

            var prob: Float = 0.99
            if let tokenProbValue = output.featureValue(for: "token_prob"),
                let tokenProbArray = tokenProbValue.multiArrayValue
            {
                let probPtr = tokenProbArray.dataPointer.bindMemory(to: Float.self, capacity: 1)
                prob = probPtr[0]
            }

            if let frame = debugFrame {
                print("[DEBUG] Frame \(frame): token=\(token), prob=\(prob)")
            }

            return JointDecision(
                token: token,
                probability: clampProbability(prob)
            )
        }

        // Fall back to logits output (raw joint model)
        let logitsArray = try extractFeatureValue(
            from: output,
            key: "logits",
            errorMessage: "Joint output missing both token_id and logits"
        )

        // Compute argmax in Swift
        let vocabSize = config.vocabSize + 1  // +1 for blank
        let logitsPtr = logitsArray.dataPointer.bindMemory(to: Float.self, capacity: vocabSize)

        var maxToken = 0
        var maxLogit: Float = logitsPtr[0]

        for i in 1..<vocabSize {
            if logitsPtr[i] > maxLogit {
                maxLogit = logitsPtr[i]
                maxToken = i
            }
        }

        if let frame = debugFrame {
            var tokenLogits: [(Int, Float)] = []
            for i in 0..<vocabSize {
                tokenLogits.append((i, logitsPtr[i]))
            }
            tokenLogits.sort { $0.1 > $1.1 }
            let top5 = Array(tokenLogits.prefix(5))
            print("[DEBUG] Frame \(frame) joint top-5: \(top5)")
        }

        // Compute softmax probability
        var expSum: Float = 0
        for i in 0..<vocabSize {
            expSum += exp(logitsPtr[i] - maxLogit)
        }
        let prob = 1.0 / expSum

        return JointDecision(
            token: maxToken,
            probability: clampProbability(prob)
        )
    }

    private func copyEncoderFrame(
        from encoder: MLMultiArray,
        frameIndex: Int,
        hiddenSize: Int,
        into dest: MLMultiArray
    ) throws {
        // Encoder shape: [1, hiddenSize, timeFrames]
        let strides = encoder.strides.map { $0.intValue }
        let srcPtr = encoder.dataPointer.bindMemory(to: Float.self, capacity: encoder.count)
        let destPtr = dest.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)

        // Copy frame at frameIndex
        let timeStride = strides[2]
        let hiddenStride = strides[1]
        let baseOffset = frameIndex * timeStride

        for h in 0..<hiddenSize {
            destPtr[h] = srcPtr[baseOffset + h * hiddenStride]
        }
    }

    
    private func extractTimeStep(
        _ projection: MLMultiArray,
        timeIndex: Int
    ) throws -> MLMultiArray {
        let shape = projection.shape.map { $0.intValue }
        let hiddenSize = config.decoderHiddenSize
        
        // Expected shape: [1, 640, 2] where 2 is time dimension
        guard shape.count == 3, shape[1] == hiddenSize else {
            throw ASRError.processingFailed("Unexpected decoder shape: \(shape)")
        }
        
        guard timeIndex < shape[2] else {
            throw ASRError.processingFailed("Time index \(timeIndex) out of bounds for shape \(shape)")
        }
        
        // Create output array for single time step: [1, 640, 1]
        let result = try MLMultiArray(
            shape: [1, NSNumber(value: hiddenSize), 1],
            dataType: .float32
        )
        
        let srcPtr = projection.dataPointer.bindMemory(to: Float.self, capacity: projection.count)
        let destPtr = result.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)
        let strides = projection.strides.map { $0.intValue }
        
        // Extract the specified time index
        // Shape [1, 640, 2], strides typically [1280, 2, 1]
        // For time=0: offset = 0
        // For time=1: offset = 1
        let timeStride = strides[2]
        let offset = timeIndex * timeStride
        
        // Copy the hidden dimension
        let hiddenStride = strides[1]
        for i in 0..<hiddenSize {
            destPtr[i] = srcPtr[i * hiddenStride + offset]
        }
        
        return result
    }
    
    private func copyDecoderProjection(
        _ projection: MLMultiArray,
        into dest: MLMultiArray
    ) throws {
        let shape = projection.shape.map { $0.intValue }
        let hiddenSize = config.decoderHiddenSize

        // Find hidden dimension
        let hiddenAxis: Int
        if shape.count >= 2 && shape[1] == hiddenSize {
            hiddenAxis = 1
        } else if shape.count >= 3 && shape[2] == hiddenSize {
            hiddenAxis = 2
        } else {
            throw ASRError.processingFailed("Decoder projection shape mismatch: \(shape)")
        }

        let srcPtr = projection.dataPointer.bindMemory(to: Float.self, capacity: projection.count)
        let destPtr = dest.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)
        let strides = projection.strides.map { $0.intValue }

        if hiddenAxis == 1 && strides[1] == 1 {
            // Contiguous copy
            memcpy(destPtr, srcPtr, hiddenSize * MemoryLayout<Float>.stride)
        } else {
            // Strided copy
            let stride = strides[hiddenAxis]
            
            // Check if there is a time/sequence dimension (U) that is > 1
            // Shape is [1, 640, 2] -> strides [1280, 2, 1]
            // We want to take the LAST frame (index 1) if U=2
            var offset = 0
            if shape.count >= 3 && shape[2] > 1 {
                // Assuming shape [B, H, U] where U is the last dim
                // If U=2, we want index 1.
                // Stride for U is strides[2]
                let uStride = strides[2]
                offset = (shape[2] - 1) * uStride
            }
            
            for i in 0..<hiddenSize {
                destPtr[i] = srcPtr[i * stride + offset]
            }
        }
    }

    private func extractFeatureValue(
        from provider: MLFeatureProvider,
        key: String,
        errorMessage: String
    ) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }

    private func clampProbability(_ value: Float) -> Float {
        guard value.isFinite else { return 0 }
        return min(max(value, 0), 1)
    }
}

/// Hypothesis result from RNNT decoding
struct RnntHypothesis: Sendable {
    var score: Float = 0.0
    var ySequence: [Int] = []
    var decState: RnntDecoderState?
    var timestamps: [Int] = []
    var tokenConfidences: [Float] = []
    var lastToken: Int?
    var eouDetected: Bool = false

    init(
        score: Float = 0.0,
        ySequence: [Int] = [],
        decState: RnntDecoderState? = nil,
        timestamps: [Int] = [],
        tokenConfidences: [Float] = [],
        eouDetected: Bool = false,
        lastToken: Int? = nil
    ) {
        self.score = score
        self.ySequence = ySequence
        self.decState = decState
        self.timestamps = timestamps
        self.tokenConfidences = tokenConfidences
        self.eouDetected = eouDetected
        self.lastToken = lastToken
    }

    var isEmpty: Bool { ySequence.isEmpty }
    var tokenCount: Int { ySequence.count }
    var hasTokens: Bool { !ySequence.isEmpty }

    var destructured: (tokens: [Int], timestamps: [Int], confidences: [Float]) {
        (ySequence, timestamps, tokenConfidences)
    }
}
