import CoreML
import Foundation

/// Greedy RNNT decode loop for Parakeet Unified 0.6B.
///
/// Shared by the streaming manager (persistent state across chunks) and the
/// offline batch manager (fresh state per window). Wraps the CoreML decoder
/// (RNNT prediction network) and single-step joint-decision models; argmax,
/// probability, and blank handling come from the joint-decision model itself.
final class UnifiedRnntDecoder {
    struct Emission {
        /// Emitted (non-blank) token id.
        let token: Int
        /// Encoder frame the token was emitted at, offset by the caller's
        /// global frame base.
        let frame: Int
        /// Softmax probability of the emitted token.
        let prob: Float
    }

    private let decoderModel: MLModel
    private let jointDecisionModel: MLModel
    private let config: UnifiedConfig

    // RNNT prediction-network state: (h, c) are the LSTM state BEFORE
    // `lastToken` is consumed; the first decoder call of a decode pass
    // re-derives the cached decoder output by applying `lastToken` to it.
    private var hState: MLMultiArray
    private var cState: MLMultiArray
    private var lastToken: Int32

    init(decoderModel: MLModel, jointDecisionModel: MLModel, config: UnifiedConfig) throws {
        self.decoderModel = decoderModel
        self.jointDecisionModel = jointDecisionModel
        self.config = config
        self.hState = try Self.zeroState(config: config)
        self.cState = try Self.zeroState(config: config)
        self.lastToken = Int32(config.blankIdx)
    }

    private static func zeroState(config: UnifiedConfig) throws -> MLMultiArray {
        let state = try MLMultiArray(
            shape: [NSNumber(value: config.decoderLayers), 1, NSNumber(value: config.decoderHidden)],
            dataType: .float32
        )
        state.reset(to: 0)
        return state
    }

    func reset() throws {
        hState = try Self.zeroState(config: config)
        cState = try Self.zeroState(config: config)
        lastToken = Int32(config.blankIdx)
    }

    /// Greedy RNNT over encoder frames `frameRange` of `encoded` [1, D, T].
    /// Emitted frames are reported as `globalFrameOffset + t`.
    func decode(
        encoded: MLMultiArray,
        frameRange: Range<Int>,
        globalFrameOffset: Int = 0
    ) throws -> [Emission] {
        var currentToken = lastToken
        var currentH = hState
        var currentC = cState
        var emissions: [Emission] = []

        var decoderStep = try runDecoder(token: currentToken, h: currentH, c: currentC)

        for t in frameRange {
            let encStep = try extractEncoderStep(from: encoded, timeIndex: t)

            for _ in 0..<config.maxSymbolsPerFrame {
                let jointOutput = try jointDecisionModel.prediction(
                    from: MLDictionaryFeatureProvider(dictionary: [
                        "encoder_step": MLFeatureValue(multiArray: encStep),
                        "decoder_step": MLFeatureValue(multiArray: decoderStep.output),
                    ])
                )
                guard let tokenArray = jointOutput.featureValue(for: "token_id")?.multiArrayValue else {
                    throw ASRError.processingFailed("Unified joint decision missing token_id")
                }
                let tokenId = tokenArray[0].int32Value

                if tokenId == Int32(config.blankIdx) {
                    break
                }
                let prob = jointOutput.featureValue(for: "token_prob")?.multiArrayValue?[0].floatValue ?? 0
                emissions.append(
                    Emission(token: Int(tokenId), frame: globalFrameOffset + t, prob: prob)
                )
                currentToken = tokenId
                currentH = decoderStep.h
                currentC = decoderStep.c
                decoderStep = try runDecoder(token: currentToken, h: currentH, c: currentC)
            }
        }

        // Persist state atomically after the pass completes.
        lastToken = currentToken
        hState = currentH
        cState = currentC
        return emissions
    }

    private struct DecoderStep {
        let output: MLMultiArray
        let h: MLMultiArray
        let c: MLMultiArray
    }

    private func runDecoder(token: Int32, h: MLMultiArray, c: MLMultiArray) throws -> DecoderStep {
        let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        targets[0] = NSNumber(value: token)
        let targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = 1

        let output = try decoderModel.prediction(
            from: MLDictionaryFeatureProvider(dictionary: [
                "targets": MLFeatureValue(multiArray: targets),
                "target_length": MLFeatureValue(multiArray: targetLength),
                "h_in": MLFeatureValue(multiArray: h),
                "c_in": MLFeatureValue(multiArray: c),
            ])
        )
        guard let decoderOut = output.featureValue(for: "decoder")?.multiArrayValue,
            let hOut = output.featureValue(for: "h_out")?.multiArrayValue,
            let cOut = output.featureValue(for: "c_out")?.multiArrayValue
        else {
            throw ASRError.processingFailed("Unified decoder failed")
        }
        // Decoder output is [1, 640, 1] (U=1 export); h/c become the state for
        // the NEXT decoder call only after this token is accepted.
        return DecoderStep(output: decoderOut, h: hOut, c: cOut)
    }

    private func extractEncoderStep(from encoded: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        let dim = encoded.shape[1].intValue
        let step = try MLMultiArray(shape: [1, NSNumber(value: dim), 1], dataType: .float32)

        let srcPtr = encoded.dataPointer.bindMemory(to: Float.self, capacity: encoded.count)
        let dstPtr = step.dataPointer.bindMemory(to: Float.self, capacity: step.count)
        let stride1 = encoded.strides[1].intValue
        let stride2 = encoded.strides[2].intValue

        for c in 0..<dim {
            dstPtr[c] = srcPtr[c * stride1 + timeIndex * stride2]
        }
        return step
    }
}
