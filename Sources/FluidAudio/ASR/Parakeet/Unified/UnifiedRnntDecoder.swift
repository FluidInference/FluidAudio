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

    // Per-step Core ML calls spent about two thirds of their time allocating
    // and binding fresh inputs and outputs rather than computing. Inputs are
    // filled in place and outputs land in buffers made once (`outputBackings`).
    // Core ML only writes straight into a backing that was never bound as an
    // input, so the decoder always writes to one output set and the accepted
    // state is copied into its own input arrays (`workH`, `workC`): two 5 KB
    // copies per emitted token. Same values, same order.
    private let targets: MLMultiArray
    private let targetLength: MLMultiArray
    private let workH: MLMultiArray
    private let workC: MLMultiArray
    private var encoderStep: MLMultiArray?
    private let decoderOptions = MLPredictionOptions()
    private let jointOptions = MLPredictionOptions()

    init(decoderModel: MLModel, jointDecisionModel: MLModel, config: UnifiedConfig) throws {
        self.decoderModel = decoderModel
        self.jointDecisionModel = jointDecisionModel
        self.config = config
        self.hState = try Self.zeroState(config: config)
        self.cState = try Self.zeroState(config: config)
        self.lastToken = Int32(config.blankIdx)

        self.targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        self.targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = 1
        self.workH = try Self.zeroState(config: config)
        self.workC = try Self.zeroState(config: config)
        decoderOptions.outputBackings = try Self.outputBuffers(for: decoderModel)
        jointOptions.outputBackings = try Self.outputBuffers(for: jointDecisionModel)
    }

    /// One buffer per multi-array output, shaped and typed as the model declares it.
    private static func outputBuffers(for model: MLModel) throws -> [String: MLMultiArray] {
        var buffers: [String: MLMultiArray] = [:]
        for (name, description) in model.modelDescription.outputDescriptionsByName {
            guard let constraint = description.multiArrayConstraint, !constraint.shape.isEmpty else { continue }
            buffers[name] = try MLMultiArray(shape: constraint.shape, dataType: constraint.dataType)
        }
        return buffers
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
        // Work on copies so the stored state changes only when the pass completes.
        try Self.copy(hState, into: workH)
        try Self.copy(cState, into: workC)
        var emissions: [Emission] = []

        var decoderStep = try runDecoder(token: currentToken)

        for t in frameRange {
            let encStep = try extractEncoderStep(from: encoded, timeIndex: t)

            for _ in 0..<config.maxSymbolsPerFrame {
                let jointOutput = try jointDecisionModel.prediction(
                    from: UnifiedJointDecisionFeatureProvider(
                        encoderStep: encStep,
                        decoderStep: decoderStep.output
                    ),
                    options: jointOptions
                )
                guard let tokenArray = jointOutput.featureValue(for: "token_id")?.multiArrayValue else {
                    throw ASRError.processingFailed("Unified joint decision missing token_id")
                }
                let tokenId = Self.firstInt32(of: tokenArray)

                if tokenId == Int32(config.blankIdx) {
                    break
                }
                let prob = jointOutput.featureValue(for: "token_prob")?.multiArrayValue.map(Self.firstFloat) ?? 0
                emissions.append(
                    Emission(token: Int(tokenId), frame: globalFrameOffset + t, prob: prob)
                )
                currentToken = tokenId
                try Self.copy(decoderStep.h, into: workH)
                try Self.copy(decoderStep.c, into: workC)
                decoderStep = try runDecoder(token: currentToken)
            }
        }

        // Persist state atomically after the pass completes.
        lastToken = currentToken
        try Self.copy(workH, into: hState)
        try Self.copy(workC, into: cState)
        return emissions
    }

    private struct DecoderStep {
        let output: MLMultiArray
        let h: MLMultiArray
        let c: MLMultiArray
    }

    /// Runs the prediction network on `token` from the state in `workH`/`workC`.
    private func runDecoder(token: Int32) throws -> DecoderStep {
        targets[0] = NSNumber(value: token)

        let output = try decoderModel.prediction(
            from: UnifiedDecoderFeatureProvider(
                targets: targets,
                targetLength: targetLength,
                hIn: workH,
                cIn: workC
            ),
            options: decoderOptions
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

    /// Copies a float32 state tensor element for element, through strides
    /// unless both sides are plainly contiguous.
    private static func copy(_ source: MLMultiArray, into destination: MLMultiArray) throws {
        guard source !== destination else { return }
        guard source.dataType == .float32, destination.dataType == .float32,
            source.shape == destination.shape
        else {
            throw ASRError.processingFailed("Unified decoder state has an unexpected layout")
        }
        let count = source.count
        let to = destination.dataPointer.bindMemory(to: Float.self, capacity: count)
        if source.isContiguous && destination.isContiguous {
            to.update(from: source.dataPointer.bindMemory(to: Float.self, capacity: count), count: count)
            return
        }
        let shape = source.shape.map(\.intValue)
        let sourceStrides = source.strides.map(\.intValue)
        let destinationStrides = destination.strides.map(\.intValue)
        let from = source.dataPointer.bindMemory(to: Float.self, capacity: 1)
        var index = [Int](repeating: 0, count: shape.count)
        for _ in 0..<count {
            var sourceOffset = 0
            var destinationOffset = 0
            for axis in 0..<shape.count {
                sourceOffset += index[axis] * sourceStrides[axis]
                destinationOffset += index[axis] * destinationStrides[axis]
            }
            to[destinationOffset] = from[sourceOffset]
            var axis = shape.count - 1
            while axis >= 0 {
                index[axis] += 1
                if index[axis] < shape[axis] { break }
                index[axis] = 0
                axis -= 1
            }
        }
    }

    private static func firstInt32(of array: MLMultiArray) -> Int32 {
        array.dataType == .int32 ? array.dataPointer.load(as: Int32.self) : array[0].int32Value
    }

    private static func firstFloat(of array: MLMultiArray) -> Float {
        array.dataType == .float32 ? array.dataPointer.load(as: Float.self) : array[0].floatValue
    }

    private func extractEncoderStep(from encoded: MLMultiArray, timeIndex: Int) throws -> MLMultiArray {
        let dim = encoded.shape[1].intValue
        let step: MLMultiArray
        if let reused = encoderStep, reused.count == dim {
            step = reused
        } else {
            step = try MLMultiArray(shape: [1, NSNumber(value: dim), 1], dataType: .float32)
            encoderStep = step
        }

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

extension MLMultiArray {
    /// Row-major with no padding, so element i sits at storage offset i.
    fileprivate var isContiguous: Bool {
        var expected = 1
        for axis in (0..<shape.count).reversed() {
            if shape[axis].intValue > 1 && strides[axis].intValue != expected { return false }
            expected *= shape[axis].intValue
        }
        return true
    }
}
