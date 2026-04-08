import Accelerate
@preconcurrency import CoreML
import Foundation

/// Model inference operations for TDT decoding.
///
/// Encapsulates execution of decoder LSTM, joint network, and decoder projection normalization.
/// These operations are separated from the main decoding loop to improve testability and clarity.
internal struct TdtModelInference: Sendable {
    private let predictionOptions: MLPredictionOptions

    init() {
        self.predictionOptions = AsrModels.optimizedPredictionOptions()
    }

    /// Execute decoder LSTM with state caching.
    ///
    /// - Parameters:
    ///   - token: Token ID to decode
    ///   - state: Current decoder LSTM state
    ///   - model: Decoder MLModel
    ///   - targetArray: Pre-allocated array for token input
    ///   - targetLengthArray: Pre-allocated array for length (always 1)
    ///
    /// - Returns: Tuple of (output features, updated state)
    func runDecoder(
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
            "target_length": MLFeatureValue(multiArray: targetLengthArray),
            "h_in": MLFeatureValue(multiArray: state.hiddenState),
            "c_in": MLFeatureValue(multiArray: state.cellState),
        ])

        // Reuse decoder state output buffers to avoid CoreML allocating new ones
        // Note: outputBackings expects raw backing objects (MLMultiArray / CVPixelBuffer)
        predictionOptions.outputBackings = [
            "h_out": state.hiddenState,
            "c_out": state.cellState,
        ]

        let output = try model.prediction(
            from: input,
            options: predictionOptions
        )

        var newState = state
        newState.update(from: output)

        return (output, newState)
    }

    /// Execute joint network with zero-copy and ANE optimization.
    ///
    /// - Parameters:
    ///   - encoderFrames: View into encoder output tensor
    ///   - timeIndex: Frame index to process
    ///   - preparedDecoderStep: Normalized decoder projection
    ///   - model: Joint MLModel
    ///   - encoderStep: Pre-allocated encoder step array
    ///   - encoderDestPtr: Pointer for encoder frame copy
    ///   - encoderDestStride: Stride for encoder copy
    ///   - inputProvider: Reusable feature provider
    ///   - tokenIdBacking: Pre-allocated output for token ID
    ///   - tokenProbBacking: Pre-allocated output for probability
    ///   - durationBacking: Pre-allocated output for duration
    ///
    /// - Returns: Joint decision (token, probability, duration bin)
    func runJointPrepared(
        encoderFrames: EncoderFrameView,
        timeIndex: Int,
        preparedDecoderStep: MLMultiArray,
        model: MLModel,
        encoderStep: MLMultiArray,
        encoderDestPtr: UnsafeMutablePointer<Float>,
        encoderDestStride: Int,
        inputProvider: MLFeatureProvider,
        tokenIdBacking: MLMultiArray,
        tokenProbBacking: MLMultiArray,
        durationBacking: MLMultiArray
    ) throws -> TdtJointDecision {

        // Fill encoder step with the requested frame
        try encoderFrames.copyFrame(at: timeIndex, into: encoderDestPtr, destinationStride: encoderDestStride)

        // Prefetch arrays for ANE
        encoderStep.prefetchToNeuralEngine()
        preparedDecoderStep.prefetchToNeuralEngine()

        // Reuse tiny output tensors for joint prediction (provide raw MLMultiArray backings)
        predictionOptions.outputBackings = [
            "token_id": tokenIdBacking,
            "token_prob": tokenProbBacking,
            "duration": durationBacking,
        ]

        // Execute joint network using the reusable provider
        let output = try model.prediction(
            from: inputProvider,
            options: predictionOptions
        )

        let tokenIdArray = try extractFeatureValue(
            from: output, key: "token_id", errorMessage: "Joint decision output missing token_id")
        let tokenProbArray = try extractFeatureValue(
            from: output, key: "token_prob", errorMessage: "Joint decision output missing token_prob")
        let durationArray = try extractFeatureValue(
            from: output, key: "duration", errorMessage: "Joint decision output missing duration")

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

        return TdtJointDecision(token: token, probability: probability, durationBin: durationBin)
    }

    /// Normalize decoder projection into [1, hiddenSize, 1] layout via BLAS copy.
    ///
    /// CoreML decoder outputs can have varying layouts ([1, 1, 640] or [1, 640, 1]).
    /// This function normalizes to the joint network's expected input format using
    /// efficient BLAS operations to handle arbitrary strides.
    ///
    /// - Parameters:
    ///   - projection: Decoder output projection (any 3D layout with hiddenSize dimension)
    ///   - destination: Optional pre-allocated destination array (for hot path)
    ///
    /// - Returns: Normalized array in [1, hiddenSize, 1] format
    @discardableResult
    func normalizeDecoderProjection(
        _ projection: MLMultiArray,
        into destination: MLMultiArray? = nil
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

        let out: MLMultiArray
        if let destination {
            let outShape = destination.shape.map { $0.intValue }
            guard destination.dataType == .float32, outShape.count == 3, outShape[0] == 1,
                outShape[2] == 1, outShape[1] == hiddenSize
            else {
                throw ASRError.processingFailed(
                    "Prepared decoder step shape mismatch: \(destination.shapeString)")
            }
            out = destination
        } else {
            out = try ANEMemoryUtils.createAlignedArray(
                shape: [1, NSNumber(value: hiddenSize), 1],
                dataType: .float32
            )
        }

        let strides = projection.strides.map { $0.intValue }
        let hiddenStride = strides[hiddenAxis]

        let dataPointer = projection.dataPointer.bindMemory(to: Float.self, capacity: projection.count)
        let startPtr = dataPointer.advanced(by: 0)

        let destPtr = out.dataPointer.bindMemory(to: Float.self, capacity: hiddenSize)
        let destStrides = out.strides.map { $0.intValue }
        let destHiddenStride = destStrides[1]
        let destStrideCblas = try makeBlasIndex(destHiddenStride, label: "Decoder destination stride")

        let count = try makeBlasIndex(hiddenSize, label: "Decoder projection length")
        let stride = try makeBlasIndex(hiddenStride, label: "Decoder projection stride")
        cblas_scopy(count, startPtr, stride, destPtr, destStrideCblas)

        return out
    }

    /// Extract MLMultiArray feature value with error handling.
    private func extractFeatureValue(
        from output: MLFeatureProvider, key: String, errorMessage: String
    ) throws
        -> MLMultiArray
    {
        guard let value = output.featureValue(for: key)?.multiArrayValue else {
            throw ASRError.processingFailed(errorMessage)
        }
        return value
    }
}
