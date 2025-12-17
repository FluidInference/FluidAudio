import CoreML
import Foundation

/// Result of decoding a chunk, including EOU detection status.
public struct DecodeResult: Sendable {
    /// Predicted token IDs for this chunk
    public let tokenIds: [Int]
    /// Whether End-of-Utterance was detected
    public let eouDetected: Bool
}

/// Implements the RNN-T greedy decoding loop for the Parakeet EOU model.
/// Matches the logic in `test_pure_coreml.py`.
public final class RnntDecoder {
    private let decoderModel: MLModel
    private let jointModel: MLModel

    // Decoder State
    private var hState: MLMultiArray
    private var cState: MLMultiArray
    private var lastToken: Int32

    // Constants
    private let blankId: Int32 = 1026
    private let eouId: Int32 = 1024  // Verified EOU ID
    private let maxSymbolsPerStep = 2
    private let hiddenSize = 640
    private let layers = 1

    public init(decoderModel: MLModel, jointModel: MLModel) {
        self.decoderModel = decoderModel
        self.jointModel = jointModel

        // Initialize state
        self.hState = try! MLMultiArray(
            shape: [NSNumber(value: layers), NSNumber(value: 1), NSNumber(value: hiddenSize)], dataType: .float32)
        self.cState = try! MLMultiArray(
            shape: [NSNumber(value: layers), NSNumber(value: 1), NSNumber(value: hiddenSize)], dataType: .float32)
        self.lastToken = blankId

        resetState()
    }

    public func resetState() {
        // Zero out states
        let count = hState.count
        hState.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            ptr.baseAddress?.assign(repeating: 0, count: count)
        }
        cState.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            ptr.baseAddress?.assign(repeating: 0, count: count)
        }
        lastToken = blankId
    }

    /// Decodes the encoder output using greedy search.
    /// - Parameter encoderOutput: [1, 512, T]
    /// - Parameter timeOffset: Global time offset for debugging
    /// - Parameter skipFrames: Number of frames to skip at start (for overlap handling)
    /// - Parameter validOutLen: Number of valid output frames to decode. If nil, decode all frames.
    ///                          For streaming, this should be set to streaming_cfg.valid_out_len.
    /// - Returns: DecodeResult containing token IDs and EOU detection status
    public func decodeWithEOU(
        encoderOutput: MLMultiArray, timeOffset: Int = 0, skipFrames: Int = 0, validOutLen: Int? = nil
    ) throws -> DecodeResult {
        var predictedIds: [Int] = []
        var eouDetected = false

        let T = encoderOutput.shape[2].intValue
        let hiddenDim = encoderOutput.shape[1].intValue

        // Determine how many frames to decode
        // NeMo truncates encoder output to valid_out_len for streaming
        let maxT = validOutLen.map { min($0, T) } ?? T

        // Skip overlapping frames from previous chunk
        let startT = min(skipFrames, maxT)

        outerLoop: for t in startT..<maxT {
            let globalT = timeOffset + t

            // Extract encoder step
            let encoderStep = try extractEncoderStep(from: encoderOutput, timeIndex: t, hiddenDim: hiddenDim)

            var symbolsAdded = 0

            while symbolsAdded < maxSymbolsPerStep {
                // 1. Run Decoder
                let decoderInput = try prepareDecoderInput(lastToken: lastToken, h: hState, c: cState)
                let decoderOutput = try decoderModel.prediction(from: decoderInput)

                var decoderStep = decoderOutput.featureValue(for: "decoder")!.multiArrayValue!
                // Decoder outputs [1, 640, 2] - NeMo uses the LAST frame
                if decoderStep.shape.count == 3 && decoderStep.shape[2].intValue > 1 {
                    // Slice to keep only the last frame [1, 640, 1]
                    decoderStep = try sliceDecoderStep(decoderStep)
                }

                // 2. Run Joint
                let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                    "encoder_step": MLFeatureValue(multiArray: encoderStep),
                    "decoder_step": MLFeatureValue(multiArray: decoderStep),
                ])

                let jointOutput = try jointModel.prediction(from: jointInput)

                // 3. Get Token ID
                // Output "token_id" is [1, 1, 1] (argmax)
                let tokenIdMultiArray = jointOutput.featureValue(for: "token_id")!.multiArrayValue!
                let tokenId = tokenIdMultiArray[0].int32Value

                if tokenId == blankId {
                    break
                } else if tokenId == eouId {
                    // EOU detected - signal and stop processing
                    eouDetected = true
                    break outerLoop
                } else {
                    predictedIds.append(Int(tokenId))
                    lastToken = tokenId

                    // Update State
                    let newH = decoderOutput.featureValue(for: "h_out")!.multiArrayValue!
                    let newC = decoderOutput.featureValue(for: "c_out")!.multiArrayValue!

                    hState = newH
                    cState = newC

                    symbolsAdded += 1
                }
            }
        }

        return DecodeResult(tokenIds: predictedIds, eouDetected: eouDetected)
    }

    /// Decodes the encoder output using greedy search (legacy method without EOU status).
    /// - Parameter encoderOutput: [1, 512, T]
    /// - Parameter timeOffset: Global time offset for debugging
    /// - Parameter skipFrames: Number of frames to skip at start (for overlap handling)
    /// - Parameter validOutLen: Number of valid output frames to decode. If nil, decode all frames.
    ///                          For streaming, this should be set to streaming_cfg.valid_out_len.
    /// - Returns: List of predicted token IDs
    public func decode(
        encoderOutput: MLMultiArray, timeOffset: Int = 0, skipFrames: Int = 0, validOutLen: Int? = nil
    ) throws -> [Int] {
        let result = try decodeWithEOU(
            encoderOutput: encoderOutput, timeOffset: timeOffset, skipFrames: skipFrames, validOutLen: validOutLen)
        return result.tokenIds
    }

    private func extractEncoderStep(
        from encoderOutput: MLMultiArray, timeIndex: Int, hiddenDim: Int
    ) throws -> MLMultiArray {
        let stepArray = try MLMultiArray(shape: [1, NSNumber(value: hiddenDim), 1], dataType: .float32)

        let srcPtr = encoderOutput.dataPointer.bindMemory(to: Float.self, capacity: encoderOutput.count)
        let dstPtr = stepArray.dataPointer.bindMemory(to: Float.self, capacity: hiddenDim)

        // encoderOutput is [1, D, T] -> strides [D*T, T, 1] or [D*T, 1, D]?
        // CoreML default is C-contiguous: [Batch, Channel, Width] -> [1, 512, T]
        // Stride for dim 0: 512*T
        // Stride for dim 1: T
        // Stride for dim 2: 1
        // Index = b*S0 + c*S1 + t*S2
        // We want encoderOutput[0, :, t]

        // Wait, let's check strides.
        let stride0 = encoderOutput.strides[0].intValue
        let stride1 = encoderOutput.strides[1].intValue
        let stride2 = encoderOutput.strides[2].intValue

        for c in 0..<hiddenDim {
            let srcIdx = 0 * stride0 + c * stride1 + timeIndex * stride2
            dstPtr[c] = srcPtr[srcIdx]
        }

        return stepArray
    }

    private func prepareDecoderInput(lastToken: Int32, h: MLMultiArray, c: MLMultiArray) throws -> MLFeatureProvider {
        let targets = try MLMultiArray(shape: [1, 1], dataType: .int32)
        targets[0] = NSNumber(value: lastToken)

        let targetLength = try MLMultiArray(shape: [1], dataType: .int32)
        targetLength[0] = 1

        return try MLDictionaryFeatureProvider(dictionary: [
            "targets": MLFeatureValue(multiArray: targets),
            "target_length": MLFeatureValue(multiArray: targetLength),
            "h_in": MLFeatureValue(multiArray: h),
            "c_in": MLFeatureValue(multiArray: c),
        ])
    }

    private func sliceDecoderStep(_ input: MLMultiArray) throws -> MLMultiArray {
        // Input: [1, 640, T] -> Output: [1, 640, 1]
        let hiddenDim = input.shape[1].intValue
        let output = try MLMultiArray(shape: [1, NSNumber(value: hiddenDim), 1], dataType: .float32)

        let srcPtr = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let dstPtr = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)

        // Copy last frame (t=T-1)
        // Matches Python: decoder_step[:, :, -1:]
        let T = input.shape[2].intValue
        let lastT = T - 1

        let stride0 = input.strides[0].intValue
        let stride1 = input.strides[1].intValue
        let stride2 = input.strides[2].intValue

        for c in 0..<hiddenDim {
            // Assuming [1, 640, T]
            // We want t=lastT
            let srcIdx = 0 * stride0 + c * stride1 + lastT * stride2
            let dstIdx = c
            dstPtr[dstIdx] = srcPtr[srcIdx]
        }

        return output
    }

}
