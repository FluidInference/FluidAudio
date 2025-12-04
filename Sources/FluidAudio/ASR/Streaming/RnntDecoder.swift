import CoreML
import Foundation

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
    private let eouId: Int32 = 1024 // Verified EOU ID
    private let maxSymbolsPerStep = 10
    private let hiddenSize = 640
    private let layers = 1
    
    public init(decoderModel: MLModel, jointModel: MLModel) {
        self.decoderModel = decoderModel
        self.jointModel = jointModel
        
        // Initialize state
        self.hState = try! MLMultiArray(shape: [NSNumber(value: layers), NSNumber(value: 1), NSNumber(value: hiddenSize)], dataType: .float32)
        self.cState = try! MLMultiArray(shape: [NSNumber(value: layers), NSNumber(value: 1), NSNumber(value: hiddenSize)], dataType: .float32)
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
    /// - Returns: List of predicted token IDs
    public func decode(encoderOutput: MLMultiArray) throws -> [Int] {
        var predictedIds: [Int] = []
        
        let T = encoderOutput.shape[2].intValue
        let hiddenDim = encoderOutput.shape[1].intValue
        
        // Helper to extract a single time step from encoder output
        // encoderOutput is [1, 512, T]
        // We need [1, 512, 1] for joint model
        
        for t in 0..<T {
            // Extract encoder step
            let encoderStep = try extractEncoderStep(from: encoderOutput, timeIndex: t, hiddenDim: hiddenDim)
            
            var symbolsAdded = 0
            
            while symbolsAdded < maxSymbolsPerStep {
                // 1. Run Decoder
                let decoderInput = try prepareDecoderInput(lastToken: lastToken, h: hState, c: cState)
                let decoderOutput = try decoderModel.prediction(from: decoderInput)
                
                var decoderStep = decoderOutput.featureValue(for: "decoder")!.multiArrayValue!
                // Ensure shape [1, 640, 1]
                if decoderStep.shape.count == 3 && decoderStep.shape[2].intValue > 1 {
                     // Slice to keep only the first frame [1, 640, 1]
                     decoderStep = try sliceDecoderStep(decoderStep)
                }
                
                // 2. Run Joint
                let jointInput = try MLDictionaryFeatureProvider(dictionary: [
                    "encoder_step": MLFeatureValue(multiArray: encoderStep),
                    "decoder_step": MLFeatureValue(multiArray: decoderStep)
                ])
                
                let jointOutput = try jointModel.prediction(from: jointInput)
                
                // 3. Get Token ID
                // Output "token_id" is [1, 1, 1] (argmax)
                let tokenIdMultiArray = jointOutput.featureValue(for: "token_id")!.multiArrayValue!
                let tokenId = tokenIdMultiArray[0].int32Value
                
                if tokenId == blankId {
                    break
                } else {
                    // Check EOU
                    if tokenId == eouId {
                        // print("EOU Detected at t=\(t)")
                        resetState()
                        break 
                    }
                    
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
        
        return predictedIds
    }
    
    private func extractEncoderStep(from encoderOutput: MLMultiArray, timeIndex: Int, hiddenDim: Int) throws -> MLMultiArray {
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
            "c_in": MLFeatureValue(multiArray: c)
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
