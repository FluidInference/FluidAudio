import CoreML
import Foundation

/// Manages LSTM hidden and cell states for the Parakeet decoder
struct DecoderState {
    var hiddenState: MLMultiArray
    var cellState: MLMultiArray
    var lastToken: Int?

    enum InitError: Error {
        case aneAllocationFailed(String)
    }

    init() throws {
        // Use ANE-aligned arrays for optimal performance
        do {
            hiddenState = try ANEOptimizer.createANEAlignedArray(
                shape: [2, 1, 640], 
                dataType: .float32
            )
            cellState = try ANEOptimizer.createANEAlignedArray(
                shape: [2, 1, 640], 
                dataType: .float32
            )
        } catch {
            // Fall back to standard MLMultiArray if ANE allocation fails
            print("Warning: ANE-aligned allocation failed, falling back to standard MLMultiArray: \(error)")
            hiddenState = try MLMultiArray(shape: [2, 1, 640], dataType: .float32)
            cellState = try MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        }
        
        // Initialize to zeros using Accelerate
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
    }

    mutating func update(from decoderOutput: MLFeatureProvider) {
        hiddenState = decoderOutput.featureValue(for: "h_out")?.multiArrayValue ?? hiddenState
        cellState = decoderOutput.featureValue(for: "c_out")?.multiArrayValue ?? cellState
    }

    init(from other: DecoderState) throws {
        hiddenState = try MLMultiArray(shape: other.hiddenState.shape, dataType: .float32)
        cellState = try MLMultiArray(shape: other.cellState.shape, dataType: .float32)
        lastToken = other.lastToken

        hiddenState.copyData(from: other.hiddenState)
        cellState.copyData(from: other.cellState)
    }
    
    /// Fallback initializer that never fails (for use in critical paths)
    init(fallback: Bool) {
        // Standard MLMultiArray allocation without ANE optimization
        hiddenState = try! MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        cellState = try! MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        
        // Initialize to zeros
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
    }
}

import Accelerate

extension MLMultiArray {
    func resetData(to value: NSNumber) {
        guard dataType == .float32 else {
            // Fallback for non-float types
            for i in 0..<count {
                self[i] = value
            }
            return
        }
        
        // Use vDSP for optimized memory fill
        var floatValue = value.floatValue
        self.dataPointer.withMemoryRebound(to: Float.self, capacity: count) { ptr in
            vDSP_vfill(&floatValue, ptr, 1, vDSP_Length(count))
        }
    }

    func copyData(from source: MLMultiArray) {
        guard dataType == .float32 && source.dataType == .float32 else {
            // Fallback for non-float types
            for i in 0..<count {
                self[i] = source[i]
            }
            return
        }
        
        // Use optimized memory copy
        let destPtr = self.dataPointer.bindMemory(to: Float.self, capacity: count)
        let srcPtr = source.dataPointer.bindMemory(to: Float.self, capacity: count)
        memcpy(destPtr, srcPtr, count * MemoryLayout<Float>.stride)
    }
}
