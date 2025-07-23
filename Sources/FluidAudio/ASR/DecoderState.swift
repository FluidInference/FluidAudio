//
//  DecoderState.swift
//  FluidAudio
//
//  Copyright © 2025 Brandon Weng. All rights reserved.
//

import CoreML
import Foundation

/// Manages LSTM hidden and cell states for the Parakeet decoder
struct DecoderState {
    var hiddenState: MLMultiArray
    var cellState: MLMultiArray

    init() {
        hiddenState = try! MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        cellState = try! MLMultiArray(shape: [2, 1, 640], dataType: .float32)
        
        hiddenState.resetData(to: 0)
        cellState.resetData(to: 0)
    }

    mutating func update(from decoderOutput: MLFeatureProvider) {
        hiddenState = decoderOutput.featureValue(for: "h_out")?.multiArrayValue ?? hiddenState
        cellState = decoderOutput.featureValue(for: "c_out")?.multiArrayValue ?? cellState
    }

    init(from other: DecoderState) {
        hiddenState = try! MLMultiArray(shape: other.hiddenState.shape, dataType: .float32)
        cellState = try! MLMultiArray(shape: other.cellState.shape, dataType: .float32)
        
        hiddenState.copyData(from: other.hiddenState)
        cellState.copyData(from: other.cellState)
    }
}

private extension MLMultiArray {
    func resetData(to value: NSNumber) {
        for i in 0..<count {
            self[i] = value
        }
    }
    
    func copyData(from source: MLMultiArray) {
        for i in 0..<count {
            self[i] = source[i]
        }
    }
}