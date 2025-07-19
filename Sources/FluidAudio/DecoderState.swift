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
        // Initialize with zeros for LSTM hidden/cell states
        // Shape: [num_layers, batch_size, hidden_size] = [2, 1, 640]
        hiddenState = try! MLMultiArray(shape: [2, 1, 640] as [NSNumber], dataType: .float32)
        cellState = try! MLMultiArray(shape: [2, 1, 640] as [NSNumber], dataType: .float32)

        // Initialize with zeros
        for i in 0..<hiddenState.count {
            hiddenState[i] = NSNumber(value: 0.0)
        }
        for i in 0..<cellState.count {
            cellState[i] = NSNumber(value: 0.0)
        }
    }

    mutating func update(from decoderOutput: MLFeatureProvider) {
        if let newHiddenState = decoderOutput.featureValue(for: "h_out")?.multiArrayValue {
            hiddenState = newHiddenState
        }
        if let newCellState = decoderOutput.featureValue(for: "c_out")?.multiArrayValue {
            cellState = newCellState
        }
    }

    /// Create a zero-initialized decoder state
    static func zero() -> DecoderState {
        return DecoderState()
    }

    /// Copy constructor for TDT hypothesis state management
    init(from other: DecoderState) {
        self.hiddenState = try! MLMultiArray(shape: other.hiddenState.shape, dataType: .float32)
        self.cellState = try! MLMultiArray(shape: other.cellState.shape, dataType: .float32)

        // Copy values
        for i in 0..<other.hiddenState.count {
            self.hiddenState[i] = other.hiddenState[i]
        }
        for i in 0..<other.cellState.count {
            self.cellState[i] = other.cellState[i]
        }
    }
}