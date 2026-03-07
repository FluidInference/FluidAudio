//
//  SortformerFilter.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/14/26.
//

import Foundation
import Accelerate

/// EMA filter for smoothing predictions using FIFO queue as reference
internal class SortformerFilter {
    private let defaultWeights: [Float]
    private var weights: [Float]
    private var tmpBuffer: [Float]
    public let windowSize: Int
    public let numSpeakers: Int

    /// - Parameters:
    ///   - weights: EMA alpha weights [T] - higher = more weight on original x, lower = more weight on new y
    ///   - numSpeakers: Number of speakers
    init(weights: [Float], numSpeakers: Int) {
        self.windowSize = weights.count
        self.numSpeakers = numSpeakers
        self.defaultWeights = weights
        // Interleave weights for [T, numSpeakers] layout
        self.weights = weights.flatMap {
            Array(repeating: $0, count: numSpeakers)
        }
        self.tmpBuffer = Array(repeating: 0, count: self.weights.count)
    }
    
    /// Apply EMA filter
    /// - Parameters:
    ///   - x: Current filtered predictions
    ///   - y: New predictions (FIFO queue)
    ///   - result: Where to store the output
    ///   - count: Number of elements to update
    func update(_ x: UnsafePointer<Float>, with y: UnsafePointer<Float>, result: UnsafeMutablePointer<Float>, count: Int) throws {
        guard count > 0, count <= weights.count else {
            throw SortformerFilterError.invalidInputCount
        }
        
        let vCount = vDSP_Length(count)
        
        // EMA: x <- (x - y) * α + y
        
        // tmp <- (x - y) * α
        vDSP_vsbm(x, 1, y, 1, weights, 1, &tmpBuffer, 1, vCount)
        
        // x <- tmp + y = [(x - y) * α] + y
        vDSP_vadd(tmpBuffer, 1, y, 1, result, 1, vCount)
    }
    
    func reset() {
        self.weights = self.defaultWeights.flatMap {
            Array(repeating: $0, count: self.numSpeakers)
        }
    }
}

enum SortformerFilterError: Error {
    case invalidInputCount
}
