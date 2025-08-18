//
//  JointNormalizer.swift
//  FluidAudio
//
//  Implements proper normalization for v3 joint network outputs
//

import Accelerate
import CoreML
import Foundation

/// Normalizes joint network outputs by applying log-softmax separately to vocab and duration logits
/// This fixes the scale mismatch issue in v3 models where vocab logits are ~130 points lower than duration logits
internal struct JointNormalizer {

    /// Apply log-softmax normalization separately to vocab and duration parts
    /// This is what the Python fix does that makes it work
    static func normalizeJointLogits(
        _ logits: MLMultiArray,
        vocabSize: Int = 8193,
        durationSize: Int = 5
    ) throws -> (tokenLogits: [Float], durationLogits: [Float]) {

        let totalElements = logits.count
        guard totalElements == vocabSize + durationSize else {
            throw ASRError.processingFailed(
                "Joint logits dimension mismatch: got \(totalElements), expected \(vocabSize + durationSize)")
        }

        // Extract raw logits
        let logitsPtr = logits.dataPointer.bindMemory(to: Float.self, capacity: totalElements)
        let vocabLogitsRaw = Array(UnsafeBufferPointer(start: logitsPtr, count: vocabSize))
        let durationLogitsRaw = Array(UnsafeBufferPointer(start: logitsPtr + vocabSize, count: durationSize))

        // Apply log-softmax to vocab logits
        let vocabNormalized = logSoftmax(vocabLogitsRaw)

        // Apply log-softmax to duration logits
        let durationNormalized = logSoftmax(durationLogitsRaw)

        return (vocabNormalized, durationNormalized)
    }

    /// Compute log-softmax: log(exp(x_i) / sum(exp(x_j)))
    /// Numerically stable implementation using the log-sum-exp trick
    private static func logSoftmax(_ logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }

        var result = [Float](repeating: 0, count: logits.count)

        // Find max for numerical stability
        var maxValue: Float = 0
        vDSP_maxv(logits, 1, &maxValue, vDSP_Length(logits.count))

        // Compute exp(x - max)
        var expValues = [Float](repeating: 0, count: logits.count)
        var negMaxValue = -maxValue
        vDSP_vsadd(logits, 1, &negMaxValue, &expValues, 1, vDSP_Length(logits.count))
        vvexpf(&expValues, expValues, [Int32(logits.count)])

        // Sum of exp values
        var sumExp: Float = 0
        vDSP_sve(expValues, 1, &sumExp, vDSP_Length(logits.count))

        // Compute log(exp(x - max) / sum) = x - max - log(sum)
        let logSum = logf(sumExp)
        var negLogSum = -logSum
        vDSP_vsadd(logits, 1, &negMaxValue, &result, 1, vDSP_Length(logits.count))
        vDSP_vsadd(result, 1, &negLogSum, &result, 1, vDSP_Length(logits.count))

        return result
    }

    /// Alternative simpler implementation for debugging
    static func logSoftmaxSimple(_ logits: [Float]) -> [Float] {
        let maxLogit = logits.max() ?? 0
        let expValues = logits.map { expf($0 - maxLogit) }
        let sumExp = expValues.reduce(0, +)
        let logSumExp = logf(sumExp) + maxLogit
        return logits.map { $0 - logSumExp }
    }
}
