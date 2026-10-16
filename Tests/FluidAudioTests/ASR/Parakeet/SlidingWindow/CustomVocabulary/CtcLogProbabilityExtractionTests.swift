import CoreML
import XCTest

@testable import FluidAudio

final class CtcLogProbabilityExtractionTests: XCTestCase {
    // A flat typed read can silently transpose the vocabulary/time axes or
    // consume ANE row padding, changing vocabulary scores and corrections.
    // Compare the complete normalized result with coordinate-based scalar reads.
    func testBothCTCLayoutsAndPaddedStoragePreserveScoresExactly() throws {
        var dataTypes: [MLMultiArrayDataType] = [.float16, .float32, .float64, .int32]
        #if compiler(>=6.2)
        if #available(macOS 26.0, iOS 26.0, tvOS 26.0, visionOS 26.0, *) { dataTypes.append(.int8) }
        #endif
        let shapes: [[NSNumber]] = [[1, 3, 7], [1, 7, 1, 3]]
        for dataType in dataTypes {
            for shape in shapes {
                for padded in [false, true] {
                    let array =
                        padded
                        ? try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: dataType)
                        : try MLMultiArray(shape: shape, dataType: dataType)
                    let rank = shape.count
                    let values: [Double] = [
                        -12.125, 0.333333333, 17.0000007, 0.000000059604644775390625, -0.0, 4.25, -3,
                    ]
                    for t in 0..<3 {
                        for v in 0..<7 {
                            let coordinates: [NSNumber] =
                                rank == 3
                                ? [0, NSNumber(value: t), NSNumber(value: v)]
                                : [0, NSNumber(value: v), 0, NSNumber(value: t)]
                            array[coordinates] = NSNumber(value: values[(v + t) % values.count])
                        }
                    }
                    let settings: [(Float, Float)] = [(1, 0), (0.7, 1.25)]
                    for (temperature, bias) in settings {
                        let actual = try CtcKeywordSpotter.makeLogProbs(
                            from: array, temperature: temperature, blankBias: bias, blankId: 6)
                        let expected = referenceLogProbs(array, temperature: temperature, blankBias: bias, blankId: 6)
                        XCTAssertEqual(
                            actual.map { $0.map(\.bitPattern) }, expected.map { $0.map(\.bitPattern) },
                            "type=\(dataType), shape=\(shape), padded=\(padded), temperature=\(temperature)")
                    }
                }
            }
        }
    }

    // Half subnormals must survive widening; FP64/int32 values must round to
    // Float before normalization, just as NSNumber.floatValue did previously.
    func testScalarConversionExtremesPreserveTheFloatScoringInput() throws {
        let cases: [(MLMultiArrayDataType, [Double])] = [
            (.float16, [0, -0.0, 0.000000059604644775390625, -0.00000011920928955078125, -65504, 0.00006103515625]),
            (.float32, [0, -0.0, 1.00000007, -1.00000007, -10000, 0.00000000000000000001]),
            (.float64, [0, -0.0, 1.00000007, -1.00000007, -10000.0003, 0.00000000000000000001]),
            (.int32, [16_777_217, 16_777_216, 16_777_215, -16_777_217, 0, -1]),
        ]
        for (dataType, values) in cases {
            let array = try MLMultiArray(shape: [1, 1, NSNumber(value: values.count)], dataType: dataType)
            for (index, value) in values.enumerated() { array[index] = NSNumber(value: value) }
            let actual = try CtcKeywordSpotter.makeLogProbs(from: array, blankId: values.count - 1)
            let expected = referenceLogProbs(array, temperature: 1, blankBias: 0, blankId: values.count - 1)
            XCTAssertEqual(actual.map { $0.map(\.bitPattern) }, expected.map { $0.map(\.bitPattern) }, "\(dataType)")
        }
    }

    /// Unoptimized coordinate reads deliberately define the old scalar-to-Float
    /// contract independently of the typed reader and its physical storage layout.
    private func referenceLogProbs(
        _ array: MLMultiArray, temperature: Float, blankBias: Float, blankId: Int
    ) -> [[Float]] {
        let rank = array.shape.count
        let timeSteps = array.shape[rank == 3 ? 1 : 3].intValue
        let vocabSize = array.shape[rank == 3 ? 2 : 1].intValue
        return (0..<timeSteps).map { t in
            let logits: [Float] = (0..<vocabSize).map { v in
                let coordinates: [NSNumber] =
                    rank == 3
                    ? [0, NSNumber(value: t), NSNumber(value: v)]
                    : [0, NSNumber(value: v), 0, NSNumber(value: t)]
                return array[coordinates].floatValue
            }
            let scaled = temperature == 1 ? logits : logits.map { $0 / temperature }
            let maximum = scaled.max() ?? 0
            let sum = scaled.reduce(Float(0)) { $0 + expf($1 - maximum) }
            let logSum = logf(sum)
            var result = scaled.map { ($0 - maximum) - logSum }
            if blankBias != 0, blankId < result.count { result[blankId] -= blankBias }
            return result
        }
    }
}
