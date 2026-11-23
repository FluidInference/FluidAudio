import CoreML
import XCTest

@testable import FluidAudio

/// The bulk copy in `CtcKeywordSpotter.makePaddedAudioArray` must store exactly the
/// bits the previous per-element path (`array[i] = NSNumber(value:)`) stored.
final class CtcPaddedAudioBulkCopyTests: XCTestCase {

    /// The previous loop, kept here as the reference.
    private func perElement(_ samples: [Float], maxSamples: Int, dataType: MLMultiArrayDataType) throws -> MLMultiArray
    {
        let array = try MLMultiArray(shape: [NSNumber(value: maxSamples)], dataType: dataType)
        array.reset(to: 0)
        for i in 0..<min(samples.count, maxSamples) {
            array[i] = NSNumber(value: samples[i])
        }
        return array
    }

    private func storageBytes(_ array: MLMultiArray) -> [UInt8] {
        array.withUnsafeBytes { Array($0) }
    }

    private func assertSameBits(
        _ samples: [Float], dataType: MLMultiArrayDataType, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let maxSamples = samples.count + 5
        let expected = try perElement(samples, maxSamples: maxSamples, dataType: dataType)
        for rank in [1, 2] {
            let (actual, count) = try CtcKeywordSpotter.makePaddedAudioArray(
                samples, maxSamples: maxSamples, rank: rank, dataType: dataType)
            XCTAssertEqual(count, samples.count, file: file, line: line)
            let a = storageBytes(actual)
            let e = storageBytes(expected)
            XCTAssertEqual(a.count, e.count, file: file, line: line)
            if let first = zip(a, e).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
                let element = first.offset / (dataType == .float16 ? 2 : 4)
                XCTFail(
                    "rank \(rank): element \(element) (\(samples[min(element, samples.count - 1)])) differs",
                    file: file, line: line)
            }
        }
    }

    /// Every half-precision rounding boundary: each tie between neighbouring halves and
    /// the floats either side of it, across the whole half range and both signs.
    func testFloat16RoundingMatchesPerElementPath() throws {
        var samples: [Float] = []
        samples.reserveCapacity(65536 * 4)
        for bits in UInt16(0)...UInt16(0x7BFF) {  // every finite non-negative half below the largest
            let low = halfToFloat(bits)
            let high = halfToFloat(bits + 1)
            let tie = (Double(low) + Double(high)) / 2
            let t = Float(tie)
            for value in [low, t.nextDown, t, t.nextUp] {
                samples.append(value)
                samples.append(-value)
            }
        }
        try assertSameBits(samples, dataType: .float16)
    }

    func testFloat16OutOfRangeAndSpecialValuesMatch() throws {
        let samples: [Float] = [
            0, -0, 1, -1, 65504, 65519.99, 65520, 70000, -70000, .infinity, -.infinity,
            .leastNonzeroMagnitude, -.leastNonzeroMagnitude, 1e-8, 2.98e-8, 2.99e-8, 6e-8, 5.96e-8,
            .greatestFiniteMagnitude, -.greatestFiniteMagnitude,
        ]
        try assertSameBits(samples, dataType: .float16)
    }

    func testRandomAudioMatchesForBothTypes() throws {
        var generator = SystemRandomNumberGenerator()
        let samples = (0..<100_000).map { _ in Float.random(in: -1...1, using: &generator) }
        try assertSameBits(samples, dataType: .float16)
        try assertSameBits(samples, dataType: .float32)
    }

    private func halfToFloat(_ bits: UInt16) -> Float {
        var source = bits
        var result: Float = 0
        Float16Conversion.toFloat32(src: &source, dst: &result, count: 1)
        return result
    }
}
