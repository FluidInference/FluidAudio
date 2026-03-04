import Foundation
import XCTest

@testable import FluidAudio

final class WeightInterpolationTests: XCTestCase {

    // MARK: - Resample 1D

    func testResampleIdentityWhenLengthsMatch() {
        let input: [Float] = [1, 2, 3, 4]
        let result = WeightInterpolation.resample(input, to: 4)
        XCTAssertEqual(result, input, "Same-length resample should return input unchanged")
    }

    func testResampleUpsampling() {
        let input: [Float] = [0, 1]
        let result = WeightInterpolation.resample(input, to: 4)

        XCTAssertEqual(result.count, 4)
        // Half-pixel offset: output positions map to (i+0.5)/2 - 0.5 in input space
        // Values should be monotonically increasing from near 0 to near 1
        XCTAssertTrue(result[0] < result[3], "Upsampled values should be monotonically ordered")
        for value in result {
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThanOrEqual(value, 1)
        }
    }

    func testResampleDownsampling() {
        let input: [Float] = [0, 0.5, 1, 0.5]
        let result = WeightInterpolation.resample(input, to: 2)

        XCTAssertEqual(result.count, 2)
        // Downsampled values should be somewhere within the input range
        for value in result {
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThanOrEqual(value, 1)
        }
    }

    func testResampleUsesHalfPixelOffsetMapping() {
        let input: [Float] = [0, 10, 20, 30]
        let result = WeightInterpolation.resample(input, to: 2)

        XCTAssertEqual(result.count, 2)
        XCTAssertEqual(result[0], 5, accuracy: 1e-5)
        XCTAssertEqual(result[1], 25, accuracy: 1e-5)
    }

    func testResampleMatchesInterpolationCoefficients() {
        let input = (0..<16).map { Float($0) * 0.25 }
        let outputLength = 7

        let direct = WeightInterpolation.resample(input, to: outputLength)
        let coefficients = WeightInterpolation.InterpolationCoefficients(
            inputLength: input.count,
            outputLength: outputLength
        )
        let gathered = coefficients.interpolate(input)

        XCTAssertEqual(direct.count, gathered.count)
        for (lhs, rhs) in zip(direct, gathered) {
            XCTAssertEqual(lhs, rhs, accuracy: 1e-5)
        }
    }

    func testResampleEmptyInputReturnsEmpty() {
        let result = WeightInterpolation.resample([], to: 5)
        XCTAssertTrue(result.isEmpty)
    }

    func testResampleZeroOutputLengthReturnsEmpty() {
        let result = WeightInterpolation.resample([1, 2, 3], to: 0)
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Resample 2D

    func testResample2DConsistency() {
        let input: [[Float]] = [[1, 2, 3], [4, 5, 6]]
        let outputLength = 5

        let result2D = WeightInterpolation.resample2D(input, to: outputLength)
        let row0 = WeightInterpolation.resample(input[0], to: outputLength)
        let row1 = WeightInterpolation.resample(input[1], to: outputLength)

        XCTAssertEqual(result2D.count, 2)
        for i in 0..<outputLength {
            XCTAssertEqual(result2D[0][i], row0[i], accuracy: 1e-6, "2D row 0 should match 1D resample")
            XCTAssertEqual(result2D[1][i], row1[i], accuracy: 1e-6, "2D row 1 should match 1D resample")
        }
    }

    func testResample2DBroadcastsRows() {
        let inputs: [[Float]] = [
            [1, 3, 5, 7],
            [2, 4, 6, 8],
        ]

        let outputs = WeightInterpolation.resample2D(inputs, to: 2)

        XCTAssertEqual(outputs.count, 2)
        XCTAssertEqual(outputs[0][0], 2, accuracy: 1e-5)
        XCTAssertEqual(outputs[0][1], 6, accuracy: 1e-5)
        XCTAssertEqual(outputs[1][0], 3, accuracy: 1e-5)
        XCTAssertEqual(outputs[1][1], 7, accuracy: 1e-5)
    }

    func testResample2DEmptyReturnsEmpty() {
        let result = WeightInterpolation.resample2D([], to: 5)
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - Zoom

    func testZoomDoubleLength() {
        let input: [Float] = [1, 2, 3, 4]
        let result = WeightInterpolation.zoom(input, factor: 2.0)

        XCTAssertEqual(result.count, 8, "Zoom factor 2 should double the length")
    }

    func testZoomHalfLength() {
        let input: [Float] = [1, 2, 3, 4]
        let result = WeightInterpolation.zoom(input, factor: 0.5)

        XCTAssertEqual(result.count, 2, "Zoom factor 0.5 should halve the length")
    }

    func testZoomEmptyReturnsEmpty() {
        let result = WeightInterpolation.zoom([], factor: 2.0)
        XCTAssertTrue(result.isEmpty)
    }

    func testZoomZeroFactorReturnsEmpty() {
        let result = WeightInterpolation.zoom([1, 2, 3], factor: 0)
        XCTAssertTrue(result.isEmpty)
    }

    // MARK: - SlidingWindow

    func testSlidingWindowTimeForFrame() {
        let window = SlidingWindow(start: 0.5, duration: 1.0, step: 0.1)
        let time = window.time(forFrame: 3)
        XCTAssertEqual(time, 0.8, accuracy: 1e-10, "0.5 + 3 * 0.1 = 0.8")
    }

    func testSlidingWindowSegmentForFrame() {
        let window = SlidingWindow(start: 0.0, duration: 2.0, step: 0.5)
        let segment = window.segment(forFrame: 2)

        XCTAssertEqual(segment.start, 1.0, accuracy: 1e-10, "0.0 + 2 * 0.5 = 1.0")
        XCTAssertEqual(segment.end, 3.0, accuracy: 1e-10, "1.0 + 2.0 = 3.0")
    }

    func testSegmentHashable() {
        let a = Segment(start: 1.0, end: 2.0)
        let b = Segment(start: 1.0, end: 2.0)
        XCTAssertEqual(a, b)
        XCTAssertEqual(a.hashValue, b.hashValue)
    }
}
