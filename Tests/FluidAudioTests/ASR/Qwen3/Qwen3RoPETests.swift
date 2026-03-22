import Foundation
import XCTest

@testable import FluidAudio

final class Qwen3RoPETests: XCTestCase {

    // MARK: - Position 0

    func testPosition0CosSinAreOneAndZero() {
        let rope = Qwen3RoPE()
        let (cos, sin) = rope.compute(position: 0)

        XCTAssertEqual(cos.count, Qwen3AsrConfig.headDim)
        XCTAssertEqual(sin.count, Qwen3AsrConfig.headDim)

        // At position 0, angle = 0 for all frequencies, so cos = 1.0, sin = 0.0
        for i in 0..<cos.count {
            XCTAssertEqual(cos[i], 1.0, accuracy: 1e-5, "cos[\(i)] at position 0 should be 1.0")
            XCTAssertEqual(sin[i], 0.0, accuracy: 1e-5, "sin[\(i)] at position 0 should be 0.0")
        }
    }

    // MARK: - Compute vs ComputeRange

    func testComputeRangeMatchesSingleCompute() {
        let rope = Qwen3RoPE()
        let (rangeCos, rangeSin) = rope.computeRange(startPosition: 5, count: 3)
        let headDim = Qwen3AsrConfig.headDim

        XCTAssertEqual(rangeCos.count, 3 * headDim)
        XCTAssertEqual(rangeSin.count, 3 * headDim)

        for p in 0..<3 {
            let (singleCos, singleSin) = rope.compute(position: 5 + p)
            let offset = p * headDim

            for i in 0..<headDim {
                XCTAssertEqual(
                    rangeCos[offset + i], singleCos[i], accuracy: 1e-5,
                    "cos mismatch at position \(5 + p), index \(i)"
                )
                XCTAssertEqual(
                    rangeSin[offset + i], singleSin[i], accuracy: 1e-5,
                    "sin mismatch at position \(5 + p), index \(i)"
                )
            }
        }
    }

    // MARK: - Fill

    func testFillMatchesCompute() {
        let rope = Qwen3RoPE()
        let headDim = Qwen3AsrConfig.headDim
        let position = 10

        let (expectedCos, expectedSin) = rope.compute(position: position)

        var cosBuffer = [Float](repeating: 0, count: headDim)
        var sinBuffer = [Float](repeating: 0, count: headDim)

        cosBuffer.withUnsafeMutableBufferPointer { cosBuf in
            sinBuffer.withUnsafeMutableBufferPointer { sinBuf in
                rope.fill(position: position, cosPtr: cosBuf.baseAddress!, sinPtr: sinBuf.baseAddress!)
            }
        }

        for i in 0..<headDim {
            XCTAssertEqual(cosBuffer[i], expectedCos[i], accuracy: 1e-5)
            XCTAssertEqual(sinBuffer[i], expectedSin[i], accuracy: 1e-5)
        }
    }

    // MARK: - Dynamic Fallback

    func testDynamicFallbackForLargePosition() {
        let rope = Qwen3RoPE()
        let headDim = Qwen3AsrConfig.headDim
        let largePosition = Qwen3AsrConfig.maxCacheSeqLen + 100

        let (cos, sin) = rope.compute(position: largePosition)
        XCTAssertEqual(cos.count, headDim)
        XCTAssertEqual(sin.count, headDim)

        // Values should be valid (not NaN/Inf)
        for i in 0..<headDim {
            XCTAssertFalse(cos[i].isNaN, "cos[\(i)] should not be NaN")
            XCTAssertFalse(sin[i].isNaN, "sin[\(i)] should not be NaN")
            XCTAssertFalse(cos[i].isInfinite, "cos[\(i)] should not be Inf")
            XCTAssertFalse(sin[i].isInfinite, "sin[\(i)] should not be Inf")
        }
    }

    func testDynamicFallbackFill() {
        let rope = Qwen3RoPE()
        let headDim = Qwen3AsrConfig.headDim
        let largePosition = Qwen3AsrConfig.maxCacheSeqLen + 50

        let (expectedCos, expectedSin) = rope.compute(position: largePosition)

        var cosBuffer = [Float](repeating: 0, count: headDim)
        var sinBuffer = [Float](repeating: 0, count: headDim)

        cosBuffer.withUnsafeMutableBufferPointer { cosBuf in
            sinBuffer.withUnsafeMutableBufferPointer { sinBuf in
                rope.fill(position: largePosition, cosPtr: cosBuf.baseAddress!, sinPtr: sinBuf.baseAddress!)
            }
        }

        for i in 0..<headDim {
            XCTAssertEqual(cosBuffer[i], expectedCos[i], accuracy: 1e-5)
            XCTAssertEqual(sinBuffer[i], expectedSin[i], accuracy: 1e-5)
        }
    }

    // MARK: - Symmetry

    func testConcatenatedHalvesSymmetry() {
        let rope = Qwen3RoPE()
        let headDim = Qwen3AsrConfig.headDim
        let halfDim = headDim / 2
        let position = 7

        let (cos, sin) = rope.compute(position: position)

        // The layout duplicates: cos[i] == cos[i + halfDim] for i in 0..<halfDim
        for i in 0..<halfDim {
            XCTAssertEqual(cos[i], cos[i + halfDim], accuracy: 1e-5, "cos should be symmetric at index \(i)")
            XCTAssertEqual(sin[i], sin[i + halfDim], accuracy: 1e-5, "sin should be symmetric at index \(i)")
        }
    }

    func testComputeRangeDynamicFallback() {
        let rope = Qwen3RoPE()
        let headDim = Qwen3AsrConfig.headDim
        let startPos = Qwen3AsrConfig.maxCacheSeqLen + 10
        let count = 3

        let (rangeCos, rangeSin) = rope.computeRange(startPosition: startPos, count: count)
        XCTAssertEqual(rangeCos.count, count * headDim)
        XCTAssertEqual(rangeSin.count, count * headDim)

        for p in 0..<count {
            let (singleCos, singleSin) = rope.compute(position: startPos + p)
            let offset = p * headDim
            for i in 0..<headDim {
                XCTAssertEqual(rangeCos[offset + i], singleCos[i], accuracy: 1e-5)
                XCTAssertEqual(rangeSin[offset + i], singleSin[i], accuracy: 1e-5)
            }
        }
    }
}
