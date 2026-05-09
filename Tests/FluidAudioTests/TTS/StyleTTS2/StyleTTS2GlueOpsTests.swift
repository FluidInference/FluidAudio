import CoreML
import XCTest

@testable import FluidAudio

final class StyleTTS2GlueOpsTests: XCTestCase {

    // MARK: - roundDurations

    func testRoundDurationsRejectsNon3DShape() {
        let arr = try! MLMultiArray(shape: [1, 4], dataType: .float32)
        XCTAssertThrowsError(try StyleTTS2GlueOps.roundDurations(arr))
    }

    func testRoundDurationsClampsAtLeastOne() throws {
        // Single time step, single channel filled with a very negative logit
        // → sigmoid ≈ 0 → rounded to 0 → clamped to 1.
        let arr = try MLMultiArray(shape: [1, 1, 1], dataType: .float32)
        arr[[0, 0, 0] as [NSNumber]] = NSNumber(value: Float(-50.0))
        let durations = try StyleTTS2GlueOps.roundDurations(arr)
        XCTAssertEqual(durations, [1])
    }

    func testRoundDurationsSumsSigmoidAcrossChannels() throws {
        // 2 time steps, 4 channels of all-zero logits → sigmoid = 0.5 each →
        // sum = 2.0 → rounded to 2.
        let arr = try MLMultiArray(shape: [1, 2, 4], dataType: .float32)
        for t in 0..<2 {
            for c in 0..<4 {
                arr[[0, t, c] as [NSNumber]] = NSNumber(value: Float(0))
            }
        }
        let durations = try StyleTTS2GlueOps.roundDurations(arr)
        XCTAssertEqual(durations, [2, 2])
    }

    // MARK: - buildAlignmentMatrix

    func testBuildAlignmentMatrixSimple() {
        let (m, total) = StyleTTS2GlueOps.buildAlignmentMatrix(durations: [2, 1, 3])
        XCTAssertEqual(total, 6)
        // row 0: 1 1 0 0 0 0
        // row 1: 0 0 1 0 0 0
        // row 2: 0 0 0 1 1 1
        let expected: [Float] = [
            1, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 1, 1,
        ]
        XCTAssertEqual(m, expected)
    }

    func testBuildAlignmentMatrixEmpty() {
        let (m, total) = StyleTTS2GlueOps.buildAlignmentMatrix(durations: [])
        XCTAssertEqual(total, 0)
        XCTAssertTrue(m.isEmpty)
    }

    // MARK: - matmulAligned

    func testMatmulAlignedExpandsFeaturesByDuration() {
        // features [C=2, K=3]:
        //   row0 = 1, 2, 3
        //   row1 = 4, 5, 6
        let features: [Float] = [
            1, 2, 3,
            4, 5, 6,
        ]
        // Durations [2, 1, 3] → alignment [3, 6] one-hot from buildAlignmentMatrix.
        let (aln, total) = StyleTTS2GlueOps.buildAlignmentMatrix(durations: [2, 1, 3])
        let out = StyleTTS2GlueOps.matmulAligned(
            features: features, channels: 2, realN: 3,
            alignment: aln, totalFrames: total)
        // Each frame copies the corresponding feature column:
        // out row 0: 1 1 2 3 3 3
        // out row 1: 4 4 5 6 6 6
        let expected: [Float] = [
            1, 1, 2, 3, 3, 3,
            4, 4, 5, 6, 6, 6,
        ]
        XCTAssertEqual(out, expected)
    }

    // MARK: - transposeLast2D

    func testTransposeLast2D() {
        // [rows=2, cols=3] row-major:
        //   1 2 3
        //   4 5 6
        // → [cols=3, rows=2]:
        //   1 4
        //   2 5
        //   3 6
        let src: [Float] = [1, 2, 3, 4, 5, 6]
        let out = StyleTTS2GlueOps.transposeLast2D(src, rows: 2, cols: 3)
        XCTAssertEqual(out, [1, 4, 2, 5, 3, 6])
    }

    // MARK: - hifiganShift

    func testHifiganShiftRightByOneAndCopiesColumnZero() {
        // 2 channels × 4 frames:
        //   ch0: 10 20 30 40
        //   ch1: 50 60 70 80
        let x: [Float] = [10, 20, 30, 40, 50, 60, 70, 80]
        let out = StyleTTS2GlueOps.hifiganShift(x, channels: 2, frames: 4)
        // col 0 stays put; col k>0 = original col (k-1):
        //   ch0: 10 10 20 30
        //   ch1: 50 50 60 70
        XCTAssertEqual(out, [10, 10, 20, 30, 50, 50, 60, 70])
    }

    func testHifiganShiftSingleFrameIsIdentity() {
        let x: [Float] = [1, 2, 3]
        let out = StyleTTS2GlueOps.hifiganShift(x, channels: 3, frames: 1)
        XCTAssertEqual(out, x)
    }

    // MARK: - blendStyle

    func testBlendStyleSplitsAtIndex128AndConvexCombines() {
        // Construct sPred and refS so we can read the formula directly.
        var sPred = [Float](repeating: 0, count: 256)
        var refS = [Float](repeating: 0, count: 256)
        for i in 0..<128 {
            sPred[i] = 1.0  // ref half of s_pred
            refS[i] = 3.0  // ref half of ref_s
            sPred[128 + i] = 7.0  // s half of s_pred
            refS[128 + i] = 9.0  // s half of ref_s
        }
        let alpha: Float = 0.25
        let beta: Float = 0.75
        let (ref, s) = StyleTTS2GlueOps.blendStyle(
            sPred256: sPred, refS256: refS, alpha: alpha, beta: beta)

        XCTAssertEqual(ref.count, 128)
        XCTAssertEqual(s.count, 128)

        // ref = α * 1 + (1 - α) * 3 = 0.25 + 2.25 = 2.5
        // s   = β * 7 + (1 - β) * 9 = 5.25 + 2.25 = 7.5
        for i in 0..<128 {
            XCTAssertEqual(ref[i], 2.5, accuracy: 1e-6)
            XCTAssertEqual(s[i], 7.5, accuracy: 1e-6)
        }
    }

    func testBlendStyleAlphaOneReturnsDiffusionRefHalf() {
        var sPred = [Float](repeating: 0, count: 256)
        var refS = [Float](repeating: 0, count: 256)
        for i in 0..<128 {
            sPred[i] = Float(i)  // ref half from diffusion
            refS[i] = -1.0
            sPred[128 + i] = -2.0
            refS[128 + i] = Float(i)  // s half from reference
        }
        let (ref, s) = StyleTTS2GlueOps.blendStyle(
            sPred256: sPred, refS256: refS, alpha: 1.0, beta: 0.0)
        for i in 0..<128 {
            XCTAssertEqual(ref[i], Float(i), accuracy: 1e-6)
            XCTAssertEqual(s[i], Float(i), accuracy: 1e-6)
        }
    }
}
