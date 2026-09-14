import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Regression tests for the padded-stride MLMultiArray bug class.
///
/// `ANEMemoryUtils.calculateOptimalStrides` pads innermost dimensions to tile
/// boundaries, so shapes like [1, T, 1] get a row stride of 16 — linear writes
/// through `dataPointer` then land at the wrong logical positions. This silently
/// corrupted the split-graph `output_mask` input (surfaced as ~90% frame agreement
/// instead of ~100%). These tests pin both directions:
/// - reads: `Nemotron3Models.floats(from:)` must honor strides for padded layouts,
/// - writes: buffers written linearly must actually be contiguous.
final class Nemotron3TensorLayoutTests: XCTestCase {

    /// Non-tile-aligned innermost sizes that trigger stride padding.
    private let awkwardSizes = [1, 3, 5, 7, 9, 15, 17, 33, 340, 341]

    private func isContiguous(_ array: MLMultiArray) -> Bool {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        var expected = 1
        for dim in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[dim] != expected { return false }
            expected *= shape[dim]
        }
        return true
    }

    func testAlignedArraysPadNonTileAlignedInnermostDims() throws {
        // Documents the underlying behavior this bug class depends on. If this ever
        // starts failing (helper made contiguous), the guards below become moot — fine.
        let optimizer = ANEMemoryOptimizer()
        let padded = try optimizer.createAlignedArray(shape: [1, 8, 1], dataType: .float32)
        XCTAssertFalse(
            isContiguous(padded),
            "expected [1, 8, 1] aligned array to be stride-padded; update layout assumptions")
    }

    func testFloatsFromStridedArrayHonorsStrides() throws {
        let optimizer = ANEMemoryOptimizer()
        for t in awkwardSizes {
            let array = try optimizer.createAlignedArray(
                shape: [1, NSNumber(value: t), 1], dataType: .float32)
            // Write via logical (stride-aware) subscripting.
            for i in 0..<t {
                array[[0, NSNumber(value: i), 0]] = NSNumber(value: Float(i) + 0.5)
            }
            let floats = Nemotron3Models.floats(from: array)
            XCTAssertEqual(floats.count, t)
            for i in 0..<t {
                XCTAssertEqual(
                    floats[i], Float(i) + 0.5, accuracy: 0,
                    "strided read scrambled at T=\(t), index \(i)")
            }
        }
    }

    func testFloatsFromPaddedRowsMatchesSubscriptPath() throws {
        // Mimic the model's real output layout: [1, T, 8] fp16-like padding (row stride 16).
        let optimizer = ANEMemoryOptimizer()
        for t in [5, 34, 341] {
            let array = try optimizer.createAlignedArray(
                shape: [1, NSNumber(value: t), 8], dataType: .float32)
            for frame in 0..<t {
                for s in 0..<8 {
                    array[[0, NSNumber(value: frame), NSNumber(value: s)]] =
                        NSNumber(value: Float(frame * 8 + s))
                }
            }
            let floats = Nemotron3Models.floats(from: array)
            for i in 0..<(t * 8) {
                XCTAssertEqual(floats[i], Float(i), accuracy: 0, "row-padded read failed at T=\(t)")
            }
        }
    }

    func testPlainMultiArraysAreContiguousForMaskShapes() throws {
        // The split-graph mask inputs rely on plain MLMultiArray being contiguous for
        // linear dataPointer writes. Pin that assumption for the awkward shapes.
        for t in awkwardSizes {
            let mask = try MLMultiArray(shape: [1, NSNumber(value: t), 1], dataType: .float32)
            XCTAssertTrue(isContiguous(mask), "plain [1, \(t), 1] array is not contiguous")
            let bias = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: t)], dataType: .float32)
            XCTAssertTrue(isContiguous(bias), "plain [1, 1, 1, \(t)] array is not contiguous")
        }
    }
}
