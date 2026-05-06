import Accelerate
@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

/// Regression coverage for the stride-aware paths in `KokoroAneArrays`.
///
/// Apple Neural Engine outputs frequently arrive as non-contiguous
/// MLMultiArrays: the fastest dim of a small-feature output is padded to a
/// hardware-friendly stride (e.g. `pred_dur_log` shape `[1, 14, 50]` with
/// strides `[896, 64, 1]`, padding 50→64; `[1, 512, 14]` with strides
/// `[16384, 32, 1]`, padding 14→32). A naive memcpy / contiguous read
/// interleaves valid rows with the padding gaps, scrambling downstream
/// stages and producing rasp-like artifacts in the final audio. These
/// tests pin the stride-aware fallbacks so a future refactor can't
/// silently regress.
final class KokoroAneArraysStrideTests: XCTestCase {

    // MARK: - Helpers

    /// Build a non-contiguous fp16 MLMultiArray with shape `[1, dim1, dim2]`
    /// and strides `[dim1 * paddedDim2, paddedDim2, 1]`. Valid elements at
    /// (0, t, c) for `c < dim2` are filled from `value(t, c)`; padding slots
    /// are filled with NaN so any code path that reads them will produce a
    /// detectable corruption (and `XCTAssertEqual` on Float will fail
    /// because NaN != NaN).
    private func makeStridedFp16(
        dim1: Int, dim2: Int, paddedDim2: Int, value: (Int, Int) -> Float
    ) throws -> MLMultiArray {
        precondition(paddedDim2 >= dim2)
        let totalPadded = dim1 * paddedDim2
        // NaN sentinel as fp16 (0x7E00 is a quiet NaN).
        let nanU16: UInt16 = 0x7E00
        var rawU16 = [UInt16](repeating: nanU16, count: totalPadded)
        // Bulk-convert via vImage to avoid per-element temporary pointers.
        var f32Flat = [Float](repeating: 0, count: dim1 * dim2)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                f32Flat[t * dim2 + c] = value(t, c)
            }
        }
        var packedU16 = [UInt16](repeating: 0, count: dim1 * dim2)
        f32Flat.withUnsafeMutableBufferPointer { srcBuf in
            packedU16.withUnsafeMutableBufferPointer { dstBuf in
                var src = vImage_Buffer(
                    data: srcBuf.baseAddress!, height: 1,
                    width: vImagePixelCount(dim1 * dim2),
                    rowBytes: dim1 * dim2 * MemoryLayout<Float>.stride)
                var dst = vImage_Buffer(
                    data: dstBuf.baseAddress!, height: 1,
                    width: vImagePixelCount(dim1 * dim2),
                    rowBytes: dim1 * dim2 * MemoryLayout<UInt16>.stride)
                vImageConvert_PlanarFtoPlanar16F(&src, &dst, 0)
            }
        }
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                rawU16[t * paddedDim2 + c] = packedU16[t * dim2 + c]
            }
        }
        // Allocate a heap buffer that survives the MLMultiArray's lifetime;
        // hand it off via the dataPointer initializer with a deallocator.
        let byteCount = totalPadded * MemoryLayout<UInt16>.stride
        let alignment = MemoryLayout<UInt16>.alignment
        let raw = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
        rawU16.withUnsafeBufferPointer { src in
            raw.copyMemory(from: src.baseAddress!, byteCount: byteCount)
        }
        let shape: [NSNumber] = [1, NSNumber(value: dim1), NSNumber(value: dim2)]
        let strides: [NSNumber] = [
            NSNumber(value: dim1 * paddedDim2),
            NSNumber(value: paddedDim2),
            1,
        ]
        return try MLMultiArray(
            dataPointer: raw,
            shape: shape,
            dataType: .float16,
            strides: strides,
            deallocator: { p in p.deallocate() }
        )
    }

    /// Same as `makeStridedFp16` but with fp32 storage.
    private func makeStridedFp32(
        dim1: Int, dim2: Int, paddedDim2: Int, value: (Int, Int) -> Float
    ) throws -> MLMultiArray {
        precondition(paddedDim2 >= dim2)
        let totalPadded = dim1 * paddedDim2
        var raw = [Float](repeating: .nan, count: totalPadded)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                raw[t * paddedDim2 + c] = value(t, c)
            }
        }
        let byteCount = totalPadded * MemoryLayout<Float>.stride
        let alignment = MemoryLayout<Float>.alignment
        let buf = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: alignment)
        raw.withUnsafeBufferPointer { src in
            buf.copyMemory(from: src.baseAddress!, byteCount: byteCount)
        }
        let shape: [NSNumber] = [1, NSNumber(value: dim1), NSNumber(value: dim2)]
        let strides: [NSNumber] = [
            NSNumber(value: dim1 * paddedDim2),
            NSNumber(value: paddedDim2),
            1,
        ]
        return try MLMultiArray(
            dataPointer: buf,
            shape: shape,
            dataType: .float32,
            strides: strides,
            deallocator: { p in p.deallocate() }
        )
    }

    private func expectedValue(_ t: Int, _ c: Int) -> Float {
        // Distinct, finite value per cell; chosen to be exactly representable
        // in fp16 to keep equality checks tight.
        return Float(t) + Float(c) * 0.0625
    }

    // MARK: - readFloats

    func testReadFloatsStridedFp16ReturnsPackedValues() throws {
        // Hand-verifiable case.
        let dim1 = 4
        let dim2 = 3
        let paddedDim2 = 4  // last dim padded 3 → 4
        let arr = try makeStridedFp16(
            dim1: dim1, dim2: dim2, paddedDim2: paddedDim2, value: expectedValue)
        XCTAssertEqual(arr.count, dim1 * dim2)  // logical, not padded
        // Sanity: arr really is non-contiguous.
        XCTAssertEqual(arr.strides.map { $0.intValue }, [dim1 * paddedDim2, paddedDim2, 1])

        let out = KokoroAneArrays.readFloats(arr)
        XCTAssertEqual(out.count, dim1 * dim2)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                XCTAssertEqual(
                    out[t * dim2 + c], expectedValue(t, c), accuracy: 1e-3,
                    "mismatch at (t=\(t), c=\(c))")
            }
        }
    }

    func testReadFloatsStridedFp32ReturnsPackedValues() throws {
        let dim1 = 4
        let dim2 = 3
        let paddedDim2 = 4
        let arr = try makeStridedFp32(
            dim1: dim1, dim2: dim2, paddedDim2: paddedDim2, value: expectedValue)
        XCTAssertEqual(arr.count, dim1 * dim2)

        let out = KokoroAneArrays.readFloats(arr)
        XCTAssertEqual(out.count, dim1 * dim2)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                XCTAssertEqual(
                    out[t * dim2 + c], expectedValue(t, c), accuracy: 1e-6,
                    "mismatch at (t=\(t), c=\(c))")
            }
        }
    }

    func testReadFloatsAneShapedStridedFp16() throws {
        // Realistic ANE shape: [1, 512, 14] with last dim padded 14 → 32.
        // Naive contiguous read would give 1*512*32 = 16384 values; the
        // stride-aware path must return exactly 1*512*14 = 7168.
        let dim1 = 512
        let dim2 = 14
        let paddedDim2 = 32
        // fp16 mantissa is 11 bits so values ≥ 1024 lose unit precision; keep
        // the encoded values in a range that's exactly representable so
        // accuracy assertions stay tight even at the high index.
        let value: (Int, Int) -> Float = { t, c in
            Float(t & 0x0F) + Float(c) * 0.0625
        }
        let arr = try makeStridedFp16(
            dim1: dim1, dim2: dim2, paddedDim2: paddedDim2, value: value)
        XCTAssertEqual(arr.count, dim1 * dim2)
        XCTAssertEqual(arr.count, 7168)
        XCTAssertEqual(arr.strides.map { $0.intValue }, [dim1 * paddedDim2, paddedDim2, 1])

        let out = KokoroAneArrays.readFloats(arr)
        XCTAssertEqual(out.count, 7168)
        // Spot-check boundaries.
        XCTAssertEqual(out[0], value(0, 0), accuracy: 1e-3)
        XCTAssertEqual(out[dim2 - 1], value(0, dim2 - 1), accuracy: 1e-3)
        XCTAssertEqual(out[dim2], value(1, 0), accuracy: 1e-3)
        XCTAssertEqual(
            out[(dim1 - 1) * dim2 + (dim2 - 1)],
            value(dim1 - 1, dim2 - 1), accuracy: 1e-3)
        // Full sweep: every element matches and no padding leaked.
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                let v = out[t * dim2 + c]
                XCTAssertFalse(v.isNaN, "padding leaked into output at (t=\(t), c=\(c))")
                XCTAssertEqual(v, value(t, c), accuracy: 1e-3)
            }
        }
    }

    // MARK: - float16Array(shape:from:)

    func testFloat16ArrayFromStridedFp16IsDenselyPacked() throws {
        let dim1 = 4
        let dim2 = 3
        let paddedDim2 = 4
        let src = try makeStridedFp16(
            dim1: dim1, dim2: dim2, paddedDim2: paddedDim2, value: expectedValue)

        let dst = try KokoroAneArrays.float16Array(shape: [1, dim1, dim2], from: src)
        XCTAssertEqual(dst.dataType, .float16)
        XCTAssertEqual(dst.shape.map { $0.intValue }, [1, dim1, dim2])
        XCTAssertEqual(dst.strides.map { $0.intValue }, [dim1 * dim2, dim2, 1])
        XCTAssertEqual(dst.count, dim1 * dim2)

        // Read back via the same stride-aware path and verify packed values.
        let readback = KokoroAneArrays.readFloats(dst)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                XCTAssertEqual(
                    readback[t * dim2 + c], expectedValue(t, c), accuracy: 1e-3)
            }
        }
    }

    func testFloat16ArrayFromStridedFp32IsDenselyPacked() throws {
        let dim1 = 4
        let dim2 = 3
        let paddedDim2 = 4
        let src = try makeStridedFp32(
            dim1: dim1, dim2: dim2, paddedDim2: paddedDim2, value: expectedValue)

        let dst = try KokoroAneArrays.float16Array(shape: [1, dim1, dim2], from: src)
        XCTAssertEqual(dst.dataType, .float16)
        XCTAssertEqual(dst.strides.map { $0.intValue }, [dim1 * dim2, dim2, 1])

        let readback = KokoroAneArrays.readFloats(dst)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                XCTAssertEqual(
                    readback[t * dim2 + c], expectedValue(t, c), accuracy: 1e-3)
            }
        }
    }

    // MARK: - float32Array(shape:from:)

    func testFloat32ArrayFromStridedFp16IsDenselyPacked() throws {
        let dim1 = 4
        let dim2 = 3
        let paddedDim2 = 4
        let src = try makeStridedFp16(
            dim1: dim1, dim2: dim2, paddedDim2: paddedDim2, value: expectedValue)

        let dst = try KokoroAneArrays.float32Array(shape: [1, dim1, dim2], from: src)
        XCTAssertEqual(dst.dataType, .float32)
        XCTAssertEqual(dst.strides.map { $0.intValue }, [dim1 * dim2, dim2, 1])

        let readback = KokoroAneArrays.readFloats(dst)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                XCTAssertEqual(
                    readback[t * dim2 + c], expectedValue(t, c), accuracy: 1e-3)
            }
        }
    }

    func testFloat32ArrayFromStridedFp32IsDenselyPacked() throws {
        let dim1 = 4
        let dim2 = 3
        let paddedDim2 = 4
        let src = try makeStridedFp32(
            dim1: dim1, dim2: dim2, paddedDim2: paddedDim2, value: expectedValue)

        let dst = try KokoroAneArrays.float32Array(shape: [1, dim1, dim2], from: src)
        XCTAssertEqual(dst.dataType, .float32)
        XCTAssertEqual(dst.strides.map { $0.intValue }, [dim1 * dim2, dim2, 1])

        let readback = KokoroAneArrays.readFloats(dst)
        for t in 0..<dim1 {
            for c in 0..<dim2 {
                XCTAssertEqual(
                    readback[t * dim2 + c], expectedValue(t, c), accuracy: 1e-6)
            }
        }
    }

    // MARK: - Contiguous baselines

    func testReadFloatsContiguousFp16Roundtrip() throws {
        // Sanity: the contiguous fast path still works when no padding exists.
        let values: [Float] = [1.0, -2.5, 3.25, 0.0, 0.5, -0.5]
        let arr = try KokoroAneArrays.float16Array(shape: [1, 2, 3], from: values)
        XCTAssertEqual(arr.strides.map { $0.intValue }, [6, 3, 1])
        let out = KokoroAneArrays.readFloats(arr)
        XCTAssertEqual(out.count, values.count)
        for (i, v) in values.enumerated() {
            XCTAssertEqual(out[i], v, accuracy: 1e-3)
        }
    }
}
