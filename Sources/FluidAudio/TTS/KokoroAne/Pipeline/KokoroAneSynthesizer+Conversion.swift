import Accelerate
@preconcurrency import CoreML
import Foundation

/// MLMultiArray builders + fp16 ↔ fp32 conversions used by the chain.
///
/// All slicing helpers assume row-major contiguous storage with a leading
/// batch dim of 1. CoreML outputs from our exported pipelines satisfy this
/// invariant; the shape preconditions enforce it at runtime so a wiring bug
/// fails loudly instead of corrupting the next stage's input.
enum KokoroAneArrays {

    // MARK: - vImage helpers

    private static func convertF32toF16(
        src: UnsafePointer<Float>, dst: UnsafeMutablePointer<UInt16>, count: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src), height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.stride)
        var dstBuf = vImage_Buffer(
            data: dst, height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.stride)
        vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
    }

    private static func convertF16toF32(
        src: UnsafePointer<UInt16>, dst: UnsafeMutablePointer<Float>, count: Int
    ) {
        var srcBuf = vImage_Buffer(
            data: UnsafeMutableRawPointer(mutating: src), height: 1,
            width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<UInt16>.stride)
        var dstBuf = vImage_Buffer(
            data: dst, height: 1, width: vImagePixelCount(count),
            rowBytes: count * MemoryLayout<Float>.stride)
        vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
    }

    /// Element-wise NSNumber-bridged copy. Slow path — used only when the
    /// source MLMultiArray has an unexpected dtype (not fp16/fp32). Trips
    /// an assert in debug builds so we notice if a stage starts emitting an
    /// unexpected dtype instead of silently taking the ~100× slower path.
    private static func genericCopy(_ source: MLMultiArray, into dst: MLMultiArray, count: Int) {
        assertionFailure(
            "KokoroAneArrays: unexpected source dtype \(source.dataType.rawValue); using slow NSNumber copy")
        for i in 0..<count {
            dst[i] = NSNumber(value: source[i].floatValue)
        }
    }

    // MARK: - Float16 (UInt16-backed) builders

    /// Build a Float16 MLMultiArray and fill it from a Float32 source.
    static func float16Array(shape: [Int], from source: [Float]) throws -> MLMultiArray {
        let total = shape.reduce(1, *)
        precondition(
            source.count == total,
            "float16Array: shape \(shape) (\(total)) ≠ source.count \(source.count)")
        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try MLMultiArray(shape: nsShape, dataType: .float16)
        let dst = arr.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        source.withUnsafeBufferPointer { srcBuf in
            convertF32toF16(src: srcBuf.baseAddress!, dst: dst, count: total)
        }
        return arr
    }

    /// Build a Float16 MLMultiArray, copying from a Float16-backed source array.
    ///
    /// Handles non-contiguous (stride-padded) sources. CoreML's ANE backend
    /// pads the fastest dim of small-feature outputs (e.g. shape `[1, 512, 14]`
    /// with strides `[16384, 32, 1]` — last dim 14 padded to 32). A naive
    /// memcpy of `total * 2` bytes would interleave valid rows with the
    /// padding gaps, scrambling the output. Detect non-contiguous storage and
    /// fall back to a stride-aware element walk in that case.
    static func float16Array(shape: [Int], from source: MLMultiArray) throws -> MLMultiArray {
        // Same shape + same dtype → just copy bytes.
        let total = shape.reduce(1, *)
        precondition(
            source.count == total,
            "float16Array(from MLMultiArray): source has \(source.count) elements, shape implies \(total)")
        let nsShape = shape.map { NSNumber(value: $0) }
        let dst = try MLMultiArray(shape: nsShape, dataType: .float16)
        let isContig = isRowMajorContiguous(source)
        if source.dataType == .float16 {
            if isContig {
                memcpy(dst.dataPointer, source.dataPointer, total * MemoryLayout<UInt16>.size)
            } else {
                stridedCopyU16(source: source, dst: dst, total: total)
            }
            return dst
        }
        if source.dataType == .float32 {
            let dstU16 = dst.dataPointer.bindMemory(to: UInt16.self, capacity: total)
            if isContig {
                let f32 = source.dataPointer.bindMemory(to: Float.self, capacity: total)
                convertF32toF16(src: f32, dst: dstU16, count: total)
            } else {
                // Walk source f32 strides into a packed temp, then convert.
                var tmp = [Float](repeating: 0, count: total)
                stridedReadF32(source: source, into: &tmp)
                tmp.withUnsafeBufferPointer { buf in
                    convertF32toF16(src: buf.baseAddress!, dst: dstU16, count: total)
                }
            }
            return dst
        }
        genericCopy(source, into: dst, count: total)
        return dst
    }

    /// Returns true when `arr.strides` is densely row-major over `arr.shape`.
    private static func isRowMajorContiguous(_ arr: MLMultiArray) -> Bool {
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }
        guard shape.count == strides.count, !shape.isEmpty else { return true }
        var expected = 1
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[axis] != expected { return false }
            expected *= shape[axis]
        }
        return true
    }

    /// Stride-aware copy from an fp16 MLMultiArray to a packed fp16 destination.
    private static func stridedCopyU16(
        source: MLMultiArray, dst: MLMultiArray, total: Int
    ) {
        let shape = source.shape.map { $0.intValue }
        let strides = source.strides.map { $0.intValue }
        let srcPtr = source.dataPointer.bindMemory(to: UInt16.self, capacity: 1)
        let dstPtr = dst.dataPointer.bindMemory(to: UInt16.self, capacity: total)
        var coord = [Int](repeating: 0, count: shape.count)
        for i in 0..<total {
            var off = 0
            for d in 0..<shape.count { off += coord[d] * strides[d] }
            dstPtr[i] = srcPtr[off]
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                coord[d] += 1
                if coord[d] < shape[d] { break }
                coord[d] = 0
            }
        }
    }

    /// Stride-aware read from an fp32 MLMultiArray into a packed fp32 buffer.
    private static func stridedReadF32(source: MLMultiArray, into dst: inout [Float]) {
        let shape = source.shape.map { $0.intValue }
        let strides = source.strides.map { $0.intValue }
        let srcPtr = source.dataPointer.bindMemory(to: Float.self, capacity: 1)
        var coord = [Int](repeating: 0, count: shape.count)
        for i in 0..<dst.count {
            var off = 0
            for d in 0..<shape.count { off += coord[d] * strides[d] }
            dst[i] = srcPtr[off]
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                coord[d] += 1
                if coord[d] < shape[d] { break }
                coord[d] = 0
            }
        }
    }

    // MARK: - Float32 builders

    static func float32Array(shape: [Int], from source: [Float]) throws -> MLMultiArray {
        let total = shape.reduce(1, *)
        precondition(
            source.count == total,
            "float32Array: shape \(shape) (\(total)) ≠ source.count \(source.count)")
        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try MLMultiArray(shape: nsShape, dataType: .float32)
        let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: total)
        source.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: total)
        }
        return arr
    }

    /// Build a Float32 MLMultiArray from a Float16-backed source.
    /// Stride-aware: see `float16Array(shape:from:)` for the rationale.
    static func float32Array(shape: [Int], from source: MLMultiArray) throws -> MLMultiArray {
        let total = shape.reduce(1, *)
        precondition(
            source.count == total,
            "float32Array(from MLMultiArray): source has \(source.count) elements, shape implies \(total)")
        let nsShape = shape.map { NSNumber(value: $0) }
        let dst = try MLMultiArray(shape: nsShape, dataType: .float32)
        let isContig = isRowMajorContiguous(source)
        if source.dataType == .float32 {
            if isContig {
                memcpy(dst.dataPointer, source.dataPointer, total * MemoryLayout<Float>.size)
            } else {
                let dstF = dst.dataPointer.bindMemory(to: Float.self, capacity: total)
                var tmp = [Float](repeating: 0, count: total)
                stridedReadF32(source: source, into: &tmp)
                tmp.withUnsafeBufferPointer { buf in
                    dstF.update(from: buf.baseAddress!, count: total)
                }
            }
            return dst
        }
        if source.dataType == .float16 {
            let dstF = dst.dataPointer.bindMemory(to: Float.self, capacity: total)
            if isContig {
                let srcU16 = source.dataPointer.bindMemory(to: UInt16.self, capacity: total)
                convertF16toF32(src: srcU16, dst: dstF, count: total)
            } else {
                stridedReadF16ToF32(source: source, dst: dstF, total: total)
            }
            return dst
        }
        genericCopy(source, into: dst, count: total)
        return dst
    }

    /// Stride-aware read from an fp16 MLMultiArray into a packed fp32 buffer.
    private static func stridedReadF16ToF32(
        source: MLMultiArray, dst: UnsafeMutablePointer<Float>, total: Int
    ) {
        let shape = source.shape.map { $0.intValue }
        let strides = source.strides.map { $0.intValue }
        let srcPtr = source.dataPointer.bindMemory(to: UInt16.self, capacity: 1)
        var coord = [Int](repeating: 0, count: shape.count)
        for i in 0..<total {
            var off = 0
            for d in 0..<shape.count { off += coord[d] * strides[d] }
            dst[i] = Float(Float16(bitPattern: srcPtr[off]))
            for d in stride(from: shape.count - 1, through: 0, by: -1) {
                coord[d] += 1
                if coord[d] < shape[d] { break }
                coord[d] = 0
            }
        }
    }

    // MARK: - Int32 builders

    static func int32Array(shape: [Int], from source: [Int32]) throws -> MLMultiArray {
        let total = shape.reduce(1, *)
        precondition(
            source.count == total,
            "int32Array: shape \(shape) (\(total)) ≠ source.count \(source.count)")
        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try MLMultiArray(shape: nsShape, dataType: .int32)
        let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: total)
        source.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: total)
        }
        return arr
    }

    /// Pad-of-ones int32 attention mask of shape `[1, T_enc]`.
    static func attentionMask(length: Int) throws -> MLMultiArray {
        return try int32Array(shape: [1, length], from: [Int32](repeating: 1, count: length))
    }

    // MARK: - Slicing

    /// Assert that `arr` is row-major contiguous (densely packed). MLMultiArray
    /// can in principle expose strided views; our pipeline never produces them,
    /// but a strided source would silently corrupt the memcpy-based slices.
    private static func assertDenseContiguous(
        _ arr: MLMultiArray, _ context: @autoclosure () -> String
    ) {
        let shape = arr.shape.map { $0.intValue }
        let strides = arr.strides.map { $0.intValue }
        var expected = 1
        for axis in stride(from: shape.count - 1, through: 0, by: -1) {
            precondition(
                strides[axis] == expected,
                "\(context()): non-contiguous source (shape \(shape), strides \(strides))")
            expected *= shape[axis]
        }
    }

    /// Slice a Float16 MLMultiArray with shape `[1, T, ...rest]` to the first
    /// `newT` frames along the second dim. Since the leading batch dim is 1
    /// and the time dim is the second axis, the per-frame trailing block is
    /// contiguous in row-major memory, so this is a simple prefix copy.
    /// Use this for shapes like `[1, T]` (F0/N curves) and `[1, T, H]`
    /// (sine_waves).
    static func sliceLeadingTimeFp16(
        _ arr: MLMultiArray, newShape: [Int]
    ) throws -> MLMultiArray {
        precondition(arr.dataType == .float16, "sliceLeadingTimeFp16 requires fp16 source")
        precondition(
            arr.shape.first?.intValue == 1,
            "sliceLeadingTimeFp16 requires leading batch dim == 1, got shape \(arr.shape)")
        precondition(
            newShape.first == 1,
            "sliceLeadingTimeFp16 requires newShape leading dim == 1, got \(newShape)")
        assertDenseContiguous(arr, "sliceLeadingTimeFp16")
        let total = newShape.reduce(1, *)
        precondition(total > 0, "sliceLeadingTimeFp16: empty newShape \(newShape)")
        precondition(
            total <= arr.count,
            "sliceLeadingTimeFp16: requested \(total) elements but source has \(arr.count)")
        let nsShape = newShape.map { NSNumber(value: $0) }
        let dst = try MLMultiArray(shape: nsShape, dataType: .float16)
        memcpy(dst.dataPointer, arr.dataPointer, total * MemoryLayout<UInt16>.size)
        return dst
    }

    /// Slice a Float32 MLMultiArray with shape `[1, T, ...rest]` to the first
    /// `newT` frames. Used for the fp32 F0_curve hand-off into the Vocoder.
    static func sliceLeadingTimeFp32(
        _ arr: MLMultiArray, newShape: [Int]
    ) throws -> MLMultiArray {
        precondition(arr.dataType == .float32, "sliceLeadingTimeFp32 requires fp32 source")
        precondition(
            arr.shape.first?.intValue == 1,
            "sliceLeadingTimeFp32 requires leading batch dim == 1, got shape \(arr.shape)")
        precondition(
            newShape.first == 1,
            "sliceLeadingTimeFp32 requires newShape leading dim == 1, got \(newShape)")
        assertDenseContiguous(arr, "sliceLeadingTimeFp32")
        let total = newShape.reduce(1, *)
        precondition(total > 0, "sliceLeadingTimeFp32: empty newShape \(newShape)")
        precondition(
            total <= arr.count,
            "sliceLeadingTimeFp32: requested \(total) elements but source has \(arr.count)")
        let nsShape = newShape.map { NSNumber(value: $0) }
        let dst = try MLMultiArray(shape: nsShape, dataType: .float32)
        memcpy(dst.dataPointer, arr.dataPointer, total * MemoryLayout<Float>.size)
        return dst
    }

    /// Slice a Float16 MLMultiArray with shape `[1, C, T]` to `[1, C, newT]`,
    /// where T is the trailing dim. C runs of `newT` elements each, source
    /// stride T. Used for the asr hand-off into the Vocoder.
    static func sliceTrailingTimeFp16(
        _ arr: MLMultiArray, channels: Int, oldT: Int, newT: Int
    ) throws -> MLMultiArray {
        precondition(arr.dataType == .float16, "sliceTrailingTimeFp16 requires fp16 source")
        precondition(
            arr.shape.first?.intValue == 1,
            "sliceTrailingTimeFp16 requires leading batch dim == 1, got shape \(arr.shape)")
        assertDenseContiguous(arr, "sliceTrailingTimeFp16")
        precondition(
            arr.count == channels * oldT,
            "shape mismatch: arr.count \(arr.count) ≠ \(channels)*\(oldT)")
        precondition(newT > 0, "sliceTrailingTimeFp16: newT must be positive, got \(newT)")
        precondition(channels > 0, "sliceTrailingTimeFp16: channels must be positive, got \(channels)")
        precondition(newT <= oldT, "newT \(newT) > oldT \(oldT)")
        let nsShape = [NSNumber(value: 1), NSNumber(value: channels), NSNumber(value: newT)]
        let dst = try MLMultiArray(shape: nsShape, dataType: .float16)
        let srcPtr = arr.dataPointer.bindMemory(to: UInt16.self, capacity: arr.count)
        let dstPtr = dst.dataPointer.bindMemory(to: UInt16.self, capacity: channels * newT)
        for c in 0..<channels {
            (dstPtr + c * newT).update(from: srcPtr + c * oldT, count: newT)
        }
        return dst
    }

    // MARK: - Output extraction

    /// Read a Float32 MLMultiArray output into a Swift `[Float]` (flat).
    /// Stride-aware: ANE outputs may pad the fastest dim (e.g.
    /// `pred_dur_log` shape `[1, 14, 50]` with strides `[896, 64, 1]`,
    /// padding 50→64). A naive contiguous read interleaves valid rows with
    /// padding gaps and silently scrambles the result.
    static func readFloats(_ arr: MLMultiArray) -> [Float] {
        let count = arr.count
        let isContig = isRowMajorContiguous(arr)
        if arr.dataType == .float32 {
            var out = [Float](repeating: 0, count: count)
            if isContig {
                let p = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
                out.withUnsafeMutableBufferPointer { buf in
                    buf.baseAddress!.update(from: p, count: count)
                }
            } else {
                stridedReadF32(source: arr, into: &out)
            }
            return out
        }
        if arr.dataType == .float16 {
            var out = [Float](repeating: 0, count: count)
            if isContig {
                let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
                out.withUnsafeMutableBufferPointer { outBuf in
                    convertF16toF32(src: p, dst: outBuf.baseAddress!, count: count)
                }
            } else {
                out.withUnsafeMutableBufferPointer { buf in
                    stridedReadF16ToF32(source: arr, dst: buf.baseAddress!, total: count)
                }
            }
            return out
        }
        // Generic fallback.
        assertionFailure(
            "KokoroAneArrays.readFloats: unexpected source dtype \(arr.dataType.rawValue); using slow NSNumber read")
        return (0..<count).map { Float(truncating: arr[$0]) }
    }
}
