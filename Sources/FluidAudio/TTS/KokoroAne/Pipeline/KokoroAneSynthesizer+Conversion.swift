import Accelerate
@preconcurrency import CoreML
import Foundation

/// MLMultiArray builders + fp16 ↔ fp32 conversions used by the chain.
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
    /// source MLMultiArray has an unexpected dtype (not fp16/fp32).
    private static func genericCopy(_ source: MLMultiArray, into dst: MLMultiArray, count: Int) {
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
    static func float16Array(shape: [Int], from source: MLMultiArray) throws -> MLMultiArray {
        // Same shape + same dtype → just copy bytes.
        let total = shape.reduce(1, *)
        let nsShape = shape.map { NSNumber(value: $0) }
        let dst = try MLMultiArray(shape: nsShape, dataType: .float16)
        precondition(
            source.count == total,
            "float16Array(from MLMultiArray): source has \(source.count) elements, shape implies \(total)")
        if source.dataType == .float16 {
            memcpy(dst.dataPointer, source.dataPointer, total * MemoryLayout<UInt16>.size)
            return dst
        }
        if source.dataType == .float32 {
            let f32 = source.dataPointer.bindMemory(to: Float.self, capacity: total)
            let dstU16 = dst.dataPointer.bindMemory(to: UInt16.self, capacity: total)
            convertF32toF16(src: f32, dst: dstU16, count: total)
            return dst
        }
        genericCopy(source, into: dst, count: total)
        return dst
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
    static func float32Array(shape: [Int], from source: MLMultiArray) throws -> MLMultiArray {
        let total = shape.reduce(1, *)
        let nsShape = shape.map { NSNumber(value: $0) }
        let dst = try MLMultiArray(shape: nsShape, dataType: .float32)
        precondition(
            source.count == total,
            "float32Array(from MLMultiArray): source has \(source.count) elements, shape implies \(total)")
        if source.dataType == .float32 {
            memcpy(dst.dataPointer, source.dataPointer, total * MemoryLayout<Float>.size)
            return dst
        }
        if source.dataType == .float16 {
            let srcU16 = source.dataPointer.bindMemory(to: UInt16.self, capacity: total)
            let dstF = dst.dataPointer.bindMemory(to: Float.self, capacity: total)
            convertF16toF32(src: srcU16, dst: dstF, count: total)
            return dst
        }
        genericCopy(source, into: dst, count: total)
        return dst
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
        let total = newShape.reduce(1, *)
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
        let total = newShape.reduce(1, *)
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
            arr.count == channels * oldT,
            "shape mismatch: arr.count \(arr.count) ≠ \(channels)*\(oldT)")
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
    static func readFloats(_ arr: MLMultiArray) -> [Float] {
        let count = arr.count
        if arr.dataType == .float32 {
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: count)
            return Array(UnsafeBufferPointer(start: p, count: count))
        }
        if arr.dataType == .float16 {
            let p = arr.dataPointer.bindMemory(to: UInt16.self, capacity: count)
            var out = [Float](repeating: 0, count: count)
            out.withUnsafeMutableBufferPointer { outBuf in
                convertF16toF32(src: p, dst: outBuf.baseAddress!, count: count)
            }
            return out
        }
        // Generic fallback.
        return (0..<count).map { Float(truncating: arr[$0]) }
    }
}
