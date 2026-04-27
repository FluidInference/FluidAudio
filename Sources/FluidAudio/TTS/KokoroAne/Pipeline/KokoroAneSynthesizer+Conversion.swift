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
