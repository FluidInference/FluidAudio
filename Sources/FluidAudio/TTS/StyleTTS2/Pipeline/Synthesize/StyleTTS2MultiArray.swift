@preconcurrency import CoreML
import Foundation

/// Tiny helpers for moving Float / Int32 buffers in and out of `MLMultiArray`.
///
/// All factories produce row-major contiguous Float32 arrays — strides are
/// computed from the shape so the StyleTTS2 stages see the layout they were
/// traced with. `extractFloats` reads either Float32 or Float16 backing
/// storage and returns a Swift Float32 buffer.
public enum StyleTTS2MultiArray {

    // MARK: - Factories

    public static func makeFloat32(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        precondition(values.count == shape.reduce(1, *), "shape product must match buffer length")
        let arr = try MLMultiArray(
            shape: shape.map(NSNumber.init), dataType: .float32)
        let dst = arr.dataPointer.bindMemory(
            to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    public static func makeInt32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        precondition(values.count == shape.reduce(1, *), "shape product must match buffer length")
        let arr = try MLMultiArray(
            shape: shape.map(NSNumber.init), dataType: .int32)
        let dst = arr.dataPointer.bindMemory(
            to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    // MARK: - Extraction

    /// Read out the contents of an `MLMultiArray` as a Swift `[Float]`,
    /// handling both `.float32` and `.float16` backing stores.
    public static func extractFloats(_ arr: MLMultiArray) -> [Float] {
        let n = arr.count
        var out = [Float](repeating: 0, count: n)
        switch arr.dataType {
        case .float32:
            let src = arr.dataPointer.bindMemory(to: Float.self, capacity: n)
            out.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: src, count: n)
            }
        case .float16:
            #if arch(arm64)
            let src = arr.dataPointer.bindMemory(to: Float16.self, capacity: n)
            for i in 0..<n {
                out[i] = Float(src[i])
            }
            #else
            // x86 hosts: route through NSNumber (slow but correct).
            for i in 0..<n {
                out[i] = arr[i].floatValue
            }
            #endif
        case .double:
            let src = arr.dataPointer.bindMemory(to: Double.self, capacity: n)
            for i in 0..<n {
                out[i] = Float(src[i])
            }
        case .int32:
            let src = arr.dataPointer.bindMemory(to: Int32.self, capacity: n)
            for i in 0..<n {
                out[i] = Float(src[i])
            }
        @unknown default:
            for i in 0..<n {
                out[i] = arr[i].floatValue
            }
        }
        return out
    }

    /// Convenience: produce a flat `[Float]` of length `S0*S1*S2…` plus the
    /// shape as `[Int]` for ergonomic debugging.
    public static func shape(of arr: MLMultiArray) -> [Int] {
        return arr.shape.map { $0.intValue }
    }
}
