@preconcurrency import CoreML
import Foundation

/// Tiny helpers for moving Float / Int32 buffers in and out of `MLMultiArray`
/// for the Supertonic-3 stages.
///
/// All factories produce row-major contiguous Float32 / Int32 arrays; strides
/// are inferred from the shape. `extractFloats` handles either Float32 or
/// Float16 backing storage so a CoreML graph using FP16 IO is transparent.
public enum Supertonic3MultiArray {

    public static func makeFloat32(_ values: [Float], shape: [Int]) throws -> MLMultiArray {
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw Supertonic3Error.invalidTensorShape(
                stage: "makeFloat32",
                expected: "\(expected) (shape \(shape))",
                got: "\(values.count)")
        }
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let dst = arr.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    public static func makeInt32(_ values: [Int32], shape: [Int]) throws -> MLMultiArray {
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw Supertonic3Error.invalidTensorShape(
                stage: "makeInt32",
                expected: "\(expected) (shape \(shape))",
                got: "\(values.count)")
        }
        let arr = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let dst = arr.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return arr
    }

    /// Read out the contents of an `MLMultiArray` as a Swift `[Float]`,
    /// handling Float32 / Float16 / Double / Int32 backing stores.
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
}
