@preconcurrency import CoreML
import Foundation

/// `MLMultiArray` plumbing for the MOSS-TTS-Nano graphs.
enum MossTtsNanoTensor {

    static func int32(_ values: [Int32], shape: [Int], stage: String) throws -> MLMultiArray {
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw MossTtsNanoError.invalidTensorShape(
                stage: stage, expected: "\(expected) (shape \(shape))", got: "\(values.count)")
        }
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .int32)
        let dst = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return array
    }

    static func float32(_ values: [Float], shape: [Int], stage: String) throws -> MLMultiArray {
        let expected = shape.reduce(1, *)
        guard values.count == expected else {
            throw MossTtsNanoError.invalidTensorShape(
                stage: stage, expected: "\(expected) (shape \(shape))", got: "\(values.count)")
        }
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float32)
        let dst = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        values.withUnsafeBufferPointer { src in
            dst.update(from: src.baseAddress!, count: values.count)
        }
        return array
    }

    static func zeros(shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let count = shape.reduce(1) { $0 * $1.intValue }
        array.dataPointer.bindMemory(to: Float.self, capacity: count).update(repeating: 0, count: count)
        return array
    }

    static func floats(_ array: MLMultiArray) -> [Float] {
        let n = array.count
        var out = [Float](repeating: 0, count: n)
        switch array.dataType {
        case .float32:
            let src = array.dataPointer.bindMemory(to: Float.self, capacity: n)
            out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: src, count: n) }
        case .float16:
            #if arch(arm64)
            let src = array.dataPointer.bindMemory(to: Float16.self, capacity: n)
            for i in 0..<n { out[i] = Float(src[i]) }
            #else
            for i in 0..<n { out[i] = array[i].floatValue }
            #endif
        case .double:
            let src = array.dataPointer.bindMemory(to: Double.self, capacity: n)
            for i in 0..<n { out[i] = Float(src[i]) }
        case .int32:
            let src = array.dataPointer.bindMemory(to: Int32.self, capacity: n)
            for i in 0..<n { out[i] = Float(src[i]) }
        @unknown default:
            for i in 0..<n { out[i] = array[i].floatValue }
        }
        return out
    }

    static func ints(_ array: MLMultiArray) -> [Int32] {
        let n = array.count
        if array.dataType == .int32 {
            let src = array.dataPointer.bindMemory(to: Int32.self, capacity: n)
            return Array(UnsafeBufferPointer(start: src, count: n))
        }
        return floats(array).map { Int32($0.rounded()) }
    }

    static func output(_ provider: MLFeatureProvider, _ name: String, stage: String) throws -> MLMultiArray {
        guard let value = provider.featureValue(for: name)?.multiArrayValue else {
            throw MossTtsNanoError.inferenceFailed(stage: stage, underlying: "missing '\(name)' output")
        }
        return value
    }

    static func predict(_ model: MLModel, _ inputs: [String: MLMultiArray], stage: String) throws -> MLFeatureProvider {
        let features = inputs.mapValues { MLFeatureValue(multiArray: $0) }
        do {
            let provider = try MLDictionaryFeatureProvider(dictionary: features)
            return try model.prediction(from: provider)
        } catch let error as MossTtsNanoError {
            throw error
        } catch {
            throw MossTtsNanoError.inferenceFailed(stage: stage, underlying: "\(error)")
        }
    }
}

/// SplitMix64 — deterministic uniform source for the in-graph sampler.
struct MossTtsNanoRandom: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { state = seed }

    /// System-seeded instance.
    init() { state = UInt64.random(in: 0...UInt64.max) }

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in [0, 1) with 24 bits of resolution.
    mutating func uniform() -> Float {
        Float(next() >> 40) / Float(1 << 24)
    }
}
