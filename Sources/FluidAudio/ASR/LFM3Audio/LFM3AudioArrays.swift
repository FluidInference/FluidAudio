#if arch(arm64)
@preconcurrency import CoreML
import Foundation

/// Preallocated Core ML buffers for the LFM3-Audio runtime: every input and output is a host-owned fp16 / int32
/// `MLMultiArray`, outputs are bound through `MLPredictionOptions.outputBackings`, and predictions are synchronous.
/// The backings avoid per-call allocation. They do not prevent the macOS 27 first-predict abort
/// `MPSGraphTensorData shape.count = 0 != strides.count = 3`: that comes from low free disk during a cold load
/// (see `LFM3AudioConfig.lowDiskBytes`).
enum LFM3AudioArrays {

    static func float16(_ shape: [Int], fill: Float16 = 0) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        array.dataPointer.bindMemory(to: Float16.self, capacity: array.count).initialize(
            repeating: fill, count: array.count)
        return array
    }

    static func int32(_ shape: [Int], fill: Int32 = 0) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .int32)
        array.dataPointer.bindMemory(to: Int32.self, capacity: array.count).initialize(
            repeating: fill, count: array.count)
        return array
    }

    /// Zero-filled arrays for every multi-array input of a model, keyed by input name.
    static func zeroInputs(_ description: MLModelDescription) throws -> [String: MLMultiArray] {
        var out: [String: MLMultiArray] = [:]
        for (name, input) in description.inputDescriptionsByName {
            guard let constraint = input.multiArrayConstraint else { continue }
            let shape = constraint.shape.map { $0.intValue }
            out[name] = constraint.dataType == .int32 ? try int32(shape) : try float16(shape)
        }
        return out
    }

    /// Output backings for every multi-array output of a model, keyed by output name.
    static func outputBackings(_ description: MLModelDescription) throws -> [String: MLMultiArray] {
        var out: [String: MLMultiArray] = [:]
        for (name, output) in description.outputDescriptionsByName {
            guard let constraint = output.multiArrayConstraint else { continue }
            let shape = constraint.shape.map { $0.intValue }
            guard !shape.isEmpty else {
                throw LFM3AudioError.invalidModel("output \(name) has no static shape")
            }
            out[name] = constraint.dataType == .int32 ? try int32(shape) : try float16(shape)
        }
        return out
    }

    static func require(_ arrays: [String: MLMultiArray], _ name: String) throws -> MLMultiArray {
        guard let array = arrays[name] else { throw LFM3AudioError.invalidModel("missing tensor \(name)") }
        return array
    }
}

extension MLMultiArray {
    var lfm3Float16: UnsafeMutablePointer<Float16> { dataPointer.bindMemory(to: Float16.self, capacity: count) }
    var lfm3Int32: UnsafeMutablePointer<Int32> { dataPointer.bindMemory(to: Int32.self, capacity: count) }
}

/// Feature provider over preallocated arrays. The `MLFeatureValue` wrappers are made once per array (Core ML reads
/// the arrays' current contents at prediction time), so a prediction allocates no provider objects.
final class LFM3AudioInputs: NSObject, MLFeatureProvider {
    private var values: [String: MLFeatureValue]
    let featureNames: Set<String>

    init(_ arrays: [String: MLMultiArray]) {
        values = arrays.mapValues { MLFeatureValue(multiArray: $0) }
        featureNames = Set(arrays.keys)
        super.init()
    }

    /// Bind another array to an existing input (state ping-pong, stage-to-stage hidden states).
    func set(_ array: MLMultiArray, for name: String) {
        assert(featureNames.contains(name), "LFM3AudioInputs has no input \(name)")
        guard values[name]?.multiArrayValue !== array else { return }
        values[name] = MLFeatureValue(multiArray: array)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? { values[featureName] }
}

/// Little-endian binary tables shipped next to the models (`assets/*.bin`).
enum LFM3AudioTables {
    static func mapped(_ url: URL, expectedBytes: Int) throws -> Data {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw LFM3AudioError.modelNotFound(url.lastPathComponent)
        }
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        guard data.count == expectedBytes else {
            throw LFM3AudioError.invalidModel(
                "\(url.lastPathComponent) is \(data.count) bytes, expected \(expectedBytes)")
        }
        return data
    }

    static func float32(_ url: URL, count: Int) throws -> [Float] {
        let data = try mapped(url, expectedBytes: count * 4)
        return data.withUnsafeBytes { raw in
            (0..<count).map {
                Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self)))
            }
        }
    }

    static func float16(_ url: URL, count: Int) throws -> [Float16] {
        let data = try mapped(url, expectedBytes: count * 2)
        return data.withUnsafeBytes { raw in
            (0..<count).map {
                Float16(bitPattern: UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)))
            }
        }
    }
}
#endif
