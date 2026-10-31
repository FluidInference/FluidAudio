#if TTS && canImport(CoreAI) && arch(arm64)
import CoreAI
import Foundation

/// MaskGIT depth decoder on Core AI: the fused `depth_frame` function (3 passes + presence penalty, temperature,
/// top-k 128 and Gumbel-max in the graph, GPU) — one call per speech frame instead of `depth_in` + 3 x `depth_pass`
/// + host sampling. 99.4% of codes identical to HF on 319 golden frames.
@available(macOS 27, iOS 27, tvOS 27, visionOS 27, *)
actor LFM3AudioCoreAIDepthDecoder: LFM3AudioDepthModel {
    // Core AI types are stored as `@CoreAIStorage`; see `CoreAIStorage`.
    @CoreAIStorage private var model: AIModel
    @CoreAIStorage private var function: InferenceFunction
    @CoreAIStorage private var descriptors: [String: NDArrayDescriptor]
    private let params: [Float16]

    init(bundle: URL, sampling: LFM3AudioSpeechSampling) async throws {
        let model = try await AIModel(
            contentsOf: bundle, options: SpecializationOptions(preferredComputeUnitKind: .gpu))
        guard let function = try model.loadFunction(named: "depth_frame"),
            let descriptor = model.functionDescriptor(for: "depth_frame")
        else {
            throw LFM3AudioError.invalidModel("Core AI depth bundle has no `depth_frame` (\(model.functionNames))")
        }
        var descriptors: [String: NDArrayDescriptor] = [:]
        for name in ["h", "sched", "noise", "present", "params"] {
            guard case .ndArray(let d) = descriptor.inputDescriptor(of: name) else {
                throw LFM3AudioError.invalidModel("Core AI depth_frame input \(name)")
            }
            descriptors[name] = d
        }
        guard case .ndArray(let codes) = descriptor.outputDescriptor(of: "codes") else {
            throw LFM3AudioError.invalidModel("Core AI depth_frame output codes")
        }
        descriptors["codes"] = codes
        self.model = model
        self.function = function
        self.descriptors = descriptors
        // (1 / temperature, presence penalty, use top-k, noise scale)
        params = [
            Float16(1 / sampling.temperature), Float16(sampling.penalty), sampling.topK > 0 ? 1 : 0, 1,
        ]
    }

    func frame(hidden: [Float16], frame: LFM3AudioDepthFrame, present: [Float16]) async throws -> [Int32] {
        let inputs: [String: NDArray] = [
            "h": try array("h", hidden),
            "sched": try array("sched", frame.schedule),
            "noise": try array("noise", frame.noise),
            "present": try array("present", present),
            "params": try array("params", params),
        ]
        guard let descriptor = descriptors["codes"] else { throw LFM3AudioError.invalidModel("codes") }
        var out = NDArray(descriptor: descriptor)
        var outputs = InferenceFunction.MutableViews()
        outputs.insert(&out, for: "codes")
        _ = try await function.run(inputs: inputs, outputViews: outputs)
        let count = LFM3AudioSpeechConfig.codebooks
        return out.view(as: Int32.self).withUnsafePointer { p, _, _ in
            Array(UnsafeBufferPointer(start: p, count: count))
        }
    }

    /// fp16 input from row-major `values`; Core AI pads rows to 64 bytes, so copy row by row with the strides.
    private func array(_ name: String, _ values: [Float16]) throws -> NDArray {
        guard let descriptor = descriptors[name] else { throw LFM3AudioError.invalidModel("input \(name)") }
        var array = NDArray(descriptor: descriptor)
        let shape = array.shape
        guard shape.reduce(1, *) == values.count, shape.count <= 2 else {
            throw LFM3AudioError.processingFailed("depth_frame \(name): \(values.count) values for shape \(shape)")
        }
        let columns = shape.last ?? 1
        let rows = values.count / max(columns, 1)
        let rowStride = array.strides.count == 2 ? array.strides[0] : columns
        values.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            array.mutableView(as: Float16.self).withUnsafeMutablePointer { dst, _, _ in
                for r in 0..<rows { (dst + r * rowStride).update(from: base + r * columns, count: columns) }
            }
        }
        return array
    }
}
#endif
