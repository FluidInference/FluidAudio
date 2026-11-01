#if canImport(CoreAI) && arch(arm64)
import Accelerate
import CoreAI
import Foundation

/// Core AI LFM3 backbone: the `inputs_embeds` LM bundle with entrypoints `prefill` (state update only), `main`
/// (logits of the last row) and `hidden` (final-norm hidden of the last row), specialized for the GPU with
/// frequent reshapes (as `llm-runner`). Rows `0 ..< n-1` of a feed go through `prefill` in chunks, the last row
/// through `main` or `hidden`.
///
/// Each entrypoint is loaded from its own `AIModel` (all bound to the same host-owned states). With one shared
/// `AIModel`, macOS 27.0 aborts in `GPUMemrefOps` (`Failed to resolve dynamic dimensions for memref.alloc`) when a
/// function runs twice at consecutive positions with the same row count and another function is then called with
/// exactly the next shape — e.g. the first `main` after a run of `hidden` speech frames in TTS.
@available(macOS 27, iOS 27, tvOS 27, visionOS 27, *)
actor LFM3AudioCoreAIBackbone: LFM3AudioLanguageModel {
    private static let stateNames = ["keyCache", "valueCache", "convStates"]

    private let models: [AIModel]
    private let prefill: InferenceFunction
    private let main: InferenceFunction
    private let hiddenFunction: InferenceFunction
    private let embeddingsInput: NDArrayDescriptor
    private let positionsInput: NDArrayDescriptor
    private let positionsName: String
    private let logitsOutput: NDArrayDescriptor
    private let hiddenOutput: NDArrayDescriptor
    private let embedding: LFM3AudioEmbeddingTable
    private let hiddenSize: Int
    private let vocab: Int
    private let chunk: Int
    private let cacheLength: Int
    // Optional so a call can move the states out (unique, so mutable views never copy the KV cache).
    private var keyCache: NDArray?
    private var valueCache: NDArray?
    private var convStates: NDArray?
    private var offset = 0
    private var logits: [Float]

    init(
        bundle: URL, embedding: LFM3AudioEmbeddingTable, cacheLength: Int = 4096, chunk: Int = 512
    ) async throws {
        var options = SpecializationOptions(preferredComputeUnitKind: .gpu)
        options.expectFrequentReshapes = true
        let model = try await AIModel(contentsOf: bundle, options: options)
        let prefillModel = try await AIModel(contentsOf: bundle, options: options)
        let hiddenModel = try await AIModel(contentsOf: bundle, options: options)
        guard let descriptor = model.functionDescriptor(for: "main"), descriptor.inputNames.count >= 2,
            descriptor.inputNames[0] == "inputs_embeds",
            case .ndArray(let embeds) = descriptor.inputDescriptor(of: descriptor.inputNames[0]),
            case .ndArray(let positions) = descriptor.inputDescriptor(of: descriptor.inputNames[1]),
            case .ndArray(let logits) = descriptor.outputDescriptor(of: "logits"),
            let hiddenDescriptor = model.functionDescriptor(for: "hidden"),
            case .ndArray(let hidden) = hiddenDescriptor.outputDescriptor(of: "hidden"),
            Set(descriptor.stateNames) == Set(Self.stateNames)
        else {
            throw LFM3AudioError.invalidModel("unexpected Core AI LM bundle contract")
        }
        let prefill = try Self.function(prefillModel, "prefill")
        let main = try Self.function(model, "main")
        let hiddenFunction = try Self.function(hiddenModel, "hidden")
        let keyCache = try Self.state(descriptor, "keyCache", cacheLength: cacheLength)
        let valueCache = try Self.state(descriptor, "valueCache", cacheLength: cacheLength)
        let convStates = try Self.state(descriptor, "convStates", cacheLength: cacheLength)
        models = [model, prefillModel, hiddenModel]
        self.prefill = prefill
        self.main = main
        self.hiddenFunction = hiddenFunction
        embeddingsInput = embeds
        positionsInput = positions
        positionsName = descriptor.inputNames[1]
        logitsOutput = logits
        hiddenOutput = hidden
        self.embedding = embedding
        hiddenSize = embeds.shape.last ?? LFM3AudioConfig.hiddenSize
        vocab = logits.shape.last ?? embedding.vocab
        self.chunk = chunk
        self.cacheLength = cacheLength
        self.keyCache = keyCache
        self.valueCache = valueCache
        self.convStates = convStates
        self.logits = [Float](repeating: 0, count: logits.shape.last ?? embedding.vocab)
        guard hiddenSize == embedding.hidden else {
            throw LFM3AudioError.invalidModel("Core AI hidden size \(hiddenSize) != embedding \(embedding.hidden)")
        }
    }

    /// Core AI loads every entrypoint with the bundle.
    func prepare(rows: Int) {}

    func reset() throws {
        guard var key = keyCache.take(), var value = valueCache.take(), var conv = convStates.take() else {
            throw LFM3AudioError.processingFailed("Core AI states in use")
        }
        Self.zero(&key)
        Self.zero(&value)
        Self.zero(&conv)
        keyCache = key
        valueCache = value
        convStates = conv
        offset = 0
    }

    func feed(_ rows: LFM3AudioRows, want: LFM3AudioWant) async throws -> LFM3AudioFeedResult {
        let h = hiddenSize
        let n = rows.count(hidden: h)
        guard n > 0 else { throw LFM3AudioError.processingFailed("empty feed") }
        guard offset + n <= cacheLength else {
            throw LFM3AudioError.contextFull(needed: offset + n, available: cacheLength)
        }
        var x = [Float16](repeating: 0, count: n * h)
        var cursor = 0
        try x.withUnsafeMutableBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            try rows.write(0, n, into: base, embedding: embedding, rowCursor: &cursor)
        }
        var start = 0
        while start < n - 1 {
            let end = min(start + chunk, n - 1)
            _ = try await run(prefill, x, rows: start..<end, output: nil)
            start = end
        }
        switch want {
        case .hidden:
            guard let out = try await run(hiddenFunction, x, rows: (n - 1)..<n, output: ("hidden", hiddenOutput))
            else { throw LFM3AudioError.processingFailed("Core AI hidden output") }
            var hidden = [Float16](repeating: 0, count: h)
            out.view(as: Float16.self).withUnsafePointer { p, _, _ in
                hidden.withUnsafeMutableBufferPointer { dst in
                    guard let d = dst.baseAddress else { return }
                    d.update(from: p, count: h)
                }
            }
            return LFM3AudioFeedResult(token: -1, hidden: hidden)
        case .token(let suppress):
            guard let out = try await run(main, x, rows: (n - 1)..<n, output: ("logits", logitsOutput)) else {
                throw LFM3AudioError.processingFailed("Core AI logits output")
            }
            let v = vocab
            out.view(as: Float16.self).withUnsafePointer { p, _, _ in
                logits.withUnsafeMutableBufferPointer { dst in
                    for i in 0..<v { dst[i] = Float(p[i]) }
                }
            }
            for id in suppress where Int(id) < v { logits[Int(id)] = -.infinity }
            var best: Float = 0
            var index: vDSP_Length = 0
            vDSP_maxvi(logits, 1, &best, &index, vDSP_Length(v))
            return LFM3AudioFeedResult(token: Int32(index))
        }
    }

    // MARK: - Private

    /// One call over `rows` of `x` at the current offset; position ids cover `0 ..< offset + rows.count`.
    private func run(
        _ function: InferenceFunction, _ x: [Float16], rows: Range<Int>, output: (name: String, NDArrayDescriptor)?
    ) async throws -> NDArray? {
        let n = rows.count
        let h = hiddenSize
        var embeds = NDArray(descriptor: embeddingsInput.resolvingDynamicDimensions([1, n, h]))
        var embedsView = embeds.mutableView(as: Float16.self)
        x.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            embedsView.copyElements(
                fromContentsOf: UnsafeBufferPointer(start: base + rows.lowerBound * h, count: n * h))
        }
        var positions = NDArray(descriptor: positionsInput.resolvingDynamicDimensions([1, offset + n]))
        var positionsView = positions.mutableView(as: Int32.self)
        positionsView.copyElements(fromContentsOf: (0..<(offset + n)).map { Int32($0) })
        let inputs: [String: NDArray] = ["inputs_embeds": embeds, positionsName: positions]

        guard var key = keyCache.take(), var value = valueCache.take(), var conv = convStates.take() else {
            throw LFM3AudioError.processingFailed("Core AI states in use")
        }
        defer {
            keyCache = key
            valueCache = value
            convStates = conv
        }
        var states = InferenceFunction.MutableViews()
        states.insert(&key, for: "keyCache")
        states.insert(&value, for: "valueCache")
        states.insert(&conv, for: "convStates")
        var result: NDArray?
        if let (name, descriptor) = output {
            var out = NDArray(descriptor: descriptor.resolvingDynamicDimensions([1, n, descriptor.shape.last ?? h]))
            var outputs = InferenceFunction.MutableViews()
            outputs.insert(&out, for: name)
            _ = try await function.run(inputs: inputs, states: states, outputViews: outputs)
            result = out
        } else {
            _ = try await function.run(inputs: inputs, states: states)
        }
        offset += n
        return result
    }

    private static func function(_ model: AIModel, _ name: String) throws -> InferenceFunction {
        guard let function = try model.loadFunction(named: name) else {
            throw LFM3AudioError.invalidModel("Core AI bundle has no `\(name)` function (\(model.functionNames))")
        }
        return function
    }

    /// Zeroed state array with the dynamic (context) dimension resolved to `cacheLength`.
    private static func state(
        _ descriptor: InferenceFunctionDescriptor, _ name: String, cacheLength: Int
    ) throws -> NDArray {
        guard case .ndArray(let d) = descriptor.stateDescriptor(of: name) else {
            throw LFM3AudioError.invalidModel("Core AI state \(name)")
        }
        var array = NDArray(descriptor: d.resolvingDynamicDimensions(d.shape.map { $0 < 0 ? cacheLength : $0 }))
        zero(&array)
        return array
    }

    /// Zero the whole buffer, row padding included (NDArray rows are padded to 64 bytes).
    private static func zero(_ array: inout NDArray) {
        let width: Int
        switch array.scalarType {
        case .float32, .int32: width = 4
        default: width = 2
        }
        array.mutableRawView().withUnsafeMutableBytes { pointer, shape, strides in
            let elements = shape.count > 0 && strides.count > 0 ? shape[0] * strides[0] : 0
            _ = memset(pointer, 0, max(elements, 1) * width)
        }
    }
}
#endif
