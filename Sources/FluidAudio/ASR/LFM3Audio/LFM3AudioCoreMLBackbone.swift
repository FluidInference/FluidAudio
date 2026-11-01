#if arch(arm64)
@preconcurrency import CoreML
import Foundation

/// One backbone call: `q` rows starting at cache slot `slotStart`; the first `nPad` rows are zero pads.
struct LFM3AudioChunk: Equatable, Sendable {
    let slotStart: Int
    let q: Int
    let nPad: Int
}

enum LFM3AudioPrefill {
    /// Chunk plan for `n` rows over the prefill buckets (descending). On a fresh state with `padFirst`, a
    /// zero-padded first chunk absorbs the remainder so the rest are full largest-bucket calls (one padded q256
    /// call is cheaper than a q64 + q16 + q13 ladder); otherwise descending buckets, then 1-row decode calls.
    static func plan(n: Int, buckets: [Int], padFirst: Bool, startSlot: Int = 0) -> [LFM3AudioChunk] {
        let sizes = Array(Set(buckets)).sorted(by: >)
        guard let big = sizes.first, n > 0 else { return [] }
        var out: [LFM3AudioChunk] = []
        var slot = startSlot
        if padFirst && startSlot == 0 {
            let remainder = n % big
            if remainder > 0 {
                let fit = sizes.filter { $0 >= remainder }.min() ?? big
                out.append(LFM3AudioChunk(slotStart: slot, q: fit, nPad: fit - remainder))
                slot += fit
            }
            for _ in 0..<(n / big) {
                out.append(LFM3AudioChunk(slotStart: slot, q: big, nPad: 0))
                slot += big
            }
            return out
        }
        var left = n
        for size in sizes {
            while left >= size {
                out.append(LFM3AudioChunk(slotStart: slot, q: size, nPad: 0))
                slot += size
                left -= size
            }
        }
        for _ in 0..<left {
            out.append(LFM3AudioChunk(slotStart: slot, q: 1, nPad: 0))
            slot += 1
        }
        return out
    }
}

/// Zero-tail RoPE of one attention type: `pairs` rotated frequency pairs, the rest have inv_freq 0.
struct LFM3AudioRope: Sendable {
    let headDim: Int
    let invFreq: [Float]

    init(theta: Double, pairs: Int, headDim: Int) {
        self.headDim = headDim
        invFreq = (0..<(headDim / 2)).map { i in
            i < pairs ? 1.0 / powf(Float(theta), Float(2 * i) / Float(headDim)) : 0
        }
    }

    /// cos/sin rows for positions `pos0 ..< pos0 + q` into `[q, headDim]` fp16 buffers.
    func fill(cos: UnsafeMutablePointer<Float16>, sin: UnsafeMutablePointer<Float16>, pos0: Int, q: Int) {
        let half = headDim / 2
        invFreq.withUnsafeBufferPointer { freq in
            for r in 0..<q {
                let p = Float(pos0 + r)
                let base = r * headDim
                for i in 0..<half {
                    let a = p * freq[i]
                    let c = Float16(Foundation.cos(a))
                    let s = Float16(Foundation.sin(a))
                    cos[base + i] = c
                    cos[base + half + i] = c
                    sin[base + i] = s
                    sin[base + half + i] = s
                }
            }
        }
    }
}

enum LFM3AudioMasks {
    static let negative: Float16 = -30000

    /// `[q, span]` additive mask over full-cache slots for rows at `slotStart ..< slotStart + q`:
    /// key <= query and the same pad class (slot >= nPad).
    static func full(_ m: UnsafeMutablePointer<Float16>, slotStart: Int, q: Int, span: Int, nPad: Int) {
        for r in 0..<q {
            let qs = slotStart + r
            let row = m + r * span
            let lo = (nPad > 0 && qs >= nPad) ? nPad : 0
            let hi = min(qs, span - 1)
            if lo > 0 { row.update(repeating: negative, count: min(lo, span)) }
            if hi >= lo { (row + lo).update(repeating: 0, count: hi - lo + 1) }
            if hi + 1 < span { (row + hi + 1).update(repeating: negative, count: span - hi - 1) }
        }
    }

    /// `[q, fifo + q]` additive mask over cat(fifo, new); fifo entry j holds slot `slotStart - fifo + j`.
    static func sliding(
        _ m: UnsafeMutablePointer<Float16>, slotStart: Int, q: Int, window: Int, fifo: Int, nPad: Int
    ) {
        let width = fifo + q
        for r in 0..<q {
            let qs = slotStart + r
            let row = m + r * width
            for j in 0..<width {
                let k = j < fifo ? slotStart - fifo + j : slotStart + (j - fifo)
                var allowed = k >= 0 && k <= qs && k > qs - window
                if allowed && nPad > 0 { allowed = (k >= nPad) == (qs >= nPad) }
                row[j] = allowed ? 0 : negative
            }
        }
    }
}

/// A backbone stage function: loaded with the pack, or on the first call of its bucket
/// (`LFM3AudioConfig.deferredPrefillRows`).
enum LFM3AudioStageModel: Sendable {
    case loaded(MLModel)
    case deferred(@Sendable () throws -> MLModel)

    var isDeferred: Bool {
        if case .deferred = self { return true }
        return false
    }

    func load() throws -> MLModel {
        switch self {
        case .loaded(let model): return model
        case .deferred(let load): return try load()
        }
    }
}

/// Core ML LFM3 backbone pack: two stateful multifunction stages (A: layers 0-11, B: 12-23; functions
/// `decode_q1`, `prefill_q256/64/16/13`) driven at the embedding level, plus a stateless chunked top-k head.
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
actor LFM3AudioCoreMLBackbone: LFM3AudioLanguageModel {

    private final class Stage {
        let model: MLModel
        let inputs: LFM3AudioInputs
        let options = MLPredictionOptions()
        let outputs: [String: MLMultiArray]

        init(model: MLModel, shared: [String: MLMultiArray]) throws {
            self.model = model
            var own: [String: MLMultiArray] = [:]
            for name in model.modelDescription.inputDescriptionsByName.keys { own[name] = shared[name] }
            inputs = LFM3AudioInputs(own)
            outputs = try LFM3AudioArrays.outputBackings(model.modelDescription)
            options.outputBackings = outputs
        }
    }

    /// One decode / prefill bucket across the stages; a deferred bucket loads its stage models on first use.
    private final class Function {
        let q: Int
        let span: Int
        private(set) var stages: [Stage] = []
        private(set) var shared: [String: MLMultiArray] = [:]
        private var pending: [LFM3AudioStageModel]?

        init(q: Int, span: Int, sources: [LFM3AudioStageModel]) throws {
            self.q = q
            self.span = span
            if sources.contains(where: \.isDeferred) {
                pending = sources
            } else {
                try build(sources)
            }
        }

        /// Loads the stage models of a deferred bucket; no-op once loaded.
        func load() throws {
            guard let sources = pending else { return }
            try build(sources)
            pending = nil
        }

        private func build(_ sources: [LFM3AudioStageModel]) throws {
            let models = try sources.map { try $0.load() }
            var shared: [String: MLMultiArray] = [:]
            for model in models {
                for (name, array) in try LFM3AudioArrays.zeroInputs(model.modelDescription) where shared[name] == nil {
                    shared[name] = array
                }
            }
            stages = try models.map { try Stage(model: $0, shared: shared) }
            self.shared = shared
        }
    }

    private let config: LFM3AudioPackConfig
    private let embedding: LFM3AudioEmbeddingTable
    private let ropeFull: LFM3AudioRope
    private let ropeSliding: LFM3AudioRope
    private let functions: [Int: [Function]]  // rows per call -> functions by ascending span
    private let buckets: [Int]
    private let head: MLModel
    private let headInputs: LFM3AudioInputs
    private let headOptions = MLPredictionOptions()
    private let headOutputs: [String: MLMultiArray]
    private let headHidden: MLMultiArray
    private var states: [MLState] = []
    private var slot = 0
    private var nPad = 0

    /// - Parameter stageModels: `"<stage index>:<function>"` -> stage model (loaded, or deferred to first use).
    init(
        config: LFM3AudioPackConfig, embedding: LFM3AudioEmbeddingTable, stageModels: [String: LFM3AudioStageModel],
        head: MLModel
    ) throws {
        guard let full = config.rope["full"], let sliding = config.rope["sliding"] else {
            throw LFM3AudioError.invalidModel("pack config has no RoPE tables")
        }
        self.config = config
        self.embedding = embedding
        ropeFull = LFM3AudioRope(theta: full.theta, pairs: full.pairs, headDim: config.headDim)
        ropeSliding = LFM3AudioRope(theta: sliding.theta, pairs: sliding.pairs, headDim: config.headDim)
        var functions: [Int: [Function]] = [:]
        for plan in config.callPlans {
            var sources: [LFM3AudioStageModel] = []
            for (index, name) in plan.calls.enumerated() {
                guard let source = stageModels["\(index):\(name)"] else {
                    throw LFM3AudioError.modelNotFound("backbone stage \(index) function \(name)")
                }
                sources.append(source)
            }
            functions[plan.q, default: []].append(try Function(q: plan.q, span: plan.span, sources: sources))
        }
        guard functions[1] != nil else { throw LFM3AudioError.invalidModel("pack has no decode function") }
        self.functions = functions.mapValues { $0.sorted { $0.span < $1.span } }
        buckets = Array(Set(config.prefill.buckets)).sorted(by: >)
        self.head = head
        var headArrays = try LFM3AudioArrays.zeroInputs(head.modelDescription)
        headHidden = try LFM3AudioArrays.require(headArrays, "h")
        headArrays["rp"] = try LFM3AudioArrays.float16([1, 1, config.vocabSize])
        headInputs = LFM3AudioInputs(headArrays)
        headOutputs = try LFM3AudioArrays.outputBackings(head.modelDescription)
        headOptions.outputBackings = headOutputs
        states = try Self.makeStates(self.functions)
    }

    private static func makeStates(_ functions: [Int: [Function]]) throws -> [MLState] {
        guard let decode = functions[1]?.first else { throw LFM3AudioError.invalidModel("no decode function") }
        try decode.load()
        return decode.stages.map { $0.model.makeState() }
    }

    func prepare(rows: Int) throws {
        for function in functions[rows] ?? [] { try function.load() }
    }

    func reset() throws {
        states = try Self.makeStates(functions)
        slot = 0
        nPad = 0
    }

    func feed(_ rows: LFM3AudioRows, want: LFM3AudioWant) throws -> LFM3AudioFeedResult {
        let h = config.hiddenSize
        let n = rows.count(hidden: h)
        guard n > 0 else { throw LFM3AudioError.processingFailed("empty feed") }
        let plan = LFM3AudioPrefill.plan(
            n: n, buckets: buckets, padFirst: config.prefill.padFirst && slot == 0, startSlot: slot)
        var consumed = 0
        var cursor = 0
        var last: MLMultiArray?
        var lastQ = 1
        for chunk in plan {
            let real = chunk.q - chunk.nPad
            let lo = consumed
            last = try call(chunk) { x in
                try rows.write(lo, real, into: x, embedding: embedding, rowCursor: &cursor)
            }
            lastQ = chunk.q
            consumed += real
        }
        guard let output = last else { throw LFM3AudioError.processingFailed("no backbone call") }
        let hidden = output.lfm3Float16 + (lastQ - 1) * h
        switch want {
        case .hidden:
            return LFM3AudioFeedResult(token: -1, hidden: Array(UnsafeBufferPointer(start: hidden, count: h)))
        case .token(let suppress):
            return LFM3AudioFeedResult(token: try greedy(hidden, suppress: suppress))
        }
    }

    // MARK: - Private

    private func function(q: Int) throws -> Function {
        let needed = slot + q
        guard let match = functions[q]?.first(where: { $0.span >= needed }) else {
            throw LFM3AudioError.contextFull(needed: needed, available: functions[q]?.last?.span ?? 0)
        }
        try match.load()
        return match
    }

    /// Runs one chunk whose real rows `fill` writes into `x` (after `nPad` zero rows); returns stage-B `h`.
    private func call(
        _ chunk: LFM3AudioChunk, fill: (UnsafeMutablePointer<Float16>) throws -> Void
    ) throws -> MLMultiArray {
        let fn = try function(q: chunk.q)
        let shared = fn.shared
        let h = config.hiddenSize
        if chunk.nPad > 0 {
            guard chunk.slotStart == 0 else { throw LFM3AudioError.processingFailed("padding after slot 0") }
            nPad = chunk.nPad
        }
        let x = try LFM3AudioArrays.require(shared, "x").lfm3Float16
        if chunk.nPad > 0 { x.update(repeating: 0, count: chunk.nPad * h) }
        try fill(x + chunk.nPad * h)
        let pos0 = chunk.slotStart - nPad
        if let c = shared["cos_g"], let s = shared["sin_g"] {
            ropeFull.fill(cos: c.lfm3Float16, sin: s.lfm3Float16, pos0: pos0, q: chunk.q)
        }
        if let c = shared["cos_s"], let s = shared["sin_s"] {
            ropeSliding.fill(cos: c.lfm3Float16, sin: s.lfm3Float16, pos0: pos0, q: chunk.q)
        }
        if let m = shared["mask_g"] {
            LFM3AudioMasks.full(
                m.lfm3Float16, slotStart: chunk.slotStart, q: chunk.q, span: m.shape[3].intValue, nPad: nPad)
        }
        if let m = shared["mask_s"] {
            LFM3AudioMasks.sliding(
                m.lfm3Float16, slotStart: chunk.slotStart, q: chunk.q, window: config.slidingWindow,
                fifo: config.fifoLen, nPad: nPad)
        }
        if let w = shared["write_pos"] { w.lfm3Int32[0] = Int32(chunk.slotStart) }
        if let oneHot = shared["onehot"] {
            let p = oneHot.lfm3Float16
            p.update(repeating: 0, count: oneHot.count)
            p[chunk.slotStart] = 1
        }
        for (index, stage) in fn.stages.enumerated() {
            if index > 0 { stage.inputs.set(try LFM3AudioArrays.require(fn.stages[index - 1].outputs, "h"), for: "x") }
            _ = try stage.model.prediction(from: stage.inputs, using: states[index], options: stage.options)
        }
        slot = chunk.slotStart + chunk.q
        guard let lastStage = fn.stages.last else { throw LFM3AudioError.invalidModel("function has no stages") }
        return try LFM3AudioArrays.require(lastStage.outputs, "h")
    }

    /// Greedy token from the merged per-chunk top-k of the head (suppressed ids removed on the host).
    private func greedy(_ hidden: UnsafePointer<Float16>, suppress: Set<Int32>) throws -> Int32 {
        headHidden.lfm3Float16.update(from: hidden, count: config.hiddenSize)
        _ = try head.prediction(from: headInputs, options: headOptions)
        let values = try LFM3AudioArrays.require(headOutputs, "topk_v")
        let ids = try LFM3AudioArrays.require(headOutputs, "topk_i")
        let chunks = values.shape.count == 3 ? values.shape[1].intValue : 1
        let k = values.count / chunks
        let chunkSize = config.topk?.chunkSize ?? config.vocabSize
        let vp = values.lfm3Float16
        let ip = ids.lfm3Int32
        var bestValue = -Float.infinity
        var bestId: Int32 = -1
        for c in 0..<chunks {
            for j in 0..<k {
                let id = ip[c * k + j] + Int32(c * chunkSize)
                let value = Float(vp[c * k + j])
                if value > bestValue && !suppress.contains(id) {
                    bestValue = value
                    bestId = id
                }
            }
        }
        guard bestId >= 0 else { throw LFM3AudioError.processingFailed("head returned no candidate") }
        return bestId
    }
}
#endif
