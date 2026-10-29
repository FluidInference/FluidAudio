#if arch(arm64)
@preconcurrency import CoreML
import Foundation

/// One call of the streaming encoder graph: 8 context frames + 8N new mel frames -> N LM rows.
struct LFM3AudioEncoderCall: Sendable {
    var mel: [Float]  // [128, 8N + 8]
    var m1: [Float]  // [4N + 4] stage-1 valid mask
    var m2: [Float]  // [2N + 2] stage-2 valid mask
    var mask: [Float]  // [N, 31 + N] additive sliding-window mask
    var outputRows: Int
}

/// Host side of the causal audio encoder: emits complete stride-8 windows (N tokens per call) and, on the final
/// call, `(frames + 7) / 8` tokens (the HF +7 flush). Exact against HF `encoder_stream`.
struct LFM3AudioEncoderPlanner {
    let tokensPerCall: Int
    private let fifo = 31
    private let window = 32
    private var pending = LFM3AudioMelBuffer()
    private var context: [Float]?  // [128, 8] frames of the previous token
    private(set) var tokens = 0
    private(set) var finished = false

    init(tokensPerCall: Int) { self.tokensPerCall = tokensPerCall }

    /// Additive `[q, fifo + q]` mask over a fifo of `fifo` previous rows + `q` new rows (sliding `window`).
    static func fifoMask(previous: Int, q: Int, window: Int, fifo: Int) -> [Float] {
        let start = min(previous, fifo)
        var m = [Float](repeating: -30000, count: q * (fifo + q))
        for r in 0..<q {
            let query = start + r
            for j in 0..<(fifo + q) {
                let key = j < fifo ? start - fifo + j : start + (j - fifo)
                if key >= 0 && key <= query && key > query - window { m[r * (fifo + q) + j] = 0 }
            }
        }
        return m
    }

    private func stageMasks(first: Bool, length: Int) -> ([Float], [Float]) {
        let n = tokensPerCall
        let offset = first ? 8 : 0
        var out: [[Float]] = []
        for (shift, size) in [(1, 4 * n + 4), (2, 2 * n + 2)] {
            let lo = offset >> shift
            let hi = lo + (length + (1 << shift) - 1) / (1 << shift)
            var m = [Float](repeating: 0, count: size)
            if lo < min(hi, size) { for i in lo..<min(hi, size) { m[i] = 1 } }
            out.append(m)
        }
        return (out[0], out[1])
    }

    private mutating func makeCall(_ new: LFM3AudioMelBuffer, final: Bool) -> LFM3AudioEncoderCall {
        let r = new.frames
        let first = context == nil
        let width = 8 * tokensPerCall + 8
        var mel = [Float](repeating: 0, count: 128 * width)
        for m in 0..<128 {
            if let c = context { for t in 0..<8 { mel[m * width + t] = c[m * 8 + t] } }
            for t in 0..<r { mel[m * width + 8 + t] = new.data[m * r + t] }
        }
        let (m1, m2) = stageMasks(first: first, length: first ? r : 8 + r)
        let outputRows = final ? (r + 7) / 8 : r / 8
        let call = LFM3AudioEncoderCall(
            mel: mel, m1: m1, m2: m2,
            mask: Self.fifoMask(previous: tokens, q: tokensPerCall, window: window, fifo: fifo),
            outputRows: outputRows)
        if outputRows > 0 {
            // next context = frames of the last emitted token ([context | new] coordinates)
            var next = [Float](repeating: 0, count: 128 * 8)
            let base = 8 + 8 * (outputRows - 1)
            for m in 0..<128 {
                for t in 0..<8 {
                    let col = base + t
                    if col < 8 {
                        next[m * 8 + t] = context?[m * 8 + col] ?? 0
                    } else if col - 8 < r {
                        next[m * 8 + t] = new.data[m * r + (col - 8)]
                    }
                }
            }
            context = next
        }
        tokens += outputRows
        return call
    }

    mutating func push(_ frames: LFM3AudioMelBuffer, isFinal: Bool = false) -> [LFM3AudioEncoderCall] {
        guard !finished else { return [] }
        pending.append(frames)
        var calls: [LFM3AudioEncoderCall] = []
        let full = 8 * tokensPerCall
        while pending.frames >= full && !(isFinal && pending.frames == full) {
            calls.append(makeCall(pending.columns(0, full), final: false))
            pending.dropFirst(full)
        }
        if isFinal {
            if pending.frames > 0 || calls.isEmpty { calls.append(makeCall(pending, final: true)) }
            pending = LFM3AudioMelBuffer()
            finished = true
        }
        return calls
    }
}

/// Core ML causal audio encoder (`AudioEncoder-fp16`: functions n12 / n96 / n375 with explicit conv / K / V
/// state I/O and ping-pong output backings): mel -> LM input rows `[n, 2048]` fp16.
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
final class LFM3AudioCoreMLEncoder {
    private final class Function {
        let model: MLModel
        let inputs: [String: MLMultiArray]
        let outputs: [[String: MLMultiArray]]
        let providers: [LFM3AudioInputs]
        let options: [MLPredictionOptions]

        init(model: MLModel) throws {
            self.model = model
            inputs = try LFM3AudioArrays.zeroInputs(model.modelDescription)
            var outputs: [[String: MLMultiArray]] = []
            var options: [MLPredictionOptions] = []
            for _ in 0..<2 {
                let backing = try LFM3AudioArrays.outputBackings(model.modelDescription)
                let option = MLPredictionOptions()
                option.outputBackings = backing
                outputs.append(backing)
                options.append(option)
            }
            self.outputs = outputs
            self.options = options
            providers = [LFM3AudioInputs(inputs), LFM3AudioInputs(inputs)]
        }
    }

    private let functions: [Int: Function]
    private let zeroConv: MLMultiArray
    private let zeroK: MLMultiArray
    private let zeroV: MLMultiArray
    private var conv: MLMultiArray
    private var k: MLMultiArray
    private var v: MLMultiArray
    private var flip = 0

    /// - Parameter models: tokens per call (12 / 96 / 375) -> loaded function.
    init(models: [Int: MLModel]) throws {
        guard !models.isEmpty else { throw LFM3AudioError.modelNotFound("AudioEncoder functions") }
        functions = try models.mapValues { try Function(model: $0) }
        guard let any = functions.values.first else { throw LFM3AudioError.modelNotFound("AudioEncoder") }
        zeroConv = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "conv").shape.map(\.intValue))
        zeroK = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "k").shape.map(\.intValue))
        zeroV = try LFM3AudioArrays.float16(try LFM3AudioArrays.require(any.inputs, "v").shape.map(\.intValue))
        conv = zeroConv
        k = zeroK
        v = zeroV
    }

    var availableTokens: [Int] { functions.keys.sorted() }

    func resetState() {
        for array in [zeroConv, zeroK, zeroV] { array.lfm3Float16.update(repeating: 0, count: array.count) }
        conv = zeroConv
        k = zeroK
        v = zeroV
    }

    /// Runs planner calls with the N-token function; appends the valid rows `[outputRows, 2048]` to `rows`.
    func run(_ calls: [LFM3AudioEncoderCall], tokensPerCall n: Int, rows: inout [Float16]) throws {
        guard let fn = functions[n] else { throw LFM3AudioError.modelNotFound("AudioEncoder function n\(n)") }
        for call in calls {
            try copy(call.mel, into: LFM3AudioArrays.require(fn.inputs, "mel"))
            try copy(call.m1, into: LFM3AudioArrays.require(fn.inputs, "m1"))
            try copy(call.m2, into: LFM3AudioArrays.require(fn.inputs, "m2"))
            try copy(call.mask, into: LFM3AudioArrays.require(fn.inputs, "mask"))
            let provider = fn.providers[flip]
            provider.set(conv, for: "conv")
            provider.set(k, for: "k")
            provider.set(v, for: "v")
            _ = try fn.model.prediction(from: provider, options: fn.options[flip])
            let out = fn.outputs[flip]
            conv = try LFM3AudioArrays.require(out, "conv_out")
            k = try LFM3AudioArrays.require(out, "k_out")
            v = try LFM3AudioArrays.require(out, "v_out")
            flip ^= 1
            let embeddings = try LFM3AudioArrays.require(out, "emb")
            rows.append(
                contentsOf: UnsafeBufferPointer(
                    start: embeddings.lfm3Float16, count: call.outputRows * LFM3AudioConfig.hiddenSize))
        }
    }

    /// Whole-clip mel -> rows; uses the smallest loaded function that covers the clip (else the largest).
    func encode(_ mel: LFM3AudioMelBuffer) throws -> [Float16] {
        resetState()
        let needed = LFM3AudioLogMel.tokenCount(frames: mel.frames)
        let offline = availableTokens.filter { $0 != LFM3AudioConfig.liveEncoderTokens }
        let sizes = offline.isEmpty ? availableTokens : offline
        guard let largest = sizes.last else { throw LFM3AudioError.modelNotFound("AudioEncoder") }
        let n = sizes.first(where: { $0 >= needed }) ?? largest
        var planner = LFM3AudioEncoderPlanner(tokensPerCall: n)
        var rows: [Float16] = []
        rows.reserveCapacity(needed * LFM3AudioConfig.hiddenSize)
        try run(planner.push(mel, isFinal: true), tokensPerCall: n, rows: &rows)
        return rows
    }

    private func copy(_ source: [Float], into array: MLMultiArray) throws {
        guard source.count == array.count else {
            throw LFM3AudioError.processingFailed("encoder input size \(source.count) != \(array.count)")
        }
        let p = array.lfm3Float16
        for i in 0..<source.count { p[i] = Float16(source[i]) }
    }
}
#endif
