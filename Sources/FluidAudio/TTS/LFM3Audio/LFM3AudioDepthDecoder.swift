#if TTS && arch(arm64)
import Accelerate
@preconcurrency import CoreML
import Foundation

/// SplitMix64 (speech-frame randomness: MaskGIT reveal order + Gumbel noise).
struct LFM3AudioRandom: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Per-codebook presence of codes over the last `window` frames (ring + counts) for the presence penalty.
struct LFM3AudioPresenceHistory {
    private let window: Int
    private var ring: [[Int32]] = []
    private var head = 0
    private(set) var counts = [Int32](
        repeating: 0, count: LFM3AudioSpeechConfig.codebooks * LFM3AudioSpeechConfig.audioVocab)

    init(window: Int) { self.window = window }

    mutating func push(_ codes: [Int32]) {
        guard window > 0 else { return }
        let v = LFM3AudioSpeechConfig.audioVocab
        if ring.count == window {
            let old = ring[head]
            for k in 0..<LFM3AudioSpeechConfig.codebooks { counts[k * v + Int(old[k])] -= 1 }
            ring[head] = codes
            head = (head + 1) % window
        } else {
            ring.append(codes)
        }
        for k in 0..<LFM3AudioSpeechConfig.codebooks { counts[k * v + Int(codes[k])] += 1 }
    }

    /// `[12, 2017]` presence mask (1 = code seen in the window) for the fused `depth_frame` graph.
    func mask() -> [Float16] { counts.map { $0 > 0 ? 1 : 0 } }
}

/// One frame's randomness, drawn the same way for every depth backend: the MaskGIT reveal order (a random
/// permutation of the 12 codebooks, 4 per pass) and Gumbel(0, 1) noise for all 12 x 2017 logits.
struct LFM3AudioDepthFrame: Sendable {
    let order: [Int]
    /// fp32 Gumbel noise rounded to fp16 (as the graph input).
    let noise: [Float16]

    /// - Parameter scratch: reusable fp32 buffer for the uniforms (resized to 12 x 2017 on first use).
    static func draw(_ random: inout LFM3AudioRandom, scratch u: inout [Float]) -> LFM3AudioDepthFrame {
        let c = LFM3AudioSpeechConfig.self
        var order = Array(0..<c.codebooks)
        for i in stride(from: c.codebooks - 1, to: 0, by: -1) {
            order.swapAt(i, Int(random.next() % UInt64(i + 1)))
        }
        // Gumbel(0, 1) = -log(-log(u)), u with 24-bit resolution (two draws per UInt64)
        let n = c.codebooks * c.audioVocab
        if u.count != n { u = [Float](repeating: 0, count: n) }
        var count = Int32(n)
        u.withUnsafeMutableBufferPointer { buffer in
            guard let p = buffer.baseAddress else { return }
            var i = 0
            while i < n {
                let r = random.next()
                p[i] = (Float(r >> 40) + 0.5) * (1.0 / 16_777_216.0)
                if i + 1 < n { p[i + 1] = (Float((r >> 16) & 0xFF_FFFF) + 0.5) * (1.0 / 16_777_216.0) }
                i += 2
            }
            var minusOne: Float = -1
            vvlogf(p, p, &count)
            vDSP_vsmul(p, 1, &minusOne, p, 1, vDSP_Length(n))
            vvlogf(p, p, &count)
            vDSP_vsmul(p, 1, &minusOne, p, 1, vDSP_Length(n))
        }
        return LFM3AudioDepthFrame(order: order, noise: u.map { Float16($0) })
    }

    /// `[3, 12]` one-hot: pass p reveals codebooks `order[4p ..< 4p + 4]`.
    var schedule: [Float16] {
        var s = [Float16](repeating: 0, count: 3 * LFM3AudioSpeechConfig.codebooks)
        for (i, k) in order.enumerated() { s[(i / 4) * LFM3AudioSpeechConfig.codebooks + k] = 1 }
        return s
    }
}

/// A depth decoder backend: one speech frame's 12 codes from the LM final-norm hidden.
protocol LFM3AudioDepthModel: Actor {
    func frame(hidden: [Float16], frame: LFM3AudioDepthFrame, present: [Float16]) async throws -> [Int32]
}

/// MaskGIT depth decoder on Core ML, per-pass: `depth_in` once per frame, then 3 x `depth_pass` (fp16) revealing
/// 4 codebooks each in the frame's order, with host fp32 sampling (presence penalty -> temperature -> top-k ->
/// Gumbel-max). 99.6% of codes identical to HF on 319 golden frames.
@available(macOS 15, iOS 18, tvOS 18, visionOS 2, *)
final class LFM3AudioDepthDecoder {
    private typealias C = LFM3AudioSpeechConfig
    private let depthIn: MLModel
    private let depthPass: MLModel
    private let inInputs: LFM3AudioInputs
    private let passInputs: LFM3AudioInputs
    private let inOptions = MLPredictionOptions()
    private let passOptions = MLPredictionOptions()
    private let hidden: MLMultiArray  // [1, 2048] depth_in input
    private let depthInput: MLMultiArray  // [1, 12, 1024] depth_in output
    private let slots: MLMultiArray  // [1, 12, 1024] depth_pass input
    private let logits: MLMultiArray  // [1, 12, 2017] depth_pass output
    private let tables: Data
    private let sampling: LFM3AudioSpeechSampling
    private var row: [Float]
    private var selection: [Float]

    init(depthIn: MLModel, depthPass: MLModel, tables: Data, sampling: LFM3AudioSpeechSampling) throws {
        self.depthIn = depthIn
        self.depthPass = depthPass
        self.tables = tables
        self.sampling = sampling
        let inArrays = try LFM3AudioArrays.zeroInputs(depthIn.modelDescription)
        hidden = try LFM3AudioArrays.require(inArrays, "h")
        inInputs = LFM3AudioInputs(inArrays)
        let inOutputs = try LFM3AudioArrays.outputBackings(depthIn.modelDescription)
        depthInput = try LFM3AudioArrays.require(inOutputs, "din")
        inOptions.outputBackings = inOutputs
        let passArrays = try LFM3AudioArrays.zeroInputs(depthPass.modelDescription)
        slots = try LFM3AudioArrays.require(passArrays, "slots")
        passInputs = LFM3AudioInputs(passArrays)
        let passOutputs = try LFM3AudioArrays.outputBackings(depthPass.modelDescription)
        logits = try LFM3AudioArrays.require(passOutputs, "logits")
        passOptions.outputBackings = passOutputs
        guard hidden.count == LFM3AudioConfig.hiddenSize, depthInput.count == C.codebooks * C.depthHidden,
            slots.count == depthInput.count, logits.count == C.codebooks * C.audioVocab
        else {
            throw LFM3AudioError.invalidModel("unexpected depth decoder shapes")
        }
        row = [Float](repeating: 0, count: C.audioVocab)
        selection = [Float](repeating: 0, count: C.audioVocab)
    }

    /// Sample one frame's 12 codes from the LM final-norm hidden `[2048]`.
    func frame(
        _ lmHidden: [Float16], frame: LFM3AudioDepthFrame, history: LFM3AudioPresenceHistory
    ) throws
        -> [Int32]
    {
        guard lmHidden.count == LFM3AudioConfig.hiddenSize else {
            throw LFM3AudioError.processingFailed("depth decoder hidden size \(lmHidden.count)")
        }
        lmHidden.withUnsafeBufferPointer { src in
            guard let base = src.baseAddress else { return }
            hidden.lfm3Float16.update(from: base, count: src.count)
        }
        _ = try depthIn.prediction(from: inInputs, options: inOptions)
        var codes = [Int32](repeating: 0, count: C.codebooks)
        var revealed = [Bool](repeating: false, count: C.codebooks)
        let din = depthInput.lfm3Float16
        let slotPointer = slots.lfm3Float16
        let logitPointer = logits.lfm3Float16
        let d = C.depthHidden
        for pass in 0..<3 {
            // slots = din + (mask_embedding | emb_k[code_k])  (fp16 adds, as the graph)
            tables.withUnsafeBytes { raw in
                guard let table = raw.baseAddress?.assumingMemoryBound(to: Float16.self) else { return }
                let maskEmbedding = table + C.codebookRows * d
                for k in 0..<C.codebooks {
                    let dst = slotPointer + k * d
                    let base = din + k * d
                    let add = revealed[k] ? table + (k * C.audioVocab + Int(codes[k])) * d : maskEmbedding
                    for j in 0..<d { dst[j] = base[j] + add[j] }
                }
            }
            _ = try depthPass.prediction(from: passInputs, options: passOptions)
            for i in (4 * pass)..<(4 * pass + 4) {
                let k = frame.order[i]
                codes[k] = sample(
                    logitPointer + k * C.audioVocab, codebook: k, noise: frame.noise, counts: history.counts)
                revealed[k] = true
            }
        }
        return codes
    }

    static func isEndOfSpeech(_ codes: [Int32]) -> Bool { codes.contains(C.eosCode) }

    /// The k-th largest of `values[0 ..< count]` (k is 1-based, clamped to 1 ... count) by quickselect, reordering
    /// `values` (no NaNs). Exactly the value a descending sort puts at index k - 1, without sorting all 2017 logits
    /// for the top-k threshold.
    static func kthLargest(_ values: UnsafeMutablePointer<Float>, count: Int, k: Int) -> Float {
        guard count > 0 else { return -.infinity }
        let target = min(max(k, 1), count) - 1
        var lo = 0
        var hi = count - 1
        while lo < hi {
            let pivot = values[lo + (hi - lo) / 2]
            var i = lo
            var j = hi
            while i <= j {  // Hoare partition, descending
                while values[i] > pivot { i += 1 }
                while values[j] < pivot { j -= 1 }
                if i <= j {
                    let swapped = values[i]
                    values[i] = values[j]
                    values[j] = swapped
                    i += 1
                    j -= 1
                }
            }
            if target <= j {
                hi = j
            } else if target >= i {
                lo = i
            } else {
                break  // values[j + 1 ..< i] all equal the pivot
            }
        }
        return values[target]
    }

    // MARK: - Private

    /// HF order: presence penalty -> temperature -> top-k -> Gumbel-max (fp32 on the host).
    private func sample(
        _ logits: UnsafePointer<Float16>, codebook k: Int, noise: [Float16], counts: [Int32]
    ) -> Int32 {
        let v = C.audioVocab
        let penalty = sampling.penalty
        let greedy = sampling.temperature == 0
        let inverseTemperature = greedy ? 1 : 1 / sampling.temperature
        let topK = sampling.topK
        return row.withUnsafeMutableBufferPointer { rowBuffer -> Int32 in
            guard let s = rowBuffer.baseAddress else { return 0 }
            counts.withUnsafeBufferPointer { countBuffer in
                guard let cnt = countBuffer.baseAddress.map({ $0 + k * v }) else { return }
                for c in 0..<v {
                    var x = Float(logits[c])
                    if penalty != 1 && cnt[c] > 0 { x = x < 0 ? x * penalty : x / penalty }
                    s[c] = x * inverseTemperature
                }
            }
            if greedy {
                var best: Float = 0
                var index: vDSP_Length = 0
                vDSP_maxvi(s, 1, &best, &index, vDSP_Length(v))
                return Int32(index)
            }
            var threshold = -Float.infinity
            if topK > 0 && topK < v {
                selection.withUnsafeMutableBufferPointer { tmp in
                    guard let t = tmp.baseAddress else { return }
                    // NaN logits rank below every number (they are skipped below), as in a descending sort
                    for c in 0..<v { t[c] = s[c].isNaN ? -.infinity : s[c] }
                    threshold = Self.kthLargest(t, count: v, k: topK)
                }
            }
            return noise.withUnsafeBufferPointer { noiseBuffer -> Int32 in
                guard let nz = noiseBuffer.baseAddress.map({ $0 + k * v }) else { return 0 }
                var best = -Float.infinity
                var index = 0
                for c in 0..<v where s[c] >= threshold {
                    let value = s[c] + Float(nz[c])
                    if value > best {
                        best = value
                        index = c
                    }
                }
                return Int32(index)
            }
        }
    }
}

/// LM input of a speech frame: sum over codebooks of `codebook_embedding[code_k + 2017 k]` (fp32 accumulate).
struct LFM3AudioCodebookEmbedding: Sendable {
    let table: Data

    func embed(_ codes: [Int32]) -> [Float16] {
        let h = LFM3AudioConfig.hiddenSize
        var sum = [Float](repeating: 0, count: h)
        var row = [Float](repeating: 0, count: h)
        table.withUnsafeBytes { raw in
            guard let src = raw.baseAddress?.assumingMemoryBound(to: UInt16.self) else { return }
            for k in 0..<LFM3AudioSpeechConfig.codebooks {
                // fp16 -> fp32 is exact; the fp32 adds keep the codebook order (bit-identical to a scalar loop)
                Float16Conversion.toFloat32(
                    src: src + (Int(codes[k]) + LFM3AudioSpeechConfig.audioVocab * k) * h, dst: &row, count: h)
                sum.withUnsafeMutableBufferPointer { acc in
                    row.withUnsafeBufferPointer { r in
                        guard let a = acc.baseAddress, let b = r.baseAddress else { return }
                        vDSP_vadd(a, 1, b, 1, a, 1, vDSP_Length(h))
                    }
                }
            }
        }
        return sum.map { Float16($0) }
    }
}
#endif
