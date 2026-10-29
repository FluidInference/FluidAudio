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
}

/// MaskGIT depth decoder, per-pass: `depth_in` once per frame, then 3 x `depth_pass` (fp16) revealing 4 codebooks
/// each in a random order, with host fp32 sampling (presence penalty -> temperature -> top-k -> Gumbel-max).
/// 99.6% of codes identical to HF on 319 golden frames.
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
    private(set) var history: LFM3AudioPresenceHistory
    private var noise: [Float]
    private var row: [Float]
    private var sorted: [Float]

    init(models: LFM3AudioSpeechModels, sampling: LFM3AudioSpeechSampling) throws {
        depthIn = models.depthIn
        depthPass = models.depthPass
        tables = models.depthTables
        self.sampling = sampling
        history = LFM3AudioPresenceHistory(window: sampling.window)
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
        noise = [Float](repeating: 0, count: C.codebooks * C.audioVocab)
        row = [Float](repeating: 0, count: C.audioVocab)
        sorted = [Float](repeating: 0, count: C.audioVocab)
    }

    func reset() { history = LFM3AudioPresenceHistory(window: sampling.window) }

    func record(_ codes: [Int32]) { history.push(codes) }

    /// Sample one frame's 12 codes from the LM final-norm hidden `[2048]`.
    func frame(_ lmHidden: [Float16], random: inout LFM3AudioRandom) throws -> [Int32] {
        guard lmHidden.count == LFM3AudioConfig.hiddenSize else {
            throw LFM3AudioError.processingFailed("depth decoder hidden size \(lmHidden.count)")
        }
        var order = Array(0..<C.codebooks)
        for i in stride(from: C.codebooks - 1, to: 0, by: -1) {
            order.swapAt(i, Int(random.next() % UInt64(i + 1)))
        }
        fillGumbel(&random)
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
                let k = order[i]
                codes[k] = sample(logitPointer + k * C.audioVocab, codebook: k)
                revealed[k] = true
            }
        }
        return codes
    }

    static func isEndOfSpeech(_ codes: [Int32]) -> Bool { codes.contains(C.eosCode) }

    // MARK: - Private

    /// Gumbel(0, 1) = -log(-log(u)) for all 12 x 2017 logits, u with 24-bit resolution (two draws per UInt64).
    private func fillGumbel(_ random: inout LFM3AudioRandom) {
        let n = noise.count
        var count = Int32(n)
        noise.withUnsafeMutableBufferPointer { buffer in
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
    }

    /// HF order: presence penalty -> temperature -> top-k -> Gumbel-max (fp32 on the host).
    private func sample(_ logits: UnsafePointer<Float16>, codebook k: Int) -> Int32 {
        let v = C.audioVocab
        let penalty = sampling.penalty
        let greedy = sampling.temperature == 0
        let inverseTemperature = greedy ? 1 : 1 / sampling.temperature
        let topK = sampling.topK
        let counts = history.counts
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
                sorted.withUnsafeMutableBufferPointer { tmp in
                    guard let t = tmp.baseAddress else { return }
                    t.update(from: s, count: v)
                    vDSP_vsort(t, vDSP_Length(v), -1)  // descending
                    threshold = t[topK - 1]
                }
            }
            return noise.withUnsafeBufferPointer { noiseBuffer -> Int32 in
                guard let nz = noiseBuffer.baseAddress.map({ $0 + k * v }) else { return 0 }
                var best = -Float.infinity
                var index = 0
                for c in 0..<v where s[c] >= threshold {
                    let value = s[c] + Float(Float16(nz[c]))
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
        table.withUnsafeBytes { raw in
            guard let src = raw.baseAddress?.assumingMemoryBound(to: Float16.self) else { return }
            for k in 0..<LFM3AudioSpeechConfig.codebooks {
                let row = src + (Int(codes[k]) + LFM3AudioSpeechConfig.audioVocab * k) * h
                for j in 0..<h { sum[j] += Float(row[j]) }
            }
        }
        return sum.map { Float16($0) }
    }
}
#endif
