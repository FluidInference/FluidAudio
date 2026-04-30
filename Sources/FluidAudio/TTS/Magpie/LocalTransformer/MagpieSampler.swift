import Foundation

/// RNG wrapper for the Magpie sampler. Always backed by `MagpieMT19937` so a
/// `seed` round-trips bit-identical against `np.random.seed(seed)` in the
/// Python reference; when no seed is supplied it auto-seeds from
/// `arc4random_buf` so behavior outside seeded mode is still random.
///
/// Not Sendable: holds mutable RNG state. Consumed within `MagpieSynthesizer`
/// actor isolation so it never crosses concurrency domains.
public final class MagpieSamplerRng {

    private let mt: MagpieMT19937

    public init(seed: UInt64?) {
        if let seed = seed {
            // NumPy raises for seed >= 2^32; clamp by mask to keep behavior stable
            // while still letting callers pass `UInt64`.
            self.mt = MagpieMT19937(seed: UInt32(truncatingIfNeeded: seed))
        } else {
            var bytes: UInt32 = 0
            withUnsafeMutableBytes(of: &bytes) { buf in
                if let base = buf.baseAddress {
                    arc4random_buf(base, buf.count)
                }
            }
            self.mt = MagpieMT19937(seed: bytes)
        }
    }

    /// `np.random.choice(len(probs), p=probs)` — see `MagpieMT19937.numpyChoice`.
    public func numpyChoice(probs: [Float]) -> Int {
        // Mirror NumPy: cumsum in fp32 (matches `probs.cumsum()` over fp32),
        // promote per-element to fp64 only for the final searchsorted compare.
        var cdf = [Double](repeating: 0, count: probs.count)
        var totalF: Float = 0
        for i in 0..<probs.count {
            let p = probs[i] > 0 ? probs[i] : 0
            totalF += p
            cdf[i] = Double(totalF)
        }
        if totalF <= 0 { return probs.count - 1 }
        let u = mt.uniformDouble() * Double(totalF)
        var lo = 0
        var hi = cdf.count
        while lo < hi {
            let mid = (lo &+ hi) >> 1
            if cdf[mid] > u { hi = mid } else { lo = mid + 1 }
        }
        return Swift.min(lo, probs.count - 1)
    }
}

/// Samples the 8 codebook tokens from one decoder hidden state by driving the
/// Swift Local Transformer auto-regressively.
///
/// Mirrors `local_transformer_sample` in
/// `mobius/models/tts/magpie/coreml/generate_coreml.py` (lines 172–242).
public struct MagpieLocalSampler: Sendable {

    private let lt: MagpieLocalTransformer
    private let audioEmbeddings: [[Float]]

    /// - Parameter audioEmbeddings: per-codebook `[numCodesPerCodebook × dModel]` fp32.
    public init(
        localTransformer: MagpieLocalTransformer,
        audioEmbeddings: [[Float]]
    ) {
        self.lt = localTransformer
        self.audioEmbeddings = audioEmbeddings
    }

    /// Sample one frame of `numCodebooks` codes.
    ///
    /// - Parameters:
    ///   - decoderHidden: conditional decoder hidden state, `[dModel]`.
    ///   - uncondDecoderHidden: unconditional path for CFG; `nil` disables CFG.
    ///   - forbidEos: mask `audioEosId` (set `true` while `t < minFrames`).
    ///   - options: temperature / topK / cfgScale.
    ///   - rng: NumPy-compatible MT19937 RNG.
    public func sample(
        decoderHidden: [Float],
        uncondDecoderHidden: [Float]? = nil,
        forbidEos: Bool,
        options: MagpieSynthesisOptions,
        rng: MagpieSamplerRng
    ) -> [Int32] {
        let numCodebooks = lt.weights.numCodebooks
        let D = lt.weights.localDim
        let useCfg = uncondDecoderHidden != nil && options.cfgScale != 1.0

        // Project decoder hidden through in_proj → first LT token.
        let condFirst = lt.projectInput(hidden: decoderHidden)
        var condSeq = condFirst  // growing buffer, flat row-major
        var condLen = 1

        var uncondSeq: [Float] = []
        var uncondLen = 0
        if let uncondHidden = uncondDecoderHidden {
            uncondSeq = lt.projectInput(hidden: uncondHidden)
            uncondLen = 1
        }

        var codes = Swift.Array<Int32>(repeating: 0, count: numCodebooks)
        let forbidden = forbiddenTokens(eosMasked: forbidEos)

        for cb in 0..<numCodebooks {
            let condOut = lt.forward(sequence: condSeq, length: condLen)
            let lastOffset = (condLen - 1) * D
            let lastHidden = Swift.Array(condOut[lastOffset..<(lastOffset + D)])
            var logits = lt.codebookLogits(lastHidden: lastHidden, codebook: cb)

            if useCfg {
                let uncondOut = lt.forward(sequence: uncondSeq, length: uncondLen)
                let uncondLast = Swift.Array(
                    uncondOut[((uncondLen - 1) * D)..<((uncondLen - 1) * D + D)])
                let uncondLogits = lt.codebookLogits(lastHidden: uncondLast, codebook: cb)
                let scale = options.cfgScale
                for i in 0..<logits.count {
                    logits[i] = scale * logits[i] + (1.0 - scale) * uncondLogits[i]
                }
            }

            // Mask forbidden tokens.
            for tok in forbidden where Int(tok) < logits.count {
                logits[Int(tok)] = -.infinity
            }

            let sampled = Self.sampleTopK(
                logits: logits, topK: options.topK, temperature: options.temperature,
                rng: rng)
            codes[cb] = Int32(sampled)

            // Embed sampled token → next LT input (both cond and uncond paths).
            let tokenEmb = audioEmbeddings[cb]
            let row = Int(sampled)
            let start = row * lt.weights.dModel
            let hiddenSlice = Swift.Array(tokenEmb[start..<(start + lt.weights.dModel)])
            let nextInput = lt.projectInput(hidden: hiddenSlice)

            condSeq.append(contentsOf: nextInput)
            condLen += 1
            if useCfg {
                uncondSeq.append(contentsOf: nextInput)
                uncondLen += 1
            }
        }

        return codes
    }

    // MARK: - Sampling utils

    private func forbiddenTokens(eosMasked: Bool) -> [Int32] {
        if eosMasked {
            // Block EOS + CTX_BOS + reserved.
            return [MagpieConstants.audioEosId] + MagpieConstants.forbiddenAudioIds
        } else {
            return MagpieConstants.forbiddenAudioIds
        }
    }

    /// Categorical sampling with optional top-k truncation + temperature.
    ///
    /// Matches the Python reference (`sample_topk` in `generate_coreml.py`):
    ///   1. Mask all but the top-k logits (set others to `-inf`).
    ///   2. Divide by `max(temperature, 1e-8)`.
    ///   3. Subtract max → softmax in fp32.
    ///   4. `np.random.choice(n, p=probs)` via `MagpieMT19937.numpyChoice`.
    ///
    /// Made `static` so the method is shared by both the instance call site and
    /// unit tests.
    static func sampleTopK(
        logits: [Float], topK: Int, temperature: Float,
        rng: MagpieSamplerRng
    ) -> Int {
        var truncated = logits
        if topK > 0 && topK < truncated.count {
            // Threshold = K-th largest. Found via a fixed-size min-heap of size K,
            // O(N + K log K) vs O(N log N) for the prior full-sort path. We keep
            // exactly K survivors to match `torch.topk`: all strictly-above-threshold
            // values, plus the earliest-index ties at the threshold up to a total
            // of K. The rest are masked to `-inf`.
            let threshold = Self.topKThreshold(values: truncated, k: topK)
            var aboveCount = 0
            for v in truncated where v > threshold { aboveCount += 1 }
            var tiesNeeded = topK - aboveCount
            for i in 0..<truncated.count {
                if truncated[i] > threshold {
                    continue
                }
                if truncated[i] == threshold && tiesNeeded > 0 {
                    tiesNeeded -= 1
                    continue
                }
                truncated[i] = -.infinity
            }
        }
        let t = max(temperature, 1e-8)
        for i in 0..<truncated.count {
            truncated[i] /= t
        }
        let maxVal = truncated.max() ?? 0
        var sum: Float = 0
        for i in 0..<truncated.count {
            let e = expf(truncated[i] - maxVal)
            truncated[i] = e
            sum += e
        }
        if sum <= 0 || !sum.isFinite {
            // Degenerate — fall back to argmax over original logits.
            return logits.indices.max(by: { logits[$0] < logits[$1] }) ?? 0
        }
        // Normalize → fp32 probability vector. Mirrors `probs / probs.sum()`.
        let inv = 1.0 / sum
        for i in 0..<truncated.count {
            truncated[i] *= inv
        }
        return rng.numpyChoice(probs: truncated)
    }

    /// Returns the K-th largest value in `values` using a fixed-size min-heap.
    /// O(N) heap construction (first K elements) + O((N - K) log K) replacements.
    /// Required: 1 <= k <= values.count.
    private static func topKThreshold(values: [Float], k: Int) -> Float {
        var heap = Swift.Array<Float>(repeating: 0, count: k)
        heap.withUnsafeMutableBufferPointer { buf in
            // Phase 1: insert first K values, sift each up.
            for i in 0..<k {
                buf[i] = values[i]
                var j = i
                while j > 0 {
                    let parent = (j - 1) >> 1
                    if buf[j] < buf[parent] {
                        let tmp = buf[j]
                        buf[j] = buf[parent]
                        buf[parent] = tmp
                        j = parent
                    } else {
                        break
                    }
                }
            }
            // Phase 2: for each remaining value, replace the min if it's larger,
            // then sift down.
            for i in k..<values.count {
                let v = values[i]
                if v <= buf[0] { continue }
                buf[0] = v
                var j = 0
                while true {
                    let left = 2 * j + 1
                    let right = left + 1
                    var smallest = j
                    if left < k && buf[left] < buf[smallest] { smallest = left }
                    if right < k && buf[right] < buf[smallest] { smallest = right }
                    if smallest == j { break }
                    let tmp = buf[j]
                    buf[j] = buf[smallest]
                    buf[smallest] = tmp
                    j = smallest
                }
            }
        }
        return heap[0]
    }
}
