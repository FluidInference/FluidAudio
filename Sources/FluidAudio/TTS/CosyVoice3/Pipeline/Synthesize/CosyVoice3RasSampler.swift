import Foundation

/// RAS (Repetition-Aware Sampling) — top-p nucleus sampling with a repetition
/// mask that re-samples if a token fires too often in the recent window.
///
/// Mirrors `ras_sampling` in
/// `mobius/.../verify/test_coreml_e2e_fp16.py`:
///   1. softmax(logp) → stable-sort desc → pick up to `topK` ids until
///      cumulative mass ≥ `topP`
///   2. multinomial draw within that candidate set
///   3. if the drawn id appears in the last `winSize` decoded tokens at least
///      `winSize * tauR` times, mask it to -inf and re-sample across the full
///      vocab
///
/// A `seedTokens` mode bypasses the RNG entirely — the sampler just emits the
/// pre-recorded Python token stream one id at a time. This is how the parity
/// harness bit-matches despite the `torch.multinomial` RNG mismatch between
/// PyTorch and Swift.
public final class CosyVoice3RasSampler {

    public let topP: Float
    public let topK: Int
    public let winSize: Int
    public let tauR: Float
    public let vocabSize: Int

    private var rng: SeedableRng
    private var seedQueue: [Int32]
    private var seedIdx: Int = 0

    public init(
        topP: Float = CosyVoice3Constants.topP,
        topK: Int = CosyVoice3Constants.topK,
        winSize: Int = CosyVoice3Constants.rasWindow,
        tauR: Float = CosyVoice3Constants.rasTauR,
        vocabSize: Int = CosyVoice3Constants.speechVocab,
        seed: UInt64 = 42
    ) {
        self.topP = topP
        self.topK = topK
        self.winSize = winSize
        self.tauR = tauR
        self.vocabSize = vocabSize
        self.rng = SeedableRng(seed: seed)
        self.seedQueue = []
    }

    /// Pre-load a token stream to replay (for parity harness).
    public func seedTokens(_ tokens: [Int32]) {
        self.seedQueue = tokens
        self.seedIdx = 0
    }

    /// Given `logits` of shape `[vocabSize]`, return the sampled token id.
    /// `decodedSoFar` is the running decoded stream for repetition checking.
    public func sample(logits: [Float], decodedSoFar: [Int32]) -> Int32 {
        // Seeded parity replay bypasses sampling.
        if seedIdx < seedQueue.count {
            let id = seedQueue[seedIdx]
            seedIdx += 1
            return id
        }
        precondition(logits.count == vocabSize, "logits count must match vocabSize")

        // Pass 1: nucleus sampling.
        let probs = logits.softmax()
        let top = nucleus(probs: probs)
        var sampled = top

        // Pass 2: repetition mask.
        let windowStart = max(0, decodedSoFar.count - winSize)
        let recent = decodedSoFar[windowStart..<decodedSoFar.count]
        let rep = recent.filter { $0 == sampled }.count
        if Float(rep) >= Float(winSize) * tauR {
            var masked = probs
            masked[Int(sampled)] = 0
            // Re-normalize + multinomial across full vocab.
            let sum = masked.reduce(0, +)
            if sum > 0 {
                for i in 0..<masked.count { masked[i] /= sum }
            }
            sampled = multinomial(probs: masked)
        }
        return sampled
    }

    // MARK: - Nucleus helper

    private func nucleus(probs: [Float]) -> Int32 {
        // Stable sort descending with index.
        let sorted = probs.enumerated().sorted {
            if $0.element != $1.element { return $0.element > $1.element }
            return $0.offset < $1.offset
        }
        var cum: Float = 0
        var selIdx: [Int] = []
        var selProb: [Float] = []
        for entry in sorted {
            if cum < topP && selProb.count < topK {
                cum += entry.element
                selProb.append(entry.element)
                selIdx.append(entry.offset)
            } else {
                break
            }
        }
        // Normalize selected candidates and multinomial pick.
        let sum = selProb.reduce(0, +)
        guard sum > 0 else { return Int32(selIdx.first ?? 0) }
        for i in 0..<selProb.count { selProb[i] /= sum }
        let picked = multinomialInSet(probs: selProb, ids: selIdx)
        return Int32(picked)
    }

    private func multinomial(probs: [Float]) -> Int32 {
        let u = rng.nextFloat()
        var cum: Float = 0
        for (i, p) in probs.enumerated() {
            cum += p
            if u < cum { return Int32(i) }
        }
        return Int32(probs.count - 1)
    }

    private func multinomialInSet(probs: [Float], ids: [Int]) -> Int {
        let u = rng.nextFloat()
        var cum: Float = 0
        for (j, p) in probs.enumerated() {
            cum += p
            if u < cum { return ids[j] }
        }
        return ids.last ?? 0
    }
}

// MARK: - Simple deterministic RNG

/// Linear-congruential PRNG wrapping SplitMix64. Used only as a fallback when
/// parity replay isn't active; the parity harness seeds an explicit token list
/// to dodge `torch.multinomial` divergence.
private struct SeedableRng {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed == 0 ? 0xdead_beef : seed }
    mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func nextFloat() -> Float {
        // 24-bit mantissa → [0, 1)
        let bits = UInt32(truncatingIfNeeded: nextUInt64() >> 40)
        return Float(bits) / Float(1 << 24)
    }
}

// MARK: - Array softmax

extension Array where Element == Float {
    fileprivate func softmax() -> [Float] {
        guard let m = self.max() else { return self }
        var exps = [Float](repeating: 0, count: self.count)
        var sum: Float = 0
        for i in 0..<self.count {
            let e = expf(self[i] - m)
            exps[i] = e
            sum += e
        }
        if sum > 0 {
            for i in 0..<exps.count { exps[i] /= sum }
        }
        return exps
    }
}
