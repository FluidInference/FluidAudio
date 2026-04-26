import Foundation

/// NumPy-compatible Mersenne Twister (MT19937) RNG used to make Magpie sampling
/// reproducible against `mobius/.../generate_coreml.py`.
///
/// NumPy's legacy `np.random.seed(seed)` initializes MT19937 via NumPy's
/// `rk_seed(unsigned long)` which is equivalent to the Matsumoto reference's
/// `init_genrand(seed)` — *not* `init_by_array`. Uniform doubles come from
/// `genrand_res53` (combine two 32-bit ints → 53-bit fraction), and
/// `np.random.choice(n, p=probs)` is: cumulative-sum → normalize → uniform
/// draw → searchsorted side='right'. This file mirrors all three.
///
/// References:
/// - Original Mersenne Twister C reference (Matsumoto/Nishimura 2002).
/// - NumPy `randomkit.c` `rk_seed` (the path `np.random.seed(int)` uses).
public final class MagpieMT19937: RandomNumberGenerator {

    private static let n = 624
    private static let m = 397
    private static let upperMask: UInt32 = 0x8000_0000
    private static let lowerMask: UInt32 = 0x7FFF_FFFF
    private static let matrixA: UInt32 = 0x9908_B0DF

    private var mt: [UInt32] = Array(repeating: 0, count: MagpieMT19937.n)
    private var mti: Int = MagpieMT19937.n

    /// Seed the generator with a single 32-bit integer (matches NumPy's
    /// `np.random.seed(seed)` for `0 ≤ seed < 2^32`).
    public init(seed: UInt32) {
        initGenrand(seed)
    }

    // MARK: - Seeding

    /// Mirrors Matsumoto's `init_genrand(s)` and NumPy's `rk_seed(s)`:
    ///   `mt[0] = s; mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i`.
    private func initGenrand(_ s: UInt32) {
        mt[0] = s
        for i in 1..<Self.n {
            mt[i] = 1_812_433_253 &* (mt[i - 1] ^ (mt[i - 1] >> 30)) &+ UInt32(i)
        }
        mti = Self.n
    }

    // MARK: - Core generation

    /// One 32-bit unsigned draw. Refills the state vector when exhausted.
    public func genrandInt32() -> UInt32 {
        if mti >= Self.n {
            let mag01: [UInt32] = [0, Self.matrixA]
            var kk = 0
            while kk < Self.n - Self.m {
                let y = (mt[kk] & Self.upperMask) | (mt[kk + 1] & Self.lowerMask)
                mt[kk] = mt[kk + Self.m] ^ (y >> 1) ^ mag01[Int(y & 1)]
                kk += 1
            }
            while kk < Self.n - 1 {
                let y = (mt[kk] & Self.upperMask) | (mt[kk + 1] & Self.lowerMask)
                mt[kk] = mt[kk &+ (Self.m - Self.n)] ^ (y >> 1) ^ mag01[Int(y & 1)]
                kk += 1
            }
            let yLast = (mt[Self.n - 1] & Self.upperMask) | (mt[0] & Self.lowerMask)
            mt[Self.n - 1] = mt[Self.m - 1] ^ (yLast >> 1) ^ mag01[Int(yLast & 1)]
            mti = 0
        }
        var y = mt[mti]
        mti += 1
        // Tempering.
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C_5680
        y ^= (y << 15) & 0xEFC6_0000
        y ^= (y >> 18)
        return y
    }

    /// 53-bit precision uniform draw in `[0, 1)` (matches `genrand_res53`).
    public func uniformDouble() -> Double {
        let a = UInt64(genrandInt32() >> 5)  // 27 bits
        let b = UInt64(genrandInt32() >> 6)  // 26 bits
        return (Double(a) * 67_108_864.0 + Double(b)) * (1.0 / 9_007_199_254_740_992.0)
    }

    // MARK: - Swift `RandomNumberGenerator` conformance

    /// Provided for Swift API parity. `Float.random(in:using:)` etc. work, but
    /// will NOT match NumPy's `random_sample()` because Swift's stdlib
    /// converts the 64-bit integer to a Double via a different masking path.
    /// Use `uniformDouble()` for NumPy parity.
    public func next() -> UInt64 {
        let lo = UInt64(genrandInt32())
        let hi = UInt64(genrandInt32())
        return (hi << 32) | lo
    }
}

// MARK: - NumPy-compatible weighted choice

extension MagpieMT19937 {

    /// Reproduces `np.random.choice(len(probs), p=probs)`.
    ///
    /// NumPy normalizes `probs`, computes `cdf = cumsum / cdf[-1]`, draws one
    /// uniform double via `random_sample()`, and returns
    /// `cdf.searchsorted(u, side='right')` — the first index where `cdf[i] > u`.
    ///
    /// `probs` may contain `-Float.infinity`-derived zeros (after softmax
    /// already eliminated forbidden / non-top-k logits). Negative weights are
    /// clamped to 0 to match NumPy's `choice`, which raises on negative inputs
    /// — callers should pre-mask.
    public func numpyChoice(probs: [Double]) -> Int {
        precondition(!probs.isEmpty, "numpyChoice requires non-empty probability vector")
        // Cumulative sum (no normalization in-place; we compare u * total).
        var cdf = [Double](repeating: 0, count: probs.count)
        var total: Double = 0
        for i in 0..<probs.count {
            let p = probs[i] > 0 ? probs[i] : 0
            total += p
            cdf[i] = total
        }
        if total <= 0 {
            return probs.count - 1
        }
        let u = uniformDouble() * total
        // `np.searchsorted(side='right')` ≡ first idx where cdf[idx] > u.
        var lo = 0
        var hi = cdf.count
        while lo < hi {
            let mid = (lo &+ hi) >> 1
            if cdf[mid] > u {
                hi = mid
            } else {
                lo = mid + 1
            }
        }
        // Clamp to last valid index (handles the rare floating-point case
        // where every cdf[i] <= u due to rounding).
        return Swift.min(lo, probs.count - 1)
    }
}
