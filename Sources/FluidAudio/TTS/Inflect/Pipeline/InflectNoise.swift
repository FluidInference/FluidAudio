import Foundation

/// Deterministic standard-normal generator for the VITS prior sample
/// (`z_p = m + noise·exp(logs)·noise_scale`). Seedable so a given
/// (text, seed) renders identical audio and unit tests are reproducible.
///
/// SplitMix64 uniform stream → Box–Muller Gaussian pairs.
struct InflectNoise {

    private var state: UInt64
    private var spare: Float?

    init(seed: UInt64) {
        // Avoid the all-zero SplitMix64 fixed point.
        self.state = seed == 0 ? 0x9E37_79B9_7F4A_7C15 : seed
    }

    private mutating func nextUInt64() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// Uniform in (0, 1].
    private mutating func nextUnit() -> Float {
        // Top 24 bits → [0, 1); shift to (0, 1] so log() is finite.
        let bits = nextUInt64() >> 40
        return (Float(bits) + 1.0) / Float(1 << 24)
    }

    /// Next N(0, 1) sample.
    mutating func nextGaussian() -> Float {
        if let s = spare {
            spare = nil
            return s
        }
        let u1 = nextUnit()
        let u2 = nextUnit()
        let radius = (-2.0 * Foundation.log(u1)).squareRoot()
        let angle = 2.0 * Float.pi * u2
        spare = radius * Foundation.sin(angle)
        return radius * Foundation.cos(angle)
    }

    /// Fill `buffer` with N(0, 1) samples.
    mutating func fill(_ buffer: inout [Float]) {
        for i in buffer.indices {
            buffer[i] = nextGaussian()
        }
    }
}
