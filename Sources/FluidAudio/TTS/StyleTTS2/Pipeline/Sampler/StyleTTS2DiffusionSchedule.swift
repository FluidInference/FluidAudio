import Foundation

/// Karras-noise + ADPM2-step helpers for the StyleTTS2 fused diffusion sampler.
///
/// The fused CoreML graph (`fused_diffusion_sampler_fp16.mlmodelc`) consumes a
/// pre-materialized stack of noise vectors instead of looping inside CoreML —
/// the Python orchestrator pre-draws `noise_init` ([1, 1, 256]) and
/// `noises_aux` ([num_steps - 1, 1, 1, 256]) under a seeded torch RNG before
/// dispatching the fused graph. We replicate that contract here.
public enum StyleTTS2DiffusionSchedule {

    /// Karras schedule:
    ///   `sigmas[i] = (sigma_max^(1/rho) + i/(N-1) * (sigma_min^(1/rho) - sigma_max^(1/rho)))^rho`
    /// padded with a final 0.0 (matches `F.pad(value=0.0)` in the upstream
    /// `_karras_sigmas`). Returned length is `numSteps + 1`.
    public static func karrasSigmas(
        numSteps: Int,
        sigmaMin: Double = StyleTTS2Constants.sigmaMin,
        sigmaMax: Double = StyleTTS2Constants.sigmaMax,
        rho: Double = StyleTTS2Constants.rhoSchedule
    ) -> [Double] {
        precondition(numSteps >= 2, "numSteps must be ≥ 2 for the Karras formula")
        let rhoInv = 1.0 / rho
        var sigmas = [Double](repeating: 0, count: numSteps + 1)
        let denom = Double(numSteps - 1)
        let baseMax = pow(sigmaMax, rhoInv)
        let baseMin = pow(sigmaMin, rhoInv)
        for i in 0..<numSteps {
            let frac = Double(i) / denom
            let inner = baseMax + frac * (baseMin - baseMax)
            sigmas[i] = pow(inner, rho)
        }
        sigmas[numSteps] = 0  // F.pad terminator
        return sigmas
    }
}

/// Box-Muller normal sampler. The fused-sampler path needs `noise_init`
/// (1 × 1 × 256) plus `numSteps - 1` aux vectors (each 1 × 1 × 256). Since the
/// fused graph is deterministic in the noise inputs, we can use any
/// reproducible Gaussian source — we don't need bit-for-bit parity with
/// torch's RNG (the upstream parity guarantee is between `--no-fused` and
/// fused for *the same* seed; fused vs Swift sampler gives equivalent-quality
/// audio with different fine detail).
public struct StyleTTS2NoiseSource: Sendable {

    private var state: UInt64

    public init(seed: UInt64) {
        // SplitMix64 init from a non-zero seed.
        self.state = seed == 0 ? 0xdead_beef_cafe_babe : seed
    }

    /// Draw the next double-precision uniform in (0, 1].
    private mutating func nextUniform() -> Double {
        // SplitMix64 step.
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        // Map to (0, 1] (avoid log(0) downstream).
        let u = Double(z >> 11) / Double(1 << 53)
        return u <= 0 ? .leastNormalMagnitude : u
    }

    /// Draw a single Gaussian sample N(0, 1).
    public mutating func nextGaussian() -> Float {
        let u1 = nextUniform()
        let u2 = nextUniform()
        let mag = sqrt(-2.0 * log(u1))
        return Float(mag * cos(2.0 * .pi * u2))
    }

    /// Fill `count` Float32 Gaussians.
    public mutating func nextGaussianArray(count: Int) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        for i in 0..<count { out[i] = nextGaussian() }
        return out
    }
}
