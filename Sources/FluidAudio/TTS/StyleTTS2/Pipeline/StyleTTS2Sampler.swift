import Foundation

/// ADPM2 + Karras-schedule diffusion sampler for the StyleTTS2 style vector.
///
/// This is the Swift port of `karras_sigmas` + `adpm2_sample` from the
/// upstream reference (`mobius-styletts2/scripts/99b_e2e_coreml.py`).
/// Operates on 256-dim float vectors (the flattened `(1, 1, 256)` style
/// latent fed to `diffusion_step_512`).
///
/// The sampler is decoupled from CoreML: callers pass a closure that
/// performs the actual `denoise` step. This keeps the math pure and
/// testable without loading any models.
public enum StyleTTS2Sampler {

    /// One diffusion step. Takes the current noisy latent (`x_noisy`,
    /// 256 floats) and the current sigma, returns the denoised latent.
    ///
    /// The closure captures `embedding` (`bert_dur_pad`) and `features`
    /// (`ref_s`) so the sampler doesn't need to know about model I/O
    /// shapes.
    public typealias DenoiseStep =
        @Sendable (
            _ xNoisy: [Float],
            _ sigma: Float
        ) async throws -> [Float]

    /// Compute the Karras sigma schedule with `num_steps + 1` entries
    /// (the trailing 0.0 marks the final denoise-to-zero step).
    ///
    /// Matches `99b_e2e_coreml.py:84-90`. Note: `rho` defaults to 9.0
    /// (StyleTTS2 contract), *not* the k-diffusion default of 7.0.
    public static func karrasSigmas(
        steps: Int,
        sigmaMin: Float = StyleTTS2Constants.karrasSigmaMin,
        sigmaMax: Float = StyleTTS2Constants.karrasSigmaMax,
        rho: Float = StyleTTS2Constants.karrasRho
    ) -> [Float] {
        precondition(steps >= 1, "diffusion steps must be >= 1")
        let rhoInv = 1.0 / Double(rho)
        let minInv = pow(Double(sigmaMin), rhoInv)
        let maxInv = pow(Double(sigmaMax), rhoInv)
        var sigmas: [Float] = []
        sigmas.reserveCapacity(steps + 1)
        // np.linspace(0, 1, steps) — inclusive at both ends.
        for i in 0..<steps {
            let t = steps == 1 ? 0.0 : Double(i) / Double(steps - 1)
            let inv = maxInv + t * (minInv - maxInv)
            sigmas.append(Float(pow(inv, Double(rho))))
        }
        sigmas.append(0.0)
        return sigmas
    }

    /// ADPM2 sampler over the diffusion step model. Mirrors the reference
    /// `adpm2_sample` exactly:
    ///
    ///   x = noise * sigmas[0]
    ///   for i in 0..<steps:
    ///       s, s_next = sigmas[i], sigmas[i+1]
    ///       if s_next == 0:
    ///           x += d * (s_next - s)        // single step
    ///       else:
    ///           x_mid = x + d * (s_mid - s)  // midpoint
    ///           d_mid = (x_mid - denoise(x_mid, s_mid)) / s_mid
    ///           x = x + d_mid * (s_next - s)
    ///
    /// - Parameters:
    ///   - steps: Number of sampler steps (Karras schedule length).
    ///   - noise: Initial Gaussian noise vector. Must be 256 floats.
    ///   - denoise: Closure performing the diffusion step.
    /// - Returns: Predicted style vector, 256 floats.
    public static func adpm2Sample(
        steps: Int,
        noise: [Float],
        denoise: DenoiseStep
    ) async throws -> [Float] {
        let dim = StyleTTS2Constants.refStyleDim
        precondition(noise.count == dim, "noise must be \(dim) floats")
        let sigmas = karrasSigmas(steps: steps)

        // x = noise * sigmas[0]
        var x = scale(noise, by: sigmas[0])

        for i in 0..<(sigmas.count - 1) {
            let s = sigmas[i]
            let sNext = sigmas[i + 1]

            if sNext == 0.0 {
                // Final step: a plain Euler update toward the denoised target.
                let denoised = try await denoise(x, s)
                let d = scale(sub(x, denoised), by: 1.0 / s)
                x = add(x, scale(d, by: sNext - s))
                continue
            }

            // ADPM2 midpoint update.
            let sMid = Float(exp((log(Double(s)) + log(Double(sNext))) / 2.0))
            let denoised = try await denoise(x, s)
            let d = scale(sub(x, denoised), by: 1.0 / s)
            let xMid = add(x, scale(d, by: sMid - s))
            let denoisedMid = try await denoise(xMid, sMid)
            let dMid = scale(sub(xMid, denoisedMid), by: 1.0 / sMid)
            x = add(x, scale(dMid, by: sNext - s))
        }
        return x
    }

    // MARK: - Tiny helpers

    private static func scale(_ a: [Float], by k: Float) -> [Float] {
        var out = a
        for i in 0..<out.count { out[i] *= k }
        return out
    }

    private static func add(_ a: [Float], _ b: [Float]) -> [Float] {
        var out = a
        for i in 0..<out.count { out[i] += b[i] }
        return out
    }

    private static func sub(_ a: [Float], _ b: [Float]) -> [Float] {
        var out = a
        for i in 0..<out.count { out[i] -= b[i] }
        return out
    }
}
