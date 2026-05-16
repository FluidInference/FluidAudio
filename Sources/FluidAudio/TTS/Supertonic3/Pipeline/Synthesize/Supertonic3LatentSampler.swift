import Foundation

/// Helper that mirrors `sampleNoisyLatent()` / `getLatentMask()` from the
/// upstream Supertonic Swift reference.
///
/// Both routines are pure (no CoreML dependency) so the synthesizer can keep
/// the denoising loop testable. The Box-Muller transform matches the
/// reference output bit-for-bit when the same RNG is supplied.
enum Supertonic3LatentSampler {

    /// Initial noisy latent + latent mask for the diffusion denoiser.
    ///
    /// - Parameters:
    ///   - durations: Per-batch utterance durations in seconds.
    ///   - sampleRate: Vocoder sample rate (used to translate duration→samples).
    ///   - baseChunkSize: Acoustic AE `base_chunk_size` (from `tts.json`).
    ///   - chunkCompress: `chunk_compress_factor` (from `tts.json`).
    ///   - latentDim: Raw `latent_dim` (from `tts.json`).
    ///   - rng: Closure producing uniform floats in [0, 1). Defaults to
    ///     `Float.random(in:)`; tests can pass a seeded generator.
    /// - Returns: `(noisyLatent, latentMask)` where both tensors are
    ///   row-major `[bsz, latentDim * chunkCompress, latentLen]` and
    ///   `[bsz, 1, latentLen]` respectively.
    static func sampleNoisyLatent(
        durations: [Float],
        sampleRate: Int,
        baseChunkSize: Int,
        chunkCompress: Int,
        latentDim: Int,
        rng: () -> Float = { Float.random(in: 0..<1) }
    ) -> (noisyLatent: [Float], latentMask: [Float], dims: (bsz: Int, channels: Int, length: Int)) {
        let bsz = durations.count
        let maxDur = durations.max() ?? 0
        let wavLenMax = Int(maxDur * Float(sampleRate))
        let chunkSize = baseChunkSize * chunkCompress
        let latentLen = wavLenMax == 0 ? 0 : (wavLenMax + chunkSize - 1) / chunkSize
        let channels = latentDim * chunkCompress

        var noisyLatent = [Float](repeating: 0, count: bsz * channels * latentLen)
        for b in 0..<bsz {
            let perBatchOffset = b * channels * latentLen
            for c in 0..<channels {
                let rowOffset = perBatchOffset + c * latentLen
                for t in 0..<latentLen {
                    // Box-Muller: avoid log(0) by sampling u1 strictly > 0.
                    let u1 = max(rng(), 1e-4)
                    let u2 = rng()
                    let z = sqrt(-2.0 * log(u1)) * cos(2.0 * Float.pi * u2)
                    noisyLatent[rowOffset + t] = z
                }
            }
        }

        let wavLengths = durations.map { Int($0 * Float(sampleRate)) }
        let latentLengths = wavLengths.map { ($0 + chunkSize - 1) / chunkSize }
        let latentMask = mask(lengths: latentLengths, maxLen: latentLen)

        // Apply the mask to the noisy latent so padding positions stay zero.
        for b in 0..<bsz {
            let validLen = latentLengths[b]
            guard validLen < latentLen else { continue }
            for c in 0..<channels {
                let rowOffset = b * channels * latentLen + c * latentLen
                for t in validLen..<latentLen {
                    noisyLatent[rowOffset + t] = 0
                }
            }
        }

        return (noisyLatent, latentMask, (bsz, channels, latentLen))
    }

    /// `[bsz, 1, maxLen]` float mask flattened row-major.
    static func mask(lengths: [Int], maxLen: Int) -> [Float] {
        var out = [Float](repeating: 0, count: lengths.count * maxLen)
        for (b, len) in lengths.enumerated() {
            let base = b * maxLen
            for t in 0..<min(len, maxLen) {
                out[base + t] = 1
            }
        }
        return out
    }
}
