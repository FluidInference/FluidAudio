import Accelerate
import Foundation

// MARK: - Rotary Position Embeddings for Qwen3-ASR

/// Computes RoPE (Rotary Position Embedding) cos/sin values for the Qwen3 decoder.
///
/// Qwen3-ASR uses M-RoPE (Multi-dimensional Rotary Position Embedding) with
/// interleaved sections [24, 20, 20] and rope_theta = 1,000,000.
/// For ASR (no spatial dimensions), temporal position is used for all 3 sections,
/// so this simplifies to standard RoPE with the full head_dim = 128.
public struct Qwen3RoPE {
    private let invFreq: [Float]
    public let headDim: Int

    /// Initialize with Qwen3-ASR config.
    ///
    /// inv_freq = 1.0 / (theta ^ (i / dim)) for i in [0, 2, 4, ..., dim-2]
    public init(config: Qwen3AsrConfig = .default) {
        self.headDim = config.headDim
        let theta = Float(config.ropeTheta)
        let dim = Float(config.headDim)

        var freq = [Float](repeating: 0.0, count: config.headDim / 2)
        for i in stride(from: 0, to: config.headDim, by: 2) {
            let exponent = Float(i) / dim
            freq[i / 2] = 1.0 / powf(theta, exponent)
        }
        self.invFreq = freq
    }

    /// Compute cos and sin embeddings for a given position.
    ///
    /// Returns (cos, sin) each of shape [headDim], suitable for creating
    /// CoreML input tensors of shape [1, 1, headDim].
    public func compute(position: Int) -> (cos: [Float], sin: [Float]) {
        let pos = Float(position)
        var cosValues = [Float](repeating: 0.0, count: headDim)
        var sinValues = [Float](repeating: 0.0, count: headDim)

        let halfDim = headDim / 2
        for i in 0..<halfDim {
            let angle = pos * invFreq[i]
            let c = cosf(angle)
            let s = sinf(angle)
            // Concatenated-halves layout: [cos0,cos1,...,cos63, cos0,cos1,...,cos63]
            // This matches the CoreML model's rotate_half which splits at head_dim/2.
            // Same layout as Python: np.concatenate([freqs, freqs])
            cosValues[i] = c
            cosValues[i + halfDim] = c
            sinValues[i] = s
            sinValues[i + halfDim] = s
        }

        return (cos: cosValues, sin: sinValues)
    }

    /// Compute cos and sin embeddings for a contiguous range of positions.
    ///
    /// Returns flat arrays of length `count * headDim`, laid out as
    /// `[pos0(128), pos1(128), ...]` for creating `[1, count, headDim]` tensors.
    /// Used for batched prefill where all prompt positions are processed at once.
    public func computeRange(startPosition: Int, count: Int) -> (cos: [Float], sin: [Float]) {
        let totalSize = count * headDim
        var cosValues = [Float](repeating: 0.0, count: totalSize)
        var sinValues = [Float](repeating: 0.0, count: totalSize)

        let halfDim = headDim / 2
        for p in 0..<count {
            let pos = Float(startPosition + p)
            let offset = p * headDim
            for i in 0..<halfDim {
                let angle = pos * invFreq[i]
                let c = cosf(angle)
                let s = sinf(angle)
                cosValues[offset + i] = c
                cosValues[offset + i + halfDim] = c
                sinValues[offset + i] = s
                sinValues[offset + i + halfDim] = s
            }
        }

        return (cos: cosValues, sin: sinValues)
    }
}
