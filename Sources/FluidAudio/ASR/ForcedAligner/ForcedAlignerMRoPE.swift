import Foundation

/// Computes interleaved MRoPE (Multi-dimensional Rotary Position Embedding) for ForcedAligner.
///
/// Qwen3 ForcedAligner uses MRoPE with sections [24, 20, 20] (T, H, W) and interleaved layout.
/// For forced alignment (no spatial dimensions), all 3 grids use the same temporal positions,
/// but the interleaving pattern still applies.
///
/// Layout: inv_freq elements are assigned to T/H/W grids in an interleaved pattern
/// (every 3rd pair starting at offset 0/1/2), then cos/sin are computed and concatenated
/// to head_dim=128.
struct ForcedAlignerMRoPE {
    private let headDim: Int
    private let halfDim: Int
    private let invFreq: [Float]
    private let mropeSection: [Int]

    init() {
        self.headDim = ForcedAlignerConfig.headDim
        self.halfDim = ForcedAlignerConfig.headDim / 2
        self.mropeSection = ForcedAlignerConfig.mropeSection

        let theta = Float(ForcedAlignerConfig.ropeTheta)
        let dim = Float(ForcedAlignerConfig.headDim)

        var freq = [Float](repeating: 0.0, count: ForcedAlignerConfig.headDim / 2)
        for i in stride(from: 0, to: ForcedAlignerConfig.headDim, by: 2) {
            let exponent = Float(i) / dim
            freq[i / 2] = 1.0 / powf(theta, exponent)
        }
        self.invFreq = freq
    }

    /// Compute MRoPE cos/sin for the padded sequence.
    ///
    /// Position IDs follow `cumsum(attention_mask) - 1`: valid positions get [0, 1, ..., contentLen-1],
    /// padded positions all repeat `contentLen - 1`. This matches the Python reference.
    ///
    /// - Parameters:
    ///   - totalLen: Total padded sequence length (e.g. 1024).
    ///   - contentLen: Actual content length before padding.
    /// - Returns: Flat (cos, sin) arrays each of length totalLen * headDim,
    ///   laid out as [pos0(128), pos1(128), ...] for creating [1, totalLen, 128] tensors.
    func compute(totalLen: Int, contentLen: Int) -> (cos: [Float], sin: [Float]) {
        let totalSize = totalLen * headDim
        var cosValues = [Float](repeating: 0.0, count: totalSize)
        var sinValues = [Float](repeating: 0.0, count: totalSize)

        let lastValidPos = max(contentLen - 1, 0)

        for p in 0..<totalLen {
            // Padded positions repeat the last valid position
            let posId = min(p, lastValidPos)
            let pos = Float(posId)
            let offset = p * headDim
            for i in 0..<halfDim {
                let angle = pos * invFreq[i]
                let c = cosf(angle)
                let s = sinf(angle)
                // Concatenated-halves layout: [cos0,...,cos63, cos0,...,cos63]
                cosValues[offset + i] = c
                cosValues[offset + i + halfDim] = c
                sinValues[offset + i] = s
                sinValues[offset + i + halfDim] = s
            }
        }

        return (cos: cosValues, sin: sinValues)
    }
}
