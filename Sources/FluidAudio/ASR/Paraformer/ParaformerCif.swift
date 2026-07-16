import Foundation

/// Host-side CIF (Continuous Integrate-and-Fire) for Paraformer.
///
/// The conv1d+linear+sigmoid that produces per-frame `alphas` is the CoreML
/// `ParaformerCifAlphas` model; this does only the integrate-and-fire that turns
/// (encoder rows, alphas) into a dynamic number of acoustic-embedding tokens.
/// Port of the numpy reference (`cif_numpy.py`), bit-exact vs FunASR.
enum ParaformerCif {

    /// - Parameters:
    ///   - encRows: encoder output rows `[T][D]` (real frames only).
    ///   - alphas: per-frame weights `[T]` from the CifAlphas model.
    /// - Returns: acoustic embeddings `[L][D]`.
    /// CIF integrate-and-fire that also records the encoder-frame index at which
    /// each acoustic token "fires" (the CIF boundary). The returned `fireFrames`
    /// is the Swift port of FunASR's `pre_peak_index` and is consumed by the
    /// timestamp routine (`ts_prediction_lfr6_standard`) to map tokens to time.
    static func integrateAndFireWithFireFrames(
        encRows: [[Float]], alphas: [Float]
    ) -> (embeds: [[Float]], fireFrames: [Int]) {
        let threshold = ParaformerConfig.cifThreshold
        let tail = ParaformerConfig.cifTailThreshold
        let T = encRows.count
        let dim = encRows.first?.count ?? ParaformerConfig.encoderDim

        var embeds: [[Float]] = []
        var fireFrames: [Int] = []
        var integrate: Float = 0
        var frame = [Float](repeating: 0, count: dim)

        // T real frames + 1 tail frame (alpha = tail_threshold, hidden = zeros)
        for t in 0...T {
            let alpha = (t < T) ? alphas[t] : tail
            let hidden = (t < T) ? encRows[t] : [Float](repeating: 0, count: dim)
            integrate += alpha
            if integrate < threshold {
                for d in 0..<dim { frame[d] += alpha * hidden[d] }
            } else {
                let used = alpha - (integrate - threshold)  // portion to reach threshold
                for d in 0..<dim { frame[d] += used * hidden[d] }
                embeds.append(frame)
                fireFrames.append(t)
                integrate -= threshold
                let leftover = alpha - used
                frame = hidden.map { $0 * leftover }  // leftover seeds next token
            }
        }
        return (embeds, fireFrames)
    }

    /// Original integrate-and-fire entry point (embeds only).
    static func integrateAndFire(encRows: [[Float]], alphas: [Float]) -> [[Float]] {
        integrateAndFireWithFireFrames(encRows: encRows, alphas: alphas).embeds
    }
}
