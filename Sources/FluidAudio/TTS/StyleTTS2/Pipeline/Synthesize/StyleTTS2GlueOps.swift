import Accelerate
import CoreML
import Foundation

/// CPU-side tensor ops that bridge between the StyleTTS2 CoreML stages.
///
/// Mirrors the eager-glue helpers in
/// `mobius/models/tts/styletts2/coreml/inference.py`:
///   - `_build_pred_aln_trg`: rounds duration_predictor logits to integer
///     frame counts and emits the one-hot `[real_n, real_frames]` alignment
///     matrix that gets matmul'd against `t_en` and `d`.
///   - `_hifigan_shift`: causal one-frame shift on the 3D `[1, C, F]` tensors
///     fed into the HiFi-GAN decoder.
///   - alpha/beta style blend: `s_pred[:, 128:]` and `s_pred[:, :128]`
///     blended with the corresponding halves of `ref_s`.
public enum StyleTTS2GlueOps {

    // MARK: - Duration → predicted alignment matrix

    /// Round + clamp `pred_dur` (sigmoid-summed logits, shape `[real_n]`)
    /// to integer frame counts ≥ 1. The returned `[real_n]` array sums to
    /// the number of frames the rest of the pipeline runs at.
    public static func roundDurations(_ logits: MLMultiArray) throws -> [Int] {
        // Expected shape: `[1, real_n, channels]` from `duration_predictor`.
        guard logits.shape.count == 3 else {
            throw StyleTTS2Error.invalidTensorShape(
                stage: "duration_logits",
                expected: "[1, T, C]",
                got: "\(logits.shape)")
        }
        let realN = logits.shape[1].intValue
        let channels = logits.shape[2].intValue

        var durations = [Int](repeating: 0, count: realN)
        // sigmoid(x) summed across the channel axis, then round + clamp(min: 1).
        for t in 0..<realN {
            var sum: Float = 0
            for c in 0..<channels {
                let x = logits[[0, t, c] as [NSNumber]].floatValue
                sum += 1.0 / (1.0 + expf(-x))
            }
            let rounded = Int(sum.rounded(.toNearestOrAwayFromZero))
            durations[t] = max(rounded, 1)
        }
        return durations
    }

    /// Build the one-hot `[real_n, total_frames]` alignment matrix from the
    /// integer durations. Each row `i` has `durations[i]` consecutive 1s
    /// starting at the prefix sum of preceding durations.
    public static func buildAlignmentMatrix(durations: [Int]) -> (matrix: [Float], totalFrames: Int) {
        let realN = durations.count
        let totalFrames = durations.reduce(0, +)
        var matrix = [Float](repeating: 0, count: realN * totalFrames)
        var col = 0
        for i in 0..<realN {
            let d = durations[i]
            for k in 0..<d {
                matrix[i * totalFrames + col + k] = 1
            }
            col += d
        }
        return (matrix, totalFrames)
    }

    // MARK: - en / asr matmuls

    /// Compute `out = features^T @ alignment` where `features` is row-major
    /// `[1, C, real_n]` and `alignment` is `[real_n, total_frames]`. Returns
    /// row-major `[1, C, total_frames]` matching `d.transpose(-1, -2) @ aln`
    /// (with `d` from duration_predictor, transposed to `[1, real_n, C]`,
    /// then re-transposed to `[1, C, real_n]` before matmul).
    public static func matmulAligned(
        features: [Float],
        channels: Int,
        realN: Int,
        alignment: [Float],
        totalFrames: Int
    ) -> [Float] {
        precondition(features.count == channels * realN)
        precondition(alignment.count == realN * totalFrames)

        var out = [Float](repeating: 0, count: channels * totalFrames)
        // `out[c, f] = Σ_i features[c, i] * alignment[i, f]`
        // Use BLAS sgemm: features is [C, K], alignment is [K, F] → out is [C, F].
        // K = realN.
        out.withUnsafeMutableBufferPointer { outPtr in
            features.withUnsafeBufferPointer { fPtr in
                alignment.withUnsafeBufferPointer { aPtr in
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        Int32(channels), Int32(totalFrames), Int32(realN),
                        1.0,
                        fPtr.baseAddress, Int32(realN),
                        aPtr.baseAddress, Int32(totalFrames),
                        0.0,
                        outPtr.baseAddress, Int32(totalFrames))
                }
            }
        }
        return out
    }

    /// Transpose a row-major `[B, C, T]` tensor along the last two axes,
    /// yielding row-major `[B, T, C]`. `B` is always 1 in the pipeline so
    /// we elide the batch loop.
    public static func transposeLast2D(_ src: [Float], rows: Int, cols: Int) -> [Float] {
        precondition(src.count == rows * cols)
        var out = [Float](repeating: 0, count: cols * rows)
        vDSP_mtrans(src, 1, &out, 1, vDSP_Length(cols), vDSP_Length(rows))
        return out
    }

    // MARK: - HiFi-GAN causal asr-shift

    /// In-place causal shift on a flat `[1, C, F]` row-major tensor:
    ///     `out[:, :, 0] = in[:, :, 0]`
    ///     `out[:, :, 1:] = in[:, :, :-1]`
    /// Mirrors `_hifigan_shift` in the Python orchestrator (which copies into
    /// a fresh tensor, so we keep the explicit copy here).
    public static func hifiganShift(_ x: [Float], channels: Int, frames: Int) -> [Float] {
        precondition(x.count == channels * frames)
        var out = [Float](repeating: 0, count: x.count)
        for c in 0..<channels {
            let inRow = c * frames
            // Column 0 stays put.
            out[inRow] = x[inRow]
            // Columns 1..F-1 are shifted right by 1 from columns 0..F-2.
            for f in 1..<frames {
                out[inRow + f] = x[inRow + f - 1]
            }
        }
        return out
    }

    // MARK: - α/β style blend

    /// Mirror the Python:
    ///   `s_diff   = s_pred[:, 128:]`
    ///   `ref_diff = s_pred[:, :128]`
    ///   `ref      = α * ref_diff + (1 - α) * ref_s[:, :128]`
    ///   `s        = β * s_diff   + (1 - β) * ref_s[:, 128:]`
    ///
    /// Inputs are flat 256-element buffers (batch 1, last dim 256).
    /// Returns `(ref128, s128)` — each 128-element flat buffers ready to feed
    /// the duration predictor / decoder_pre.
    public static func blendStyle(
        sPred256: [Float], refS256: [Float], alpha: Float, beta: Float
    ) -> (ref128: [Float], s128: [Float]) {
        precondition(sPred256.count == 256)
        precondition(refS256.count == 256)
        var ref = [Float](repeating: 0, count: 128)
        var s = [Float](repeating: 0, count: 128)
        let oneMinusAlpha = 1.0 - alpha
        let oneMinusBeta = 1.0 - beta
        for i in 0..<128 {
            ref[i] = alpha * sPred256[i] + oneMinusAlpha * refS256[i]
            s[i] = beta * sPred256[128 + i] + oneMinusBeta * refS256[128 + i]
        }
        return (ref, s)
    }
}
