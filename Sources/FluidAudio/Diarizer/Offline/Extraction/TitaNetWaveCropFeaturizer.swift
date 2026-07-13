import Accelerate
import Foundation

/// Per-crop waveform featurizer for the TitaNet `.cleanWaveform` pooling path.
///
/// This is the reference-faithful fix for the `.cleanRegion` over-split. Instead
/// of cropping full-window MEL columns (which carry the full-window per-bin
/// mean/std normalization and whose boundary frames were computed by STFT windows
/// that bled into the overlapping speaker), it re-featurizes each clean *waveform*
/// crop on its own, so:
///   • normalization statistics come from clean audio only, and
///   • no STFT window ever spans cross-talk (each crop is padded from its own
///     samples, not the window's).
///
/// The DSP is delegated to the already-validated `AudioMelSpectrogram` (preemph
/// off; we apply preemph + reflect padding ourselves so the padding matches
/// librosa's `center=True, pad_mode="reflect"` exactly). The pipeline mirrors the
/// Python reference `_infer_features_concat` / the `featurizer.py` prototype:
///
///   preemph(0.97) → reflect-pad n_fft/2 → STFT(512/160/400, periodic hann) →
///   power → Slaney mel(80) → log(+2^-24) → per-mel-bin mean/std norm →
///   drop `joinFrameDrop` edge frames per crop.
///
/// Output is a mel-major `[melBins, T']` flat buffer plus `T'`, ready to be
/// concatenated across crops in the feature domain.
@available(macOS 14.0, iOS 17.0, *)
struct TitaNetWaveCropFeaturizer {
    static let melBins = 80
    static let nFFT = 512
    static let hop = 160
    static let win = 400
    static let preemph: Float = 0.97
    /// Edge mel frames dropped per crop (reflect-pad / co-articulation artifact),
    /// matching the reference `join_frame_drop = 1`.
    static let joinFrameDrop = 1

    private let mel: AudioMelSpectrogram

    init() {
        // preemph: 0 — we apply preemphasis + reflect padding ourselves and feed
        // the result in `.prePadded` mode, so the model must not pad or preemph
        // again. windowPeriodic: true matches librosa's default `fftbins=True`
        // hann. logFloor 2^-24 additive matches the front / featurizer.py.
        mel = AudioMelSpectrogram(
            sampleRate: 16_000,
            nMels: Self.melBins,
            nFFT: Self.nFFT,
            hopLength: Self.hop,
            winLength: Self.win,
            preemph: 0,
            padTo: 0,
            logFloor: powf(2, -24),
            logFloorMode: .additive,
            windowPeriodic: true
        )
    }

    /// Featurize one clean waveform crop.
    /// - Returns: mel-major `[melBins * frames]` (per-bin normalized, edge-dropped)
    ///   and the surviving frame count, or `nil` if the crop is too short.
    func featurize(crop: [Float]) -> (feats: [Float], frames: Int)? {
        guard crop.count >= Self.win else { return nil }

        // 1. Preemphasis: pre[0] = crop[0]; pre[i] = crop[i] - 0.97 * crop[i-1].
        var pre = [Float](repeating: 0, count: crop.count)
        pre[0] = crop[0]
        if crop.count > 1 {
            crop.withUnsafeBufferPointer { src in
                pre.withUnsafeMutableBufferPointer { dst in
                    var neg = -Self.preemph
                    vDSP_vsma(
                        src.baseAddress!, 1,
                        &neg,
                        src.baseAddress! + 1, 1,
                        dst.baseAddress! + 1, 1,
                        vDSP_Length(crop.count - 1))
                }
            }
        }

        // 2. Reflect-pad the pre-emphasized signal by n_fft/2 (numpy 'reflect':
        // mirror around the edge sample without repeating it), matching
        // librosa.stft(center=True, pad_mode="reflect"). Feeding `.prePadded`
        // then reproduces librosa's framing exactly.
        let padded = Self.reflectPad(pre, Self.nFFT / 2)

        // 3. STFT → power → mel → log. Frame-major `[T * melBins]`.
        let (flatTM, frameCount, _) = mel.computeFlatTransposed(
            audio: padded, lastAudioSample: 0, paddingMode: .prePadded, expectedFrameCount: nil)
        let t = frameCount
        guard t > 0 else { return nil }

        // 4. To mel-major `[melBins, T]` with per-mel-bin mean/std normalization
        // (over this crop's frames only — the whole point of waveform cropping).
        var mm = [Float](repeating: 0, count: Self.melBins * t)
        for m in 0..<Self.melBins {
            var mean: Float = 0
            for frame in 0..<t { mean += flatTM[frame * Self.melBins + m] }
            mean /= Float(t)
            var variance: Float = 0
            for frame in 0..<t {
                let d = flatTM[frame * Self.melBins + m] - mean
                variance += d * d
            }
            let std = (variance / Float(t)).squareRoot()
            let inv = 1.0 / (std + 1e-5)
            let row = m * t
            for frame in 0..<t { mm[row + frame] = (flatTM[frame * Self.melBins + m] - mean) * inv }
        }

        // 5. Join-frame-drop: drop the first/last `joinFrameDrop` frames (only if
        // the crop is long enough to keep at least one frame).
        let d = Self.joinFrameDrop
        guard t > 2 * d else { return (mm, t) }
        let t2 = t - 2 * d
        var out = [Float](repeating: 0, count: Self.melBins * t2)
        for m in 0..<Self.melBins {
            let srcRow = m * t + d
            let dstRow = m * t2
            for frame in 0..<t2 { out[dstRow + frame] = mm[srcRow + frame] }
        }
        return (out, t2)
    }

    /// numpy 'reflect' padding of width `p` (mirror around the edge sample, no
    /// edge repeat). Callers only pass crops longer than `p`, so a single
    /// reflection suffices; the index clamps are defensive.
    static func reflectPad(_ x: [Float], _ p: Int) -> [Float] {
        let n = x.count
        guard n > 1, p > 0 else { return x }
        var out = [Float]()
        out.reserveCapacity(n + 2 * p)
        // left: x[p], x[p-1], …, x[1]
        for i in stride(from: p, through: 1, by: -1) { out.append(x[min(i, n - 1)]) }
        out.append(contentsOf: x)
        // right: x[n-2], x[n-3], …, x[n-1-p]
        for i in stride(from: n - 2, through: n - 1 - p, by: -1) { out.append(x[max(i, 0)]) }
        return out
    }
}
