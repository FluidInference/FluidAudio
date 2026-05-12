import Accelerate
import Foundation

/// HTK-mel spectrogram for StyleTTS2 reference audio.
///
/// Replicates `torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048,
/// win_length=1200, hop_length=300)` followed by the
/// `(log(mel + 1e-5) - (-4)) / 4` normalization from `make_preprocess`.
///
/// The training-time call to `MelSpectrogram` in `run_inference.py` does
/// **not** override `sample_rate`, so torchaudio falls back to its default
/// `16000`. The mel filterbank's bin centers were therefore computed for
/// 16 kHz audio — the model never saw 24 kHz-aware filters even though
/// the audio is loaded at 24 kHz. The extractor below replicates that
/// quirk so the mel bins line up with what the StyleTTS2 ref encoder
/// expects.
public final class StyleTTS2MelExtractor {

    // Cached config (from `StyleTTS2Constants`).
    private let nFFT: Int
    private let winLength: Int
    private let hopLength: Int
    private let nMels: Int
    private let mean: Float
    private let std: Float
    private let logEps: Float

    // Pre-computed.
    private let window: [Float]  // Length `nFFT`; zero-padded outside `winLength`.
    private let melFilterbankFlat: [Float]  // [nMels * (nFFT/2 + 1)] row-major.
    private let nFreqBins: Int  // `nFFT/2 + 1`.
    private let log2N: vDSP_Length  // `log2(nFFT)`.
    private let fftSetup: vDSP_DFT_Setup?

    public init(
        nFFT: Int = StyleTTS2Constants.melNFFT,
        winLength: Int = StyleTTS2Constants.melWinLength,
        hopLength: Int = StyleTTS2Constants.melHopLength,
        nMels: Int = StyleTTS2Constants.melNMels,
        filterSampleRate: Int = StyleTTS2Constants.melFilterSampleRate,
        mean: Float = StyleTTS2Constants.melLogMean,
        std: Float = StyleTTS2Constants.melLogStd,
        logEpsilon: Float = StyleTTS2Constants.melLogEpsilon
    ) {
        self.nFFT = nFFT
        self.winLength = winLength
        self.hopLength = hopLength
        self.nMels = nMels
        self.mean = mean
        self.std = std
        self.logEps = logEpsilon
        self.nFreqBins = nFFT / 2 + 1
        self.log2N = vDSP_Length(log2(Double(nFFT)).rounded())

        self.window = Self.hannWindowPadded(winLength: winLength, nFFT: nFFT)
        self.melFilterbankFlat = Self.htkMelFilterbank(
            nMels: nMels, nFFT: nFFT,
            sampleRate: filterSampleRate,
            fMin: 0.0, fMax: Float(filterSampleRate) / 2.0)

        // vDSP DFT setup. Real-input forward transform via the real-to-complex
        // packed format requires `kvDSP_DFT_FORWARD` over `nFFT` samples.
        // We do the full complex DFT on a real-imag-zeroed buffer for clarity.
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil, vDSP_Length(nFFT), .FORWARD)
    }

    deinit {
        if let fftSetup { vDSP_DFT_DestroySetup(fftSetup) }
    }

    // MARK: - Public API

    /// Compute the normalized log-mel spectrogram for the given waveform.
    /// Returns a flat row-major `[nMels * nFrames]` `Float` buffer plus the
    /// number of frames produced.
    public func compute(audio: [Float]) -> (mel: [Float], frames: Int) {
        // `center=True, pad_mode='reflect'` → pad by `nFFT/2` on each side.
        let pad = nFFT / 2
        let padded = reflectPad(audio, pad: pad)

        // `nFrames = 1 + (len(padded) - nFFT) / hopLength` for valid frames.
        let usable = padded.count - nFFT
        let frames = usable >= 0 ? (usable / hopLength) + 1 : 0
        if frames == 0 {
            return ([], 0)
        }

        var mel = [Float](repeating: 0, count: nMels * frames)
        var frame = [Float](repeating: 0, count: nFFT)
        var realIn = [Float](repeating: 0, count: nFFT)
        var imagIn = [Float](repeating: 0, count: nFFT)
        var realOut = [Float](repeating: 0, count: nFFT)
        var imagOut = [Float](repeating: 0, count: nFFT)
        var power = [Float](repeating: 0, count: nFreqBins)

        for f in 0..<frames {
            let start = f * hopLength
            // Window (already zero-padded to nFFT — see hannWindowPadded).
            padded.withUnsafeBufferPointer { src in
                frame.withUnsafeMutableBufferPointer { dst in
                    let srcPtr = src.baseAddress!.advanced(by: start)
                    let dstPtr = dst.baseAddress!
                    vDSP_vmul(srcPtr, 1, window, 1, dstPtr, 1, vDSP_Length(nFFT))
                }
            }

            // Real → complex (imag stays zero).
            for i in 0..<nFFT {
                realIn[i] = frame[i]
                imagIn[i] = 0
            }

            if let fftSetup {
                vDSP_DFT_Execute(fftSetup, realIn, imagIn, &realOut, &imagOut)
            }

            // Power spectrum: |X|² for the first `nFFT/2 + 1` bins.
            for k in 0..<nFreqBins {
                let re = realOut[k]
                let im = imagOut[k]
                power[k] = re * re + im * im
            }

            // mel = filterbank @ power → write directly into `mel[:, f]`
            // (column f of the [nMels, nFrames] output).
            for m in 0..<nMels {
                var acc: Float = 0
                let rowOffset = m * nFreqBins
                vDSP_dotpr(
                    melFilterbankFlat.withUnsafeBufferPointer {
                        $0.baseAddress!.advanced(by: rowOffset)
                    },
                    1, power, 1, &acc, vDSP_Length(nFreqBins))
                // `(log(mel + eps) - mean) / std` — fused in-place.
                mel[m * frames + f] = (log(acc + logEps) - mean) / std
            }
        }

        return (mel, frames)
    }

    // MARK: - Static helpers

    /// Periodic hann window of length `winLength`, zero-padded into a
    /// length-`nFFT` buffer with the window centered. Mirrors
    /// torchaudio's centered placement when `win_length < n_fft`.
    private static func hannWindowPadded(winLength: Int, nFFT: Int) -> [Float] {
        var w = [Float](repeating: 0, count: nFFT)
        let twoPi = Float.pi * 2.0
        let denom = Float(winLength)
        let pad = (nFFT - winLength) / 2
        for n in 0..<winLength {
            // periodic hann: 0.5 * (1 - cos(2π n / N))
            w[pad + n] = 0.5 * (1.0 - cos(twoPi * Float(n) / denom))
        }
        return w
    }

    /// HTK mel scale: `mel(f) = 2595 * log10(1 + f / 700)`.
    @inline(__always)
    private static func hzToMel(_ hz: Float) -> Float {
        return 2595.0 * log10(1.0 + hz / 700.0)
    }

    @inline(__always)
    private static func melToHz(_ mel: Float) -> Float {
        return 700.0 * (powf(10.0, mel / 2595.0) - 1.0)
    }

    /// Build a row-major `[nMels, nFFT/2 + 1]` filterbank using the HTK mel
    /// scale and triangular filters with **no normalization** (matches
    /// torchaudio's `norm=None`).
    private static func htkMelFilterbank(
        nMels: Int, nFFT: Int, sampleRate: Int, fMin: Float, fMax: Float
    ) -> [Float] {
        let nFreqBins = nFFT / 2 + 1
        var fb = [Float](repeating: 0, count: nMels * nFreqBins)

        // Bin frequency for each FFT output: k * sr / nFFT.
        var binFreqs = [Float](repeating: 0, count: nFreqBins)
        let binStep = Float(sampleRate) / Float(nFFT)
        for k in 0..<nFreqBins {
            binFreqs[k] = Float(k) * binStep
        }

        // nMels + 2 mel points equally spaced from mel(fMin) to mel(fMax).
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            let frac = Float(i) / Float(nMels + 1)
            melPoints[i] = melMin + (melMax - melMin) * frac
        }
        let hzPoints = melPoints.map(melToHz)

        // Triangular filters between consecutive (left, center, right) hz points.
        for m in 0..<nMels {
            let left = hzPoints[m]
            let center = hzPoints[m + 1]
            let right = hzPoints[m + 2]
            let leftSlope = center - left
            let rightSlope = right - center
            let rowOffset = m * nFreqBins
            for k in 0..<nFreqBins {
                let f = binFreqs[k]
                if f < left || f > right {
                    continue
                }
                let val: Float
                if f <= center {
                    val = leftSlope > 0 ? (f - left) / leftSlope : 0
                } else {
                    val = rightSlope > 0 ? (right - f) / rightSlope : 0
                }
                fb[rowOffset + k] = max(val, 0)
            }
        }

        return fb
    }

    /// Reflect padding without reflecting across the boundary sample
    /// (matches NumPy / torch `mode='reflect'`). E.g. for `[a, b, c, d]`
    /// with `pad=2` returns `[c, b, a, b, c, d, c, b]`.
    private func reflectPad(_ x: [Float], pad: Int) -> [Float] {
        if pad == 0 { return x }
        if x.isEmpty { return [Float](repeating: 0, count: pad * 2) }
        let n = x.count
        var out = [Float](repeating: 0, count: n + 2 * pad)
        // Front pad.
        for i in 0..<pad {
            // Index `pad - i` reflected across position 0 → `i`.
            // Reflect: idx = pad - i (so first sample is x[pad], last is x[1]).
            // Boundary not reflected, hence the +1 below to skip x[0] for i = pad.
            let mirror = min(pad - i, n - 1)
            out[i] = x[mirror]
        }
        // Body.
        for i in 0..<n {
            out[pad + i] = x[i]
        }
        // Back pad: reflect from the end without re-emitting x[n-1].
        for i in 0..<pad {
            // idx = n - 2 - i, clamped to 0.
            let mirror = max(n - 2 - i, 0)
            out[pad + n + i] = x[mirror]
        }
        return out
    }
}
