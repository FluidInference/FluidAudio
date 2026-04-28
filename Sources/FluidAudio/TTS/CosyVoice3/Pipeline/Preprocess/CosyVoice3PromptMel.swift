import Accelerate
import Foundation

/// On-device mel spectrogram extractor for CosyVoice3 prompt audio.
///
/// Matches `matcha.utils.audio.mel_spectrogram` invoked from
/// `cosyvoice/cli/frontend.py:_extract_speech_feat` with the CosyVoice3 config
/// (see `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml`):
///
/// ```
/// n_fft: 1920
/// num_mels: 80
/// sampling_rate: 24000
/// hop_size: 480
/// win_size: 1920
/// fmin: 0
/// fmax: null  (→ sampling_rate / 2 = 12000 per librosa default)
/// center: False
/// ```
///
/// Pipeline (verbatim from the Python reference):
///   1. reflect-pad the waveform by `(n_fft - hop_size) / 2 = 720` on each side
///   2. framed STFT with `n_fft=1920, hop=480, win=1920`, periodic Hann window
///      (`torch.hann_window` default), `center=False`
///   3. magnitude = `sqrt(real² + imag² + 1e-9)`   (Matcha convention)
///   4. `mel = mel_basis @ magnitude` using Slaney-normalized mel filterbank
///      (librosa default: HTK=False, norm='slaney')
///   5. `log_mel = log(clamp(mel, min=1e-5))`
///
/// The output is flattened `[T, 80]` row-major fp32, which is the layout
/// `CosyVoice3PromptAssets.promptMel` stores and the Flow model consumes as
/// `[1, 2*N_speech, 80]` after slicing to match the prompt-speech id count.
///
/// Use `trimToTokenRatio(...)` to enforce the `frames == 2 * N_speech`
/// invariant before passing to Flow (matches the
/// `speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len`
/// clamp in the Python frontend).
public final class CosyVoice3PromptMel {

    public static let sampleRate = 24_000
    public static let nFFT = 1_920
    public static let hopSize = 480
    public static let winSize = 1_920
    public static let numMels = 80
    public static let fMin: Float = 0
    public static let fMax: Float = 12_000  // sr / 2
    /// Reflect-pad each side by `(n_fft - hop_size) / 2`.
    public static let padLength = (nFFT - hopSize) / 2  // 720
    /// Magnitude epsilon before sqrt (prevents NaN gradients in training; kept
    /// here for bit parity with the reference).
    private static let magEps: Float = 1e-9
    /// Log floor clamp applied inside `log(clamp(x, min=1e-5))`.
    private static let logFloor: Float = 1e-5

    // Precomputed resources
    private let hannWindow: [Float]
    private let melBasis: [Float]  // flat [numMels * numFreqBins]
    private let numFreqBins: Int
    private var fftSetup: vDSP_DFT_Setup?

    // Reusable buffers (not thread-safe; wrap with a queue if shared).
    private var frameBuf: [Float]
    private var realIn: [Float]
    private var imagIn: [Float]
    private var realOut: [Float]
    private var imagOut: [Float]
    private var magnitude: [Float]
    private var imagSq: [Float]

    public init() {
        self.numFreqBins = Self.nFFT / 2 + 1
        // torch.hann_window(N) defaults to periodic=True — sample i of length
        // N is `0.5 * (1 - cos(2πi/N))`. This matches Matcha's code path via
        // the torch.stft default.
        self.hannWindow = Self.hannWindowPeriodic(length: Self.winSize)
        self.melBasis = Self.buildSlaneyMelBasis(
            sampleRate: Self.sampleRate,
            nFFT: Self.nFFT,
            numMels: Self.numMels,
            fMin: Self.fMin,
            fMax: Self.fMax)
        self.fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(Self.nFFT), .FORWARD)
        self.frameBuf = [Float](repeating: 0, count: Self.nFFT)
        self.realIn = [Float](repeating: 0, count: Self.nFFT)
        self.imagIn = [Float](repeating: 0, count: Self.nFFT)
        self.realOut = [Float](repeating: 0, count: Self.nFFT)
        self.imagOut = [Float](repeating: 0, count: Self.nFFT)
        self.magnitude = [Float](repeating: 0, count: numFreqBins)
        self.imagSq = [Float](repeating: 0, count: numFreqBins)
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    public struct Result: Sendable {
        /// `[frames * numMels]` row-major, fp32.
        public let mel: [Float]
        public let frames: Int
    }

    /// Compute the log-mel spectrogram for a 24 kHz mono waveform.
    ///
    /// - Parameter audio: fp32 PCM samples at 24 kHz, range ≈ [-1, 1].
    /// - Returns: `[T * 80]` row-major fp32 mel, where
    ///   `T = floor((len + 2·padLength - nFFT) / hopSize) + 1`.
    public func compute(audio: [Float]) throws -> Result {
        guard let setup = fftSetup else {
            throw CosyVoice3Error.invalidShape("vDSP_DFT setup failed")
        }
        guard audio.count > 0 else {
            return Result(mel: [], frames: 0)
        }

        let padded = Self.reflectPad(audio, pad: Self.padLength)
        let paddedCount = padded.count
        let frames = max(0, (paddedCount - Self.nFFT) / Self.hopSize + 1)
        guard frames > 0 else {
            return Result(mel: [], frames: 0)
        }

        var mel = [Float](repeating: 0, count: frames * Self.numMels)

        for frameIdx in 0..<frames {
            let start = frameIdx * Self.hopSize

            // Window the frame: frameBuf[i] = padded[start+i] * hann[i].
            padded.withUnsafeBufferPointer { paddedPtr in
                hannWindow.withUnsafeBufferPointer { hannPtr in
                    frameBuf.withUnsafeMutableBufferPointer { fPtr in
                        vDSP_vmul(
                            paddedPtr.baseAddress! + start, 1,
                            hannPtr.baseAddress!, 1,
                            fPtr.baseAddress!, 1,
                            vDSP_Length(Self.winSize))
                    }
                }
            }

            // FFT. realIn ← frameBuf, imagIn ← 0.
            frameBuf.withUnsafeBufferPointer { src in
                realIn.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!, src.baseAddress!, Self.nFFT * MemoryLayout<Float>.size)
                }
            }
            vDSP_vclr(&imagIn, 1, vDSP_Length(Self.nFFT))
            vDSP_DFT_Execute(setup, realIn, imagIn, &realOut, &imagOut)

            // magnitude = sqrt(real² + imag² + 1e-9) over one-sided bins.
            vDSP_vsq(realOut, 1, &magnitude, 1, vDSP_Length(numFreqBins))
            vDSP_vsq(imagOut, 1, &imagSq, 1, vDSP_Length(numFreqBins))
            vDSP_vadd(magnitude, 1, imagSq, 1, &magnitude, 1, vDSP_Length(numFreqBins))
            var eps = Self.magEps
            vDSP_vsadd(magnitude, 1, &eps, &magnitude, 1, vDSP_Length(numFreqBins))
            var n = Int32(numFreqBins)
            vvsqrtf(&magnitude, magnitude, &n)

            // mel = melBasis[80, numFreqBins] @ magnitude[numFreqBins]
            var melFrame = [Float](repeating: 0, count: Self.numMels)
            melBasis.withUnsafeBufferPointer { basisPtr in
                magnitude.withUnsafeBufferPointer { magPtr in
                    melFrame.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            basisPtr.baseAddress!, 1,
                            magPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(Self.numMels),
                            vDSP_Length(1),
                            vDSP_Length(numFreqBins))
                    }
                }
            }

            // log(clamp(x, min=1e-5))
            for m in 0..<Self.numMels {
                let clamped = max(melFrame[m], Self.logFloor)
                mel[frameIdx * Self.numMels + m] = log(clamped)
            }
        }

        return Result(mel: mel, frames: frames)
    }

    /// Enforce `frames == 2 * tokenCount`. Trims excess frames if needed; if
    /// the mel is shorter than `2 * tokenCount`, an error is thrown (callers
    /// should ensure the prompt WAV is long enough for its token count).
    public static func trimToTokenRatio(
        mel: [Float], frames: Int, tokenCount: Int
    ) throws -> (mel: [Float], frames: Int) {
        let targetFrames = 2 * tokenCount
        guard frames >= targetFrames else {
            throw CosyVoice3Error.invalidShape(
                "prompt mel has \(frames) frames but tokenCount=\(tokenCount) requires \(targetFrames)"
            )
        }
        if frames == targetFrames {
            return (mel, frames)
        }
        let trimmed = Array(mel.prefix(targetFrames * numMels))
        return (trimmed, targetFrames)
    }

    // MARK: - Helpers

    /// PyTorch `F.pad(..., mode="reflect")` on a 1-D signal:
    ///   - left:  [y[pad], y[pad-1], ..., y[1]]
    ///   - core:  y[0..<N]
    ///   - right: [y[N-2], y[N-3], ..., y[N-1-pad]]
    /// Reflection excludes the endpoint (matches librosa / numpy reflect).
    static func reflectPad(_ y: [Float], pad: Int) -> [Float] {
        let n = y.count
        if pad <= 0 { return y }
        // PyTorch requires pad < n for reflect. Guard loudly for a silently
        // bad prompt (very short audio).
        precondition(pad < n, "reflect pad=\(pad) requires signal length > \(pad), got \(n)")
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad {
            out[i] = y[pad - i]
        }
        for i in 0..<n {
            out[pad + i] = y[i]
        }
        for i in 0..<pad {
            out[pad + n + i] = y[n - 2 - i]
        }
        return out
    }

    /// `torch.hann_window(N)` (periodic=True): sample i of length N is
    /// `0.5 * (1 - cos(2πi / N))`.
    static func hannWindowPeriodic(length: Int) -> [Float] {
        var w = [Float](repeating: 0, count: length)
        let divisor = Float(length)
        for i in 0..<length {
            w[i] = 0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / divisor))
        }
        return w
    }

    /// Build a `[numMels, numFFT/2 + 1]` row-major mel filterbank matching
    /// `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax)` defaults:
    /// HTK=False (Slaney mel), norm='slaney' (triangle area = 2/(f_right−f_left)).
    static func buildSlaneyMelBasis(
        sampleRate: Int,
        nFFT: Int,
        numMels: Int,
        fMin: Float,
        fMax: Float
    ) -> [Float] {
        let numFreqBins = nFFT / 2 + 1

        let melMin = hzToMelSlaney(fMin)
        let melMax = hzToMelSlaney(fMax)

        var melPoints = [Float](repeating: 0, count: numMels + 2)
        for i in 0..<(numMels + 2) {
            let mel = melMin + Float(i) * (melMax - melMin) / Float(numMels + 1)
            melPoints[i] = melToHzSlaney(mel)
        }

        var fftFreqs = [Float](repeating: 0, count: numFreqBins)
        for i in 0..<numFreqBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        var basis = [Float](repeating: 0, count: numMels * numFreqBins)
        for m in 0..<numMels {
            let fLeft = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fRight = melPoints[m + 2]
            let norm = 2.0 / (fRight - fLeft)
            for f in 0..<numFreqBins {
                let freq = fftFreqs[f]
                var w: Float = 0
                if freq >= fLeft && freq < fCenter {
                    w = norm * (freq - fLeft) / (fCenter - fLeft)
                } else if freq >= fCenter && freq <= fRight {
                    w = norm * (fRight - freq) / (fRight - fCenter)
                }
                basis[m * numFreqBins + f] = w
            }
        }
        return basis
    }

    static func hzToMelSlaney(_ hz: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000.0
        let minLogMel: Float = minLogHz / fSp
        let logStep: Float = log(6.4) / 27.0
        return hz >= minLogHz
            ? minLogMel + log(hz / minLogHz) / logStep
            : hz / fSp
    }

    static func melToHzSlaney(_ mel: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1_000.0
        let minLogMel: Float = minLogHz / fSp
        let logStep: Float = log(6.4) / 27.0
        return mel >= minLogMel
            ? minLogHz * exp(logStep * (mel - minLogMel))
            : fSp * mel
    }
}
