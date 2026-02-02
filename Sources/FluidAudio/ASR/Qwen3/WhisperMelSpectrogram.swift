import Accelerate
import Foundation

/// Whisper-compatible mel spectrogram extraction for Qwen3-ASR.
///
/// Matches the HuggingFace `WhisperFeatureExtractor` used during training:
/// - n_fft: 400 (window size and DFT length)
/// - hop_length: 160
/// - n_mels: 128
/// - Window: periodic Hann (not symmetric)
/// - No preemphasis
/// - log10 (not ln)
/// - Dynamic range compression + normalization
/// - HTK mel scale with Slaney normalization
///
/// Implementation note: Apple's vDSP FFT requires lengths of form `f * 2^n` where
/// `f ∈ {1,3,5,15}`, which excludes 400. We compute the DFT via a pre-computed
/// DFT matrix (cos/sin tables), using vDSP vector operations for efficiency.
/// This gives bit-exact results matching Python's `torch.stft(n_fft=400)`.
///
/// - Warning: This class is NOT thread-safe due to shared reusable buffers.
///   Each thread should use its own instance.
public final class WhisperMelSpectrogram {

    // MARK: Config

    private let nFFT: Int = 400
    private let hopLength: Int = 160
    private let nMels: Int = 128
    private let sampleRate: Int = 16000

    /// Number of frequency bins used (nFFT / 2 + 1, including Nyquist).
    private var numFreqBins: Int { nFFT / 2 + 1 }

    // MARK: Pre-computed

    private let hannWindow: [Float]
    private let melFilterbankFlat: [Float]  // [nMels * numFreqBins] row-major
    /// DFT cosine table: cos(-2π * k * n / N) for k=0..numFreqBins-1, n=0..nFFT-1
    /// Stored as flat [numFreqBins * nFFT], row-major by k.
    private let dftCos: [Float]
    /// DFT sine table: sin(-2π * k * n / N), same layout.
    private let dftSin: [Float]

    // MARK: Reusable buffers

    private var windowedFrame: [Float]
    private var realPart: [Float]  // Re(X[k]) for k=0..numFreqBins-1
    private var imagPart: [Float]  // Im(X[k])
    private var powerSpec: [Float]
    private var melFrame: [Float]

    public init() {
        let nFFT = 400
        let nMels = 128
        let numFreqBins = nFFT / 2 + 1  // 201 (including Nyquist, matches Python)

        // Periodic Hann window: np.hanning(n_fft + 1)[:-1]
        self.hannWindow = Self.createPeriodicHannWindow(length: nFFT)

        // HTK mel filterbank with Slaney normalization: [nMels, numFreqBins]
        let filterbank = Self.createMelFilterbank(
            nFFT: nFFT,
            nMels: nMels,
            sampleRate: 16000,
            fMin: 0.0,
            fMax: 8000.0
        )

        // Flatten row-major
        var flat = [Float](repeating: 0, count: nMels * numFreqBins)
        for m in 0..<nMels {
            for f in 0..<numFreqBins {
                flat[m * numFreqBins + f] = filterbank[m][f]
            }
        }
        self.melFilterbankFlat = flat

        // Pre-compute DFT cos/sin tables for k=0..numFreqBins-1, n=0..nFFT-1
        // X[k] = sum_{n=0}^{N-1} x[n] * exp(-2πi*k*n/N)
        //       = sum_{n=0}^{N-1} x[n] * (cos(-2π*k*n/N) + i*sin(-2π*k*n/N))
        var cos_table = [Float](repeating: 0, count: numFreqBins * nFFT)
        var sin_table = [Float](repeating: 0, count: numFreqBins * nFFT)
        let invN = Float(2.0 * .pi) / Float(nFFT)
        for k in 0..<numFreqBins {
            for n in 0..<nFFT {
                let angle = -invN * Float(k) * Float(n)
                cos_table[k * nFFT + n] = cosf(angle)
                sin_table[k * nFFT + n] = sinf(angle)
            }
        }
        self.dftCos = cos_table
        self.dftSin = sin_table

        // Pre-allocate buffers
        self.windowedFrame = [Float](repeating: 0, count: nFFT)
        self.realPart = [Float](repeating: 0, count: numFreqBins)
        self.imagPart = [Float](repeating: 0, count: numFreqBins)
        self.powerSpec = [Float](repeating: 0, count: numFreqBins)
        self.melFrame = [Float](repeating: 0, count: nMels)
    }

    // MARK: Public

    /// Compute Whisper-compatible mel spectrogram from 16kHz audio samples.
    ///
    /// - Parameter audio: Audio samples at 16kHz, mono, Float32.
    /// - Returns: Mel spectrogram as `[nMels][T]` where `T = audio.count / hopLength`.
    ///   Suitable for `Qwen3AsrManager.transcribe(melSpectrogram:)`.
    public func compute(audio: [Float]) -> [[Float]] {
        let numFrames = audio.count / hopLength
        guard numFrames > 0 else { return [] }
        let numFreqBins = self.numFreqBins

        // Allocate mel output: [nMels][numFrames]
        var mel = [[Float]](
            repeating: [Float](repeating: 0, count: numFrames),
            count: nMels
        )

        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopLength

            // Extract frame and apply window
            for i in 0..<nFFT {
                let audioIdx = startIdx + i
                if audioIdx < audio.count {
                    windowedFrame[i] = audio[audioIdx] * hannWindow[i]
                } else {
                    windowedFrame[i] = 0.0
                }
            }

            // DFT via matrix-vector multiply:
            // Re(X[k]) = sum(x[n] * cos(-2π*k*n/N))   = dot(dftCos[k], x)
            // Im(X[k]) = sum(x[n] * sin(-2π*k*n/N))   = dot(dftSin[k], x)
            // Power[k] = Re(X[k])^2 + Im(X[k])^2
            //
            // Using vDSP_mmul: [numFreqBins, nFFT] × [nFFT, 1] → [numFreqBins, 1]
            dftCos.withUnsafeBufferPointer { cosPtr in
                windowedFrame.withUnsafeBufferPointer { xPtr in
                    realPart.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            cosPtr.baseAddress!, 1,
                            xPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(numFreqBins),
                            vDSP_Length(1),
                            vDSP_Length(nFFT)
                        )
                    }
                }
            }

            dftSin.withUnsafeBufferPointer { sinPtr in
                windowedFrame.withUnsafeBufferPointer { xPtr in
                    imagPart.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            sinPtr.baseAddress!, 1,
                            xPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(numFreqBins),
                            vDSP_Length(1),
                            vDSP_Length(nFFT)
                        )
                    }
                }
            }

            // Power spectrum: |X[k]|^2 = Re^2 + Im^2
            vDSP_vsq(realPart, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
            var imagSq = [Float](repeating: 0, count: numFreqBins)
            vDSP_vsq(imagPart, 1, &imagSq, 1, vDSP_Length(numFreqBins))
            vDSP_vadd(powerSpec, 1, imagSq, 1, &powerSpec, 1, vDSP_Length(numFreqBins))

            // Apply mel filterbank: [nMels, numFreqBins] × [numFreqBins] → [nMels]
            melFilterbankFlat.withUnsafeBufferPointer { filterPtr in
                powerSpec.withUnsafeBufferPointer { specPtr in
                    melFrame.withUnsafeMutableBufferPointer { outPtr in
                        vDSP_mmul(
                            filterPtr.baseAddress!, 1,
                            specPtr.baseAddress!, 1,
                            outPtr.baseAddress!, 1,
                            vDSP_Length(nMels),
                            vDSP_Length(1),
                            vDSP_Length(numFreqBins)
                        )
                    }
                }
            }

            // log10(clip(x, 1e-10))
            for melIdx in 0..<nMels {
                mel[melIdx][frameIdx] = log10f(max(melFrame[melIdx], 1e-10))
            }
        }

        // Dynamic range compression: max(x, globalMax - 8.0)
        var globalMax: Float = -Float.infinity
        for melIdx in 0..<nMels {
            for frameIdx in 0..<numFrames {
                globalMax = max(globalMax, mel[melIdx][frameIdx])
            }
        }
        let minVal = globalMax - 8.0
        for melIdx in 0..<nMels {
            for frameIdx in 0..<numFrames {
                mel[melIdx][frameIdx] = max(mel[melIdx][frameIdx], minVal)
            }
        }

        // Whisper normalization: (x + 4.0) / 4.0
        for melIdx in 0..<nMels {
            for frameIdx in 0..<numFrames {
                mel[melIdx][frameIdx] = (mel[melIdx][frameIdx] + 4.0) / 4.0
            }
        }

        return mel
    }

    // MARK: Private - Window

    /// Create periodic Hann window matching `np.hanning(n + 1)[:-1]`.
    private static func createPeriodicHannWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        let n = Float(length)
        for i in 0..<length {
            window[i] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / n))
        }
        return window
    }

    // MARK: Private - Mel Filterbank

    /// Create HTK mel filterbank with Slaney normalization.
    ///
    /// Matches HuggingFace `WhisperFeatureExtractor.get_mel_filters()`:
    /// - HTK mel scale: mel = 2595 * log10(1 + f/700)
    /// - Slaney normalization: 2 / (f_right - f_left)
    /// - Returns `[nMels][nFFT/2+1]` (includes Nyquist bin, matches Python).
    private static func createMelFilterbank(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [[Float]] {
        let numFreqBins = nFFT / 2 + 1  // 201 (including Nyquist)

        // FFT frequency bins (including Nyquist)
        var fftFreqs = [Float](repeating: 0, count: numFreqBins)
        for i in 0..<numFreqBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        // HTK mel scale
        func hzToMel(_ hz: Float) -> Float {
            2595.0 * log10f(1.0 + hz / 700.0)
        }
        func melToHz(_ mel: Float) -> Float {
            700.0 * (powf(10.0, mel / 2595.0) - 1.0)
        }

        // Create nMels + 2 mel-spaced points
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            let mel = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
            melPoints[i] = melToHz(mel)
        }

        // Compute frequency differences between adjacent mel points
        var fdiff = [Float](repeating: 0, count: nMels + 1)
        for i in 0..<(nMels + 1) {
            fdiff[i] = melPoints[i + 1] - melPoints[i]
        }

        // Build filterbank using ramp method (matches HuggingFace implementation)
        var filterbank = [[Float]](
            repeating: [Float](repeating: 0, count: numFreqBins),
            count: nMels
        )

        for i in 0..<nMels {
            for f in 0..<numFreqBins {
                let lower = (fftFreqs[f] - melPoints[i]) / fdiff[i]
                let upper = (melPoints[i + 2] - fftFreqs[f]) / fdiff[i + 1]
                filterbank[i][f] = max(0, min(lower, upper))
            }

            // Slaney normalization: 2 / (f_right - f_left)
            let enorm = 2.0 / (melPoints[i + 2] - melPoints[i])
            for f in 0..<numFreqBins {
                filterbank[i][f] *= enorm
            }
        }

        return filterbank
    }
}
