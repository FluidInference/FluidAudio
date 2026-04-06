import Accelerate
import Foundation

/// Cohere Transcribe mel spectrogram extraction.
///
/// Matches the exact preprocessing used by Cohere models:
/// - n_fft: 1024
/// - hop_length: 160
/// - n_mels: 128
/// - Window: Hann
/// - Preemphasis: 0.97
/// - log (natural log, not log10)
/// - fmin: 0.0, fmax: 8000.0
/// - Mel scale: HTK
///
/// Unlike Whisper (n_fft=400), Cohere uses n_fft=1024 which is a power of 2,
/// allowing direct use of vDSP FFT for efficiency.
///
/// - Warning: This class is NOT thread-safe and NOT `Sendable` due to mutable
///   reusable buffers. Each thread/task should use its own instance.
public final class CohereMelSpectrogram {

    // MARK: Config

    private let nFFT: Int = CohereAsrConfig.MelSpec.nFFT
    private let hopLength: Int = CohereAsrConfig.MelSpec.hopLength
    private let nMels: Int = CohereAsrConfig.MelSpec.nMels
    private let sampleRate: Int = CohereAsrConfig.sampleRate
    private let preemphasis: Float = CohereAsrConfig.MelSpec.preemphasis
    private let fMin: Float = CohereAsrConfig.MelSpec.fMin
    private let fMax: Float = CohereAsrConfig.MelSpec.fMax

    /// Number of frequency bins (nFFT / 2 + 1, including Nyquist).
    private var numFreqBins: Int { nFFT / 2 + 1 }

    // MARK: Pre-computed

    private let hannWindow: [Float]
    private let melFilterbankFlat: [Float]  // [nMels * numFreqBins] row-major
    private let fftSetup: vDSP.FFT<DSPSplitComplex>

    // MARK: Reusable buffers

    private var windowedFrame: [Float]
    private var realPart: [Float]
    private var imagPart: [Float]
    private var powerSpec: [Float]
    private var melFrame: [Float]

    public init() {
        let numFreqBins = nFFT / 2 + 1

        // Create Hann window
        self.hannWindow = Self.createHannWindow(length: nFFT)

        // Create mel filterbank
        let filterbank = Self.createMelFilterbank(
            nFFT: nFFT,
            nMels: nMels,
            sampleRate: sampleRate,
            fMin: fMin,
            fMax: fMax
        )

        // Flatten row-major
        var flat = [Float](repeating: 0, count: nMels * numFreqBins)
        for m in 0..<nMels {
            for f in 0..<numFreqBins {
                flat[m * numFreqBins + f] = filterbank[m][f]
            }
        }
        self.melFilterbankFlat = flat

        // Create FFT setup (n_fft=1024 = 2^10, so log2n=10)
        let log2n = vDSP_Length(10)  // log2(1024)
        self.fftSetup = vDSP.FFT(log2n: log2n, radix: .radix2, ofType: DSPSplitComplex.self)!

        // Initialize buffers
        self.windowedFrame = [Float](repeating: 0, count: nFFT)
        self.realPart = [Float](repeating: 0, count: numFreqBins)
        self.imagPart = [Float](repeating: 0, count: numFreqBins)
        self.powerSpec = [Float](repeating: 0, count: numFreqBins)
        self.melFrame = [Float](repeating: 0, count: nMels)
    }

    /// Compute mel spectrogram from audio samples.
    ///
    /// - Parameter audio: Raw audio samples (Float32, 16kHz mono).
    /// - Returns: Mel spectrogram as [nMels, nFrames] (row-major).
    public func compute(audio: [Float]) -> [[Float]] {
        // Apply pre-emphasis filter
        let preemphasized = applyPreemphasis(audio)

        // Pad audio for reflection padding
        let padLength = nFFT / 2
        let paddedAudio = reflectionPad(preemphasized, padLength: padLength)

        // Calculate number of frames
        let numFrames = 1 + (paddedAudio.count - nFFT) / hopLength

        // Extract frames and compute mel spectrogram
        var melSpec = [[Float]](repeating: [Float](repeating: 0, count: numFrames), count: nMels)

        for frameIdx in 0..<numFrames {
            let start = frameIdx * hopLength
            let frame = Array(paddedAudio[start..<start + nFFT])

            // Compute power spectrum
            computePowerSpectrum(frame: frame)

            // Apply mel filterbank
            applyMelFilterbank()

            // Copy mel frame
            for m in 0..<nMels {
                melSpec[m][frameIdx] = melFrame[m]
            }
        }

        return melSpec
    }

    // MARK: - Private Helpers

    /// Apply pre-emphasis filter: y[n] = x[n] - alpha * x[n-1].
    private func applyPreemphasis(_ audio: [Float]) -> [Float] {
        guard !audio.isEmpty else { return [] }

        var result = [Float](repeating: 0, count: audio.count)
        result[0] = audio[0]

        for i in 1..<audio.count {
            result[i] = audio[i] - preemphasis * audio[i - 1]
        }

        return result
    }

    /// Reflection padding (mirrors the signal at boundaries).
    private func reflectionPad(_ audio: [Float], padLength: Int) -> [Float] {
        var padded = [Float](repeating: 0, count: audio.count + 2 * padLength)

        // Left padding (reverse)
        for i in 0..<padLength {
            padded[i] = audio[padLength - i]
        }

        // Original signal
        for i in 0..<audio.count {
            padded[i + padLength] = audio[i]
        }

        // Right padding (reverse)
        for i in 0..<padLength {
            padded[padLength + audio.count + i] = audio[audio.count - 2 - i]
        }

        return padded
    }

    /// Compute power spectrum from a single frame.
    private func computePowerSpectrum(frame: [Float]) {
        // Apply Hann window
        vDSP.multiply(frame, hannWindow, result: &windowedFrame)

        // Perform FFT
        let halfN = nFFT / 2

        realPart.withUnsafeMutableBufferPointer { realPtr in
            imagPart.withUnsafeMutableBufferPointer { imagPtr in
                var splitComplex = DSPSplitComplex(realp: realPtr.baseAddress!, imagp: imagPtr.baseAddress!)

                windowedFrame.withUnsafeBufferPointer { framePtr in
                    let complexBuffer = framePtr.baseAddress!.withMemoryRebound(
                        to: DSPComplex.self,
                        capacity: halfN
                    ) { $0 }
                    vDSP_ctoz(complexBuffer, 2, &splitComplex, 1, vDSP_Length(halfN))
                }

                fftSetup.transform(input: splitComplex, output: &splitComplex, direction: .forward)
            }
        }

        // Compute power spectrum: |X[k]|^2 = Re^2 + Im^2
        vDSP.squareAndAdd(realPart, imagPart, result: &powerSpec)
    }

    /// Apply mel filterbank to power spectrum.
    private func applyMelFilterbank() {
        let numFreqBins = self.numFreqBins

        melFilterbankFlat.withUnsafeBufferPointer { filterPtr in
            powerSpec.withUnsafeBufferPointer { powerPtr in
                melFrame.withUnsafeMutableBufferPointer { melPtr in
                    for m in 0..<nMels {
                        let filterRow = filterPtr.baseAddress! + m * numFreqBins
                        var sum: Float = 0
                        vDSP_dotpr(filterRow, 1, powerPtr.baseAddress!, 1, &sum, vDSP_Length(numFreqBins))
                        // Log scale with floor to prevent log(0)
                        melPtr[m] = log(max(sum, 1e-10))
                    }
                }
            }
        }
    }

    // MARK: - Window Functions

    private static func createHannWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        vDSP_hann_window(&window, vDSP_Length(length), Int32(vDSP_HANN_NORM))
        return window
    }

    // MARK: - Mel Filterbank

    private static func createMelFilterbank(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [[Float]] {
        let numFreqBins = nFFT / 2 + 1

        // Hz to Mel (HTK formula)
        func hzToMel(_ hz: Float) -> Float {
            return 2595.0 * log10(1.0 + hz / 700.0)
        }

        func melToHz(_ mel: Float) -> Float {
            return 700.0 * (pow(10.0, mel / 2595.0) - 1.0)
        }

        // Create mel scale
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        let melPoints = (0...nMels + 1).map { i in
            melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
        }

        // Convert to FFT bin numbers
        let binPoints = melPoints.map { hz in
            Int(floor(Float(nFFT + 1) * hz / Float(sampleRate)))
        }

        // Create filterbank
        var filterbank = [[Float]](
            repeating: [Float](repeating: 0, count: numFreqBins),
            count: nMels
        )

        for m in 0..<nMels {
            let fLeft = binPoints[m]
            let fCenter = binPoints[m + 1]
            let fRight = binPoints[m + 2]

            // Left slope
            for k in fLeft..<fCenter where k < numFreqBins {
                filterbank[m][k] = Float(k - fLeft) / Float(fCenter - fLeft)
            }

            // Right slope
            for k in fCenter..<fRight where k < numFreqBins {
                filterbank[m][k] = Float(fRight - k) / Float(fRight - fCenter)
            }
        }

        return filterbank
    }
}

extension vDSP {
    /// Compute Re^2 + Im^2 for complex number magnitude squared.
    static func squareAndAdd(_ real: [Float], _ imag: [Float], result: inout [Float]) {
        var realSq = [Float](repeating: 0, count: real.count)
        var imagSq = [Float](repeating: 0, count: imag.count)

        vDSP.square(real, result: &realSq)
        vDSP.square(imag, result: &imagSq)
        vDSP.add(realSq, imagSq, result: &result)
    }
}
