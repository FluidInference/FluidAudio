import Accelerate
import Foundation

/// Whisper-compatible mel spectrogram for ForcedAligner using the **Slaney** mel scale.
///
/// The HuggingFace `WhisperFeatureExtractor` uses `mel_scale="slaney"` + `norm="slaney"`,
/// which differs from the HTK scale used by `WhisperMelSpectrogram`. This class replicates
/// the exact HF behavior:
/// - Slaney mel scale: linear below 1000Hz, logarithmic above
/// - Slaney normalization: `2 / (f_high - f_low)` per band
/// - Audio padded to 30 seconds before STFT (affects dynamic range normalization)
/// - Periodic Hann window, n_fft=400, hop=160, 128 mel bins
///
/// - Warning: Not thread-safe. Each thread/task should use its own instance.
final class ForcedAlignerMelSpectrogram {

    // MARK: Config

    private let nFFT: Int = 400
    private let hopLength: Int = 160
    private let nMels: Int = 128
    private let sampleRate: Int = 16000

    private var numFreqBins: Int { nFFT / 2 + 1 }

    // MARK: Pre-computed

    private let hannWindow: [Float]
    private let melFilterbankFlat: [Float]  // [nMels * numFreqBins] row-major
    private let dftCos: [Float]
    private let dftSin: [Float]

    // MARK: Reusable buffers

    private var windowedFrame: [Float]
    private var realPart: [Float]
    private var imagPart: [Float]
    private var powerSpec: [Float]
    private var imagSq: [Float]
    private var melFrame: [Float]

    init() {
        let numFreqBins = nFFT / 2 + 1

        self.hannWindow = Self.createPeriodicHannWindow(length: nFFT)
        let filterbank = Self.createSlaneyMelFilterbank(
            nFFT: nFFT, nMels: nMels, sampleRate: sampleRate, fMin: 0.0, fMax: 8000.0
        )

        var flat = [Float](repeating: 0, count: nMels * numFreqBins)
        for m in 0..<nMels {
            for f in 0..<numFreqBins {
                flat[m * numFreqBins + f] = filterbank[m][f]
            }
        }
        self.melFilterbankFlat = flat

        var cosTable = [Float](repeating: 0, count: numFreqBins * nFFT)
        var sinTable = [Float](repeating: 0, count: numFreqBins * nFFT)
        let invN = Float(2.0 * .pi) / Float(nFFT)
        for k in 0..<numFreqBins {
            for n in 0..<nFFT {
                let angle = -invN * Float(k) * Float(n)
                cosTable[k * nFFT + n] = cosf(angle)
                sinTable[k * nFFT + n] = sinf(angle)
            }
        }
        self.dftCos = cosTable
        self.dftSin = sinTable

        self.windowedFrame = [Float](repeating: 0, count: nFFT)
        self.realPart = [Float](repeating: 0, count: numFreqBins)
        self.imagPart = [Float](repeating: 0, count: numFreqBins)
        self.powerSpec = [Float](repeating: 0, count: numFreqBins)
        self.imagSq = [Float](repeating: 0, count: numFreqBins)
        self.melFrame = [Float](repeating: 0, count: nMels)
    }

    /// Compute mel spectrogram matching HF `WhisperFeatureExtractor` output.
    ///
    /// Audio is padded to 30 seconds internally (matching HF behavior), then cropped to
    /// `featureLen` actual frames.
    ///
    /// - Parameters:
    ///   - audio: 16kHz mono Float32 audio samples.
    /// - Returns: `(mel, featureLen)` where mel is `[nMels][featureLen]`.
    func compute(audio: [Float]) -> (mel: [[Float]], featureLen: Int) {
        let featureLen = audio.count / hopLength
        guard featureLen > 0 else { return ([], 0) }

        // Pad audio to 30 seconds (matches WhisperFeatureExtractor)
        let maxSamples = sampleRate * 30
        var paddedAudio = audio
        if paddedAudio.count < maxSamples {
            paddedAudio.append(contentsOf: [Float](repeating: 0.0, count: maxSamples - paddedAudio.count))
        }

        // Apply center=True reflect padding (torch.stft default)
        // Pad nFFT/2 on each side using reflect mode
        let padLen = nFFT / 2  // 200
        let centeredAudio = Self.reflectPad(paddedAudio, padLen: padLen)

        let totalFrames = paddedAudio.count / hopLength
        let numFreqBins = self.numFreqBins

        // Compute mel on padded audio (needed for correct normalization)
        var mel = [[Float]](
            repeating: [Float](repeating: 0, count: totalFrames),
            count: nMels
        )

        for frameIdx in 0..<totalFrames {
            let startIdx = frameIdx * hopLength

            for i in 0..<nFFT {
                let audioIdx = startIdx + i
                if audioIdx < centeredAudio.count {
                    windowedFrame[i] = centeredAudio[audioIdx] * hannWindow[i]
                } else {
                    windowedFrame[i] = 0.0
                }
            }

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

            vDSP_vsq(realPart, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
            vDSP_vsq(imagPart, 1, &imagSq, 1, vDSP_Length(numFreqBins))
            vDSP_vadd(powerSpec, 1, imagSq, 1, &powerSpec, 1, vDSP_Length(numFreqBins))

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

            var minClip: Float = 1e-10
            var maxClip: Float = Float.greatestFiniteMagnitude
            vDSP_vclip(melFrame, 1, &minClip, &maxClip, &melFrame, 1, vDSP_Length(nMels))

            var count = Int32(nMels)
            vvlog10f(&melFrame, melFrame, &count)

            for melIdx in 0..<nMels {
                mel[melIdx][frameIdx] = melFrame[melIdx]
            }
        }

        // Dynamic range compression over ALL frames (including padding)
        var globalMax: Float = -Float.infinity
        for melIdx in 0..<nMels {
            var rowMax: Float = 0
            vDSP_maxv(mel[melIdx], 1, &rowMax, vDSP_Length(totalFrames))
            globalMax = max(globalMax, rowMax)
        }
        let minVal = globalMax - 8.0

        for melIdx in 0..<nMels {
            var low = minVal
            var high = Float.greatestFiniteMagnitude
            mel[melIdx].withUnsafeMutableBufferPointer { buffer in
                vDSP_vclip(buffer.baseAddress!, 1, &low, &high, buffer.baseAddress!, 1, vDSP_Length(totalFrames))
            }
        }

        // Whisper normalization
        var addVal: Float = 4.0
        var divVal: Float = 4.0
        for melIdx in 0..<nMels {
            mel[melIdx].withUnsafeMutableBufferPointer { buffer in
                vDSP_vsadd(buffer.baseAddress!, 1, &addVal, buffer.baseAddress!, 1, vDSP_Length(totalFrames))
                vDSP_vsdiv(buffer.baseAddress!, 1, &divVal, buffer.baseAddress!, 1, vDSP_Length(totalFrames))
            }
        }

        // Crop to actual feature length
        let croppedMel = mel.map { Array($0.prefix(featureLen)) }
        return (croppedMel, featureLen)
    }

    // MARK: Private

    /// Reflect-pad array by `padLen` on each side.
    ///
    /// Matches `np.pad(x, (padLen, padLen), mode='reflect')` / `torch.stft(center=True)`.
    private static func reflectPad(_ input: [Float], padLen: Int) -> [Float] {
        let n = input.count
        var result = [Float](repeating: 0, count: padLen + n + padLen)

        // Left reflect: input[padLen], input[padLen-1], ..., input[1]
        for i in 0..<padLen {
            result[i] = input[padLen - i]
        }

        // Center: copy input
        for i in 0..<n {
            result[padLen + i] = input[i]
        }

        // Right reflect: input[n-2], input[n-3], ..., input[n-1-padLen]
        for i in 0..<padLen {
            result[padLen + n + i] = input[n - 2 - i]
        }

        return result
    }

    private static func createPeriodicHannWindow(length: Int) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        let n = Float(length)
        for i in 0..<length {
            window[i] = 0.5 * (1.0 - cosf(2.0 * .pi * Float(i) / n))
        }
        return window
    }

    /// Create mel filterbank using Slaney mel scale + Slaney normalization.
    ///
    /// Matches `transformers.audio_utils.mel_filter_bank(mel_scale="slaney", norm="slaney")`.
    ///
    /// Slaney scale:
    /// - Below 1000Hz: mel = 3 * f / 200 (linear)
    /// - Above 1000Hz: mel = 15 + 27/ln(6.4) * ln(f/1000) (logarithmic)
    private static func createSlaneyMelFilterbank(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [[Float]] {
        let numFreqBins = nFFT / 2 + 1

        // FFT frequency bins
        var fftFreqs = [Float](repeating: 0, count: numFreqBins)
        for i in 0..<numFreqBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        // Slaney mel scale
        let minLogHz: Float = 1000.0
        let minLogMel: Float = 15.0
        let logStep: Float = 27.0 / logf(6.4)

        func hzToMel(_ hz: Float) -> Float {
            if hz >= minLogHz {
                return minLogMel + logf(hz / minLogHz) * logStep
            } else {
                return 3.0 * hz / 200.0
            }
        }
        func melToHz(_ mel: Float) -> Float {
            if mel >= minLogMel {
                return minLogHz * expf((mel - minLogMel) / logStep)
            } else {
                return 200.0 * mel / 3.0
            }
        }

        // Create nMels + 2 linearly-spaced points in mel space
        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)
        var hzPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            let mel = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
            hzPoints[i] = melToHz(mel)
        }

        // Frequency differences
        var fdiff = [Float](repeating: 0, count: nMels + 1)
        for i in 0..<(nMels + 1) {
            fdiff[i] = hzPoints[i + 1] - hzPoints[i]
        }

        // Build filterbank
        var filterbank = [[Float]](
            repeating: [Float](repeating: 0, count: numFreqBins),
            count: nMels
        )

        for i in 0..<nMels {
            for f in 0..<numFreqBins {
                let lower = (fftFreqs[f] - hzPoints[i]) / fdiff[i]
                let upper = (hzPoints[i + 2] - fftFreqs[f]) / fdiff[i + 1]
                filterbank[i][f] = max(0, min(lower, upper))
            }

            // Slaney normalization: 2 / (f_high - f_low)
            let enorm = 2.0 / (hzPoints[i + 2] - hzPoints[i])
            for f in 0..<numFreqBins {
                filterbank[i][f] *= enorm
            }
        }

        return filterbank
    }
}
