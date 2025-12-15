import Accelerate
import Foundation

/// Native Swift mel spectrogram implementation matching NeMo's AudioToMelSpectrogramPreprocessor.
/// This replaces the CoreML preprocessor to ensure exact numerical parity with PyTorch.
///
/// Config (matches nvidia/parakeet_realtime_eou_120m-v1):
/// - sample_rate: 16000
/// - window_size: 0.025 (400 samples)
/// - window_stride: 0.01 (160 samples / hop)
/// - n_fft: 512
/// - features: 128 mel bins
/// - window: hann (symmetric)
/// - preemph: 0.97 (high-pass preemphasis filter)
/// - center: True with pad_mode='constant' (zero padding)
/// - normalize: "NA" (no normalization)
/// - dither: 0.0 (disabled for determinism)
public final class NeMoMelSpectrogram {
    // Config
    private let sampleRate: Int = 16000
    private let nFFT: Int = 512
    private let hopLength: Int = 160  // window_stride * sample_rate
    private let winLength: Int = 400  // window_size * sample_rate
    private let nMels: Int = 128
    private let fMin: Float = 0.0
    private let fMax: Float = 8000.0  // sample_rate / 2
    private let preemph: Float = 0.97  // NeMo preemphasis coefficient
    private let logZeroGuard: Float = 5.960464477539063e-08  // NeMo log_zero_guard_value
    private let logZero: Float = -16.635532  // log(1e-10) for padding

    // Pre-computed
    private let hannWindow: [Float]
    private let melFilterbank: [[Float]]  // [nMels, nFFT/2 + 1]
    private var fftSetup: vDSP_DFT_Setup?

    public init() {
        // 1. Create symmetric Hann window (matches PyTorch hann_window(periodic=False))
        self.hannWindow = Self.createHannWindow(length: winLength, periodic: false)

        // 2. Create mel filterbank with Slaney normalization
        self.melFilterbank = Self.createMelFilterbank(
            nFFT: nFFT,
            nMels: nMels,
            sampleRate: sampleRate,
            fMin: fMin,
            fMax: fMax
        )

        // 3. Setup FFT
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(nFFT),
            .FORWARD
        )
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// Compute mel spectrogram from audio samples.
    /// - Parameter audio: Audio samples at 16kHz
    /// - Returns: (mel, mel_length) where mel is [1, nMels, T] and mel_length is valid frame count
    public func compute(audio: [Float]) -> (mel: [[[Float]]], melLength: Int) {
        let numFrames = 1 + (audio.count - winLength) / hopLength

        guard numFrames > 0 else {
            return (mel: [[[Float]]](), melLength: 0)
        }

        var melFrames: [[Float]] = []
        melFrames.reserveCapacity(numFrames)

        // Process each frame
        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopLength

            // Extract and window the frame
            var frame = [Float](repeating: 0, count: nFFT)
            for i in 0..<winLength {
                let audioIdx = startIdx + i
                if audioIdx < audio.count {
                    frame[i] = audio[audioIdx] * hannWindow[i]
                }
            }

            // Compute power spectrum (magnitude squared)
            let powerSpec = computePowerSpectrum(frame: frame)

            // Apply mel filterbank
            let melSpec = applyMelFilterbank(powerSpec: powerSpec)

            // Apply log (with floor for numerical stability)
            let logMelSpec = melSpec.map { value -> Float in
                let floored = max(value, 1e-10)
                return log(floored)
            }

            melFrames.append(logMelSpec)
        }

        // Reshape to [1, nMels, T]
        var mel = [[[Float]]](repeating: [[Float]](), count: 1)
        mel[0] = [[Float]](repeating: [Float](), count: nMels)

        for melIdx in 0..<nMels {
            mel[0][melIdx] = melFrames.map { $0[melIdx] }
        }

        return (mel: mel, melLength: numFrames)
    }

    /// Compute mel spectrogram and return as flat array for MLMultiArray compatibility.
    /// - Parameter audio: Audio samples at 16kHz
    /// - Returns: (mel, mel_length, numFrames) where mel is flat [nMels * T]
    public func computeFlat(audio: [Float]) -> (mel: [Float], melLength: Int, numFrames: Int) {
        guard !audio.isEmpty else {
            return (mel: [Float](repeating: logZero, count: nMels), melLength: 0, numFrames: 1)
        }

        // Step 1: Apply preemphasis filter (y[n] = x[n] - preemph * x[n-1])
        var preemphAudio = [Float](repeating: 0, count: audio.count)
        preemphAudio[0] = audio[0]  // Keep first sample unchanged
        for i in 1..<audio.count {
            preemphAudio[i] = audio[i] - preemph * audio[i - 1]
        }

        // Step 2: Apply center padding with CONSTANT (zeros), not reflect
        // NeMo uses pad_mode='constant' in torch.stft
        let padLength = nFFT / 2
        var paddedAudio = [Float](repeating: 0, count: preemphAudio.count + 2 * padLength)

        // Zero padding at start (already zeros from initialization)
        // Copy preemph audio to center
        for i in 0..<preemphAudio.count {
            paddedAudio[padLength + i] = preemphAudio[i]
        }
        // Zero padding at end (already zeros from initialization)

        // Calculate number of frames with center padding
        let numFrames = 1 + (paddedAudio.count - winLength) / hopLength

        guard numFrames > 0 else {
            return (mel: [Float](repeating: logZero, count: nMels), melLength: 0, numFrames: 1)
        }

        // Allocate output: [nMels, numFrames] in row-major order
        var mel = [Float](repeating: logZero, count: nMels * numFrames)

        // Window centering offset: torch.stft centers the window within n_fft when win_length < n_fft
        let windowOffset = (nFFT - winLength) / 2  // = 56 for nFFT=512, winLength=400

        // Process each frame
        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopLength

            // Extract and window the frame
            // torch.stft extracts n_fft samples, then applies window centered within the frame
            var frame = [Float](repeating: 0, count: nFFT)

            // Apply centered window: window covers frame[windowOffset : windowOffset + winLength]
            // The samples at position windowOffset+i in the frame come from paddedAudio[startIdx + windowOffset + i]
            for i in 0..<winLength {
                let audioIdx = startIdx + windowOffset + i
                if audioIdx < paddedAudio.count {
                    frame[windowOffset + i] = paddedAudio[audioIdx] * hannWindow[i]
                }
            }

            // Compute power spectrum (magnitude squared)
            let powerSpec = computePowerSpectrum(frame: frame)

            // Apply mel filterbank and log with NeMo's log_zero_guard_type='add'
            for melIdx in 0..<nMels {
                var sum: Float = 0
                for freqIdx in 0..<(nFFT / 2 + 1) {
                    sum += melFilterbank[melIdx][freqIdx] * powerSpec[freqIdx]
                }
                // NeMo uses log(x + guard_value), not log(max(x, guard_value))
                let logVal = log(sum + logZeroGuard)
                mel[melIdx * numFrames + frameIdx] = logVal
            }
        }

        return (mel: mel, melLength: numFrames, numFrames: numFrames)
    }

    // MARK: - Debug Methods

    /// Get the mel filterbank for debugging
    public func getFilterbank() -> [[Float]] {
        return melFilterbank
    }

    /// Get the Hann window for debugging
    public func getHannWindow() -> [Float] {
        return hannWindow
    }

    // MARK: - Private Methods

    private func computePowerSpectrum(frame: [Float]) -> [Float] {
        guard let setup = fftSetup else {
            return [Float](repeating: 0, count: nFFT / 2 + 1)
        }

        // Split into real and imaginary parts for vDSP
        var realIn = [Float](repeating: 0, count: nFFT)
        var imagIn = [Float](repeating: 0, count: nFFT)
        var realOut = [Float](repeating: 0, count: nFFT)
        var imagOut = [Float](repeating: 0, count: nFFT)

        // Copy frame to real input
        for i in 0..<min(frame.count, nFFT) {
            realIn[i] = frame[i]
        }

        // Execute FFT
        vDSP_DFT_Execute(setup, realIn, imagIn, &realOut, &imagOut)

        // Compute power spectrum: real^2 + imag^2 (magnitude squared)
        // This matches NeMo's use of magnitude squared in mel spectrogram
        var power = [Float](repeating: 0, count: nFFT / 2 + 1)
        for i in 0..<(nFFT / 2 + 1) {
            let real = realOut[i]
            let imag = imagOut[i]
            power[i] = real * real + imag * imag
        }

        return power
    }

    private func applyMelFilterbank(powerSpec: [Float]) -> [Float] {
        var melSpec = [Float](repeating: 0, count: nMels)

        for melIdx in 0..<nMels {
            var sum: Float = 0
            for freqIdx in 0..<min(powerSpec.count, melFilterbank[melIdx].count) {
                sum += melFilterbank[melIdx][freqIdx] * powerSpec[freqIdx]
            }
            melSpec[melIdx] = sum
        }

        return melSpec
    }

    // MARK: - Static Factory Methods

    private static func createHannWindow(length: Int, periodic: Bool) -> [Float] {
        var window = [Float](repeating: 0, count: length)
        // Symmetric Hann window: divide by (length - 1) for symmetric, by length for periodic
        let divisor = periodic ? Float(length) : Float(length - 1)
        for i in 0..<length {
            let phase = 2.0 * Float.pi * Float(i) / divisor
            window[i] = 0.5 * (1.0 - cos(phase))
        }
        return window
    }

    private static func createMelFilterbank(
        nFFT: Int,
        nMels: Int,
        sampleRate: Int,
        fMin: Float,
        fMax: Float
    ) -> [[Float]] {
        let numFreqBins = nFFT / 2 + 1

        // Convert Hz to Mel scale using Slaney formula (librosa default)
        // Below 1000 Hz: linear, Above 1000 Hz: logarithmic
        func hzToMel(_ hz: Float) -> Float {
            let fSp: Float = 200.0 / 3.0  // ~66.67 Hz
            let minLogHz: Float = 1000.0
            let minLogMel: Float = minLogHz / fSp  // 15.0
            let logStep: Float = log(6.4) / 27.0  // log(6400/1000) / 27

            if hz >= minLogHz {
                return minLogMel + log(hz / minLogHz) / logStep
            } else {
                return hz / fSp
            }
        }

        func melToHz(_ mel: Float) -> Float {
            let fSp: Float = 200.0 / 3.0
            let minLogHz: Float = 1000.0
            let minLogMel: Float = minLogHz / fSp
            let logStep: Float = log(6.4) / 27.0

            if mel >= minLogMel {
                return minLogHz * exp(logStep * (mel - minLogMel))
            } else {
                return fSp * mel
            }
        }

        let melMin = hzToMel(fMin)
        let melMax = hzToMel(fMax)

        // Create mel points evenly spaced in mel scale
        var melPoints = [Float](repeating: 0, count: nMels + 2)
        for i in 0..<(nMels + 2) {
            let mel = melMin + Float(i) * (melMax - melMin) / Float(nMels + 1)
            melPoints[i] = melToHz(mel)
        }

        // FFT frequency bins
        var fftFreqs = [Float](repeating: 0, count: numFreqBins)
        for i in 0..<numFreqBins {
            fftFreqs[i] = Float(i) * Float(sampleRate) / Float(nFFT)
        }

        // Create filterbank matrix with Slaney normalization
        var filterbank = [[Float]](repeating: [Float](repeating: 0, count: numFreqBins), count: nMels)

        for melIdx in 0..<nMels {
            let fLeft = melPoints[melIdx]
            let fCenter = melPoints[melIdx + 1]
            let fRight = melPoints[melIdx + 2]

            // Slaney normalization factor: 2 / (fRight - fLeft)
            let norm = 2.0 / (fRight - fLeft)

            for freqIdx in 0..<numFreqBins {
                let freq = fftFreqs[freqIdx]

                if freq >= fLeft && freq < fCenter {
                    // Rising slope
                    filterbank[melIdx][freqIdx] = norm * (freq - fLeft) / (fCenter - fLeft)
                } else if freq >= fCenter && freq <= fRight {
                    // Falling slope
                    filterbank[melIdx][freqIdx] = norm * (fRight - freq) / (fRight - fCenter)
                }
            }
        }

        return filterbank
    }
}
