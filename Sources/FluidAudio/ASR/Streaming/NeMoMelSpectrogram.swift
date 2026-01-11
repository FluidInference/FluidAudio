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
///
/// - Warning: This class is NOT thread-safe. Each thread should use its own instance
///   due to shared reusable buffers for FFT computation.
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
    private let logZeroGuard: Float = powf(2, -24)  // 5.960464477539063e-08  // NeMo log_zero_guard_value
    private let logZero: Float = 0  // Padding value (0 works because log(0 + guard) ≈ -16.6)

    // Pre-computed
    private let hannWindow: [Float]
    private let melFilterbank: [[Float]]  // [nMels, nFFT/2 + 1]
    private let melFilterbankFlat: [Float]  // Flat [nMels * (nFFT/2 + 1)] for vDSP
    private var fftSetup: vDSP_DFT_Setup?

    // Reusable buffers to avoid allocations in hot path
    private var realIn: [Float]
    private var imagIn: [Float]
    private var realOut: [Float]
    private var imagOut: [Float]
    private var powerSpec: [Float]
    private var imagSq: [Float]
    private var frame: [Float]

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

        // 3. Flatten filterbank for vDSP matrix multiply: [nMels, numFreqBins] row-major
        let numFreqBins = nFFT / 2 + 1
        var flat = [Float](repeating: 0, count: nMels * numFreqBins)
        for m in 0..<nMels {
            for f in 0..<numFreqBins {
                flat[m * numFreqBins + f] = melFilterbank[m][f]
            }
        }
        self.melFilterbankFlat = flat

        // 4. Setup FFT
        self.fftSetup = vDSP_DFT_zop_CreateSetup(
            nil,
            vDSP_Length(nFFT),
            .FORWARD
        )

        // 5. Pre-allocate reusable buffers
        self.realIn = [Float](repeating: 0, count: nFFT)
        self.imagIn = [Float](repeating: 0, count: nFFT)
        self.realOut = [Float](repeating: 0, count: nFFT)
        self.imagOut = [Float](repeating: 0, count: nFFT)
        self.powerSpec = [Float](repeating: 0, count: numFreqBins)
        self.imagSq = [Float](repeating: 0, count: numFreqBins)
        self.frame = [Float](repeating: 0, count: nFFT)
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// Compute mel spectrogram from audio samples.
    /// - Note: This method does NOT apply preemphasis filtering. Use `computeFlat()` for
    ///   full NeMo-compatible preprocessing including preemphasis.
    /// - Parameter audio: Audio samples at 16kHz
    /// - Returns: (mel, mel_length) where mel is [1, nMels, T] and mel_length is valid frame count
    public func compute(audio: [Float]) -> (mel: [[[Float]]], melLength: Int) {
        guard !audio.isEmpty else {
            return (mel: [[[Float]]](), melLength: 0)
        }

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
    /// - Parameters:
    ///   - audio: Audio samples at 16kHz
    ///   - lastAudioSample: The last audio sample that's not in the audio buffer. Used to initialize the preemphasis state
    /// - Returns: (mel, mel_length, numFrames) where mel is flat [nMels * T]
    public func computeFlat(
        audio: [Float], lastAudioSample: Float = 0
    ) -> (mel: [Float], melLength: Int, numFrames: Int) {
        let audioCount = audio.count

        // Step 1: Apply preemphasis filter using vDSP (y[n] = x[n] - preemph * x[n-1])
        // This will be copied into an already padded buffer to save time.

        let padLength = nFFT / 2
        let paddedCount = audioCount + 2 * padLength

        // Calculate number of frames (must match NeMo's center=True padding)
        let numFrames = 1 + (paddedCount - winLength) / hopLength

        guard numFrames > 0 else {
            return (mel: [Float](repeating: 0, count: nMels), melLength: 0, numFrames: 1)
        }
        var paddedAudio = [Float](repeating: 0, count: paddedCount)

        paddedAudio[padLength] = audio[0] - preemph * lastAudioSample

        // Compute x[n] - preemph * x[n-1] vectorized
        paddedAudio.withUnsafeMutableBufferPointer { dstPtr in
            audio.withUnsafeBufferPointer { srcPtr in
                let src = srcPtr.baseAddress!
                let dst = dstPtr.baseAddress! + padLength + 1
                var negPreemph = -preemph
                vDSP_vsma(
                    src, 1,
                    &negPreemph,
                    src + 1, 1,
                    dst, 1,
                    vDSP_Length(audioCount - 1)
                )
            }
        }

        // Allocate output: [nMels, numFrames] in row-major order
        var mel = [Float](repeating: logZero, count: nMels * numFrames)
        let numFreqBins = nFFT / 2 + 1

        // Window centering offset
        let windowOffset = (nFFT - winLength) / 2

        // Temporary buffer for mel values of current frame
        var melFrame = [Float](repeating: 0, count: nMels)

        // Process each frame
        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopLength

            // Clear frame buffer
            vDSP_vclr(&frame, 1, vDSP_Length(nFFT))

            // Extract and window using vDSP_vmul
            let audioStart = startIdx + windowOffset
            let availableSamples = min(winLength, paddedCount - audioStart)

            if availableSamples > 0 {
                paddedAudio.withUnsafeBufferPointer { paddedPtr in
                    hannWindow.withUnsafeBufferPointer { windowPtr in
                        frame.withUnsafeMutableBufferPointer { framePtr in
                            vDSP_vmul(
                                paddedPtr.baseAddress! + audioStart, 1,
                                windowPtr.baseAddress!, 1,
                                framePtr.baseAddress! + windowOffset, 1,
                                vDSP_Length(availableSamples)
                            )
                        }
                    }
                }
            }

            // Compute power spectrum using reusable buffers
            computePowerSpectrumInPlace()

            // Apply mel filterbank using vDSP matrix-vector multiply
            // melFrame = melFilterbankFlat * powerSpec (matrix [nMels x numFreqBins] * vector [numFreqBins])
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

            // Apply log(x + guard_value) and store in output
            for melIdx in 0..<nMels {
                mel[melIdx * numFrames + frameIdx] = log(melFrame[melIdx] + logZeroGuard)
            }
        }

        return (mel: mel, melLength: numFrames, numFrames: numFrames)
    }

    /// Compute mel spectrogram and return as flat array for MLMultiArray compatibility.
    ///
    /// - Note: Uses frame count formula `audioCount / hopLength` (different from `computeFlat()` which uses
    ///   NeMo's center=True formula). This matches Sortformer's expected input format.
    /// - Parameters:
    ///   - audio: Audio samples at 16kHz
    ///   - lastAudioSample: The last audio sample that's not in the audio buffer. Used to initialize the preemphasis state
    /// - Returns: (mel, mel_length, numFrames) where mel is flat [numFrames * nMels] (transposed layout)
    public func computeFlatTransposed(
        audio: [Float], lastAudioSample: Float = 0
    ) -> (mel: [Float], melLength: Int, numFrames: Int) {
        let audioCount = audio.count
        let numFrames = audioCount / hopLength

        guard numFrames > 0 else {
            return (mel: [Float](repeating: 0, count: nMels), melLength: 0, numFrames: 1)
        }

        // Step 1: Apply preemphasis filter using vDSP (y[n] = x[n] - preemph * x[n-1])
        // This will be copied into an already padded buffer to save time.

        let padLength = nFFT / 2
        let paddedCount = audioCount + 2 * padLength
        var paddedAudio = [Float](repeating: 0, count: paddedCount)

        paddedAudio[padLength] = audio[0] - preemph * lastAudioSample

        // Compute x[n] - preemph * x[n-1] vectorized
        paddedAudio.withUnsafeMutableBufferPointer { dstPtr in
            audio.withUnsafeBufferPointer { srcPtr in
                let src = srcPtr.baseAddress!
                let dst = dstPtr.baseAddress! + padLength + 1
                var negPreemph = -preemph
                vDSP_vsma(
                    src, 1,
                    &negPreemph,
                    src + 1, 1,
                    dst, 1,
                    vDSP_Length(audioCount - 1)
                )
            }
        }

        // Allocate output: [numFrames, nMels] in row-major order (transposed layout for Sortformer)
        var mel = [Float](repeating: 0, count: nMels * numFrames)
        let numFreqBins = nFFT / 2 + 1

        // Window centering offset
        let windowOffset = (nFFT - winLength) / 2

        // Temporary buffer for mel values of current frame
        var melFrame = [Float](repeating: 0, count: nMels)

        // Process each frame
        for frameIdx in 0..<numFrames {
            let startIdx = frameIdx * hopLength

            // Clear frame buffer
            vDSP_vclr(&frame, 1, vDSP_Length(nFFT))

            // Extract and window using vDSP_vmul
            let audioStart = startIdx + windowOffset
            let availableSamples = min(winLength, paddedCount - audioStart)

            if availableSamples > 0 {
                paddedAudio.withUnsafeBufferPointer { paddedPtr in
                    hannWindow.withUnsafeBufferPointer { windowPtr in
                        frame.withUnsafeMutableBufferPointer { framePtr in
                            vDSP_vmul(
                                paddedPtr.baseAddress! + audioStart, 1,
                                windowPtr.baseAddress!, 1,
                                framePtr.baseAddress! + windowOffset, 1,
                                vDSP_Length(availableSamples)
                            )
                        }
                    }
                }
            }

            // Compute power spectrum using reusable buffers
            computePowerSpectrumInPlace()

            // Apply mel filterbank using vDSP matrix-vector multiply
            // melFrame = melFilterbankFlat * powerSpec (matrix [nMels x numFreqBins] * vector [numFreqBins])
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

            // Apply log(x + guard_value) and store in transposed layout [frameIdx, melIdx]
            for melIdx in 0..<nMels {
                mel[frameIdx * nMels + melIdx] = log(melFrame[melIdx] + logZeroGuard)
            }
        }

        return (mel: mel, melLength: numFrames, numFrames: numFrames)
    }

    /// Compute power spectrum in-place using pre-allocated buffers
    private func computePowerSpectrumInPlace() {
        guard let setup = fftSetup else { return }

        // Copy frame to real input and clear imaginary
        frame.withUnsafeBufferPointer { src in
            realIn.withUnsafeMutableBufferPointer { dst in
                _ = memcpy(dst.baseAddress!, src.baseAddress!, nFFT * MemoryLayout<Float>.size)
            }
        }
        vDSP_vclr(&imagIn, 1, vDSP_Length(nFFT))

        // Execute FFT
        vDSP_DFT_Execute(setup, realIn, imagIn, &realOut, &imagOut)

        // Compute power: real^2 + imag^2 using vDSP
        let numFreqBins = nFFT / 2 + 1
        // real^2
        vDSP_vsq(realOut, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
        // imag^2 into pre-allocated buffer
        vDSP_vsq(imagOut, 1, &imagSq, 1, vDSP_Length(numFreqBins))
        // powerSpec += imagSq
        vDSP_vadd(powerSpec, 1, imagSq, 1, &powerSpec, 1, vDSP_Length(numFreqBins))
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

    /// Compute power spectrum with per-call buffer allocation.
    /// - Note: This method allocates temporary buffers on each call. It's only used by `compute()`
    ///   which is kept for API compatibility. For performance-critical paths, use `computePowerSpectrumInPlace()`.
    private func computePowerSpectrum(frame: [Float]) -> [Float] {
        guard let setup = fftSetup else {
            return [Float](repeating: 0, count: nFFT / 2 + 1)
        }

        // Split into real and imaginary parts for vDSP
        var realIn = [Float](repeating: 0, count: nFFT)
        let imagIn = [Float](repeating: 0, count: nFFT)
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
