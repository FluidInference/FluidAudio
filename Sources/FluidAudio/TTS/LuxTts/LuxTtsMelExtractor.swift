import Accelerate
import Foundation

/// Native Swift port of the upstream LuxTTS mel frontend (`VocosFbank`):
/// `torchaudio.transforms.MelSpectrogram(sample_rate: 24000, n_fft: 1024,
/// hop_length: 256, n_mels: 100, center: true, power: 1)` followed by
/// `log(clamp(min: 1e-7))`, with the frame count adjusted to lhotse's
/// `compute_num_frames` (truncate, or replicate-pad the last frame).
///
/// torchaudio specifics mirrored here (they differ from the NeMo-flavoured
/// `AudioMelSpectrogram`): periodic Hann window, reflect padding, magnitude
/// (power = 1) spectrum, HTK mel scale with no filterbank normalization.
///
/// Structure mirrors `NemotronMelExtractor` / `AudioMelSpectrogram`.
public final class LuxTtsMelExtractor {

    private let nFFT = LuxTtsConstants.nFFT
    private let hop = LuxTtsConstants.hopLength
    private let nMels = LuxTtsConstants.nMels
    private let sampleRate = LuxTtsConstants.melSampleRate

    private let hannWindow: [Float]
    private let melFilterbankFlat: [Float]  // [nMels x (nFFT/2+1)] row-major
    // Immutable after init (only read in extract, freed in deinit).
    private let fftSetup: vDSP_DFT_Setup?

    public init() {
        // The window/filterbank are call-independent pure-value tables; pull
        // them from a process-wide Sendable cache so repeated construction
        // (the synthesizer builds one extractor per call to stay Sendable)
        // only pays for the cheap vDSP DFT setup, not the table math.
        let tables = Self.sharedTables
        self.hannWindow = tables.hannWindow
        self.melFilterbankFlat = tables.melFilterbankFlat
        self.fftSetup = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(nFFT), .FORWARD)
    }

    deinit {
        if let setup = fftSetup {
            vDSP_DFT_DestroySetup(setup)
        }
    }

    /// lhotse `compute_num_frames`: `(num_samples + hop/2) / hop`.
    public func frameCount(sampleCount: Int) -> Int {
        (sampleCount + hop / 2) / hop
    }

    /// Compute the log-mel spectrogram of 24 kHz mono audio.
    /// - Returns: `[T][nMels]` log-mel frames (unscaled — the caller applies
    ///   `LuxTtsConstants.featScale`), `T = frameCount(sampleCount:)`.
    public func extract(audio: [Float]) -> [[Float]] {
        let n = audio.count
        let targetFrames = frameCount(sampleCount: n)
        guard n > 0, targetFrames > 0, let setup = fftSetup else { return [] }

        // Reflect-pad by nFFT/2 on both sides (torch pad_mode="reflect").
        let pad = nFFT / 2
        var padded = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<pad {
            padded[i] = audio[min(pad - i, n - 1)]
            padded[pad + n + i] = audio[max(n - 2 - i, 0)]
        }
        padded.replaceSubrange(pad..<(pad + n), with: audio)

        // torchaudio frame count with center=true: 1 + floor(n / hop).
        let stftFrames = 1 + n / hop

        let bins = nFFT / 2 + 1
        var realIn = [Float](repeating: 0, count: nFFT)
        let imagIn = [Float](repeating: 0, count: nFFT)
        var realOut = [Float](repeating: 0, count: nFFT)
        var imagOut = [Float](repeating: 0, count: nFFT)
        var magnitude = [Float](repeating: 0, count: bins)
        var imagSq = [Float](repeating: 0, count: bins)
        var melFrame = [Float](repeating: 0, count: nMels)
        var logMel = [Float](repeating: 0, count: nMels)
        var floor = LuxTtsConstants.logMelFloor
        var melCount = Int32(nMels)

        var frames: [[Float]] = []
        frames.reserveCapacity(targetFrames)

        for frameIdx in 0..<min(stftFrames, targetFrames) {
            let start = frameIdx * hop
            padded.withUnsafeBufferPointer { src in
                hannWindow.withUnsafeBufferPointer { win in
                    realIn.withUnsafeMutableBufferPointer { dst in
                        vDSP_vmul(
                            src.baseAddress! + start, 1,
                            win.baseAddress!, 1,
                            dst.baseAddress!, 1,
                            vDSP_Length(nFFT))
                    }
                }
            }

            vDSP_DFT_Execute(setup, realIn, imagIn, &realOut, &imagOut)

            // Magnitude (power = 1): sqrt(re^2 + im^2).
            vDSP_vsq(realOut, 1, &magnitude, 1, vDSP_Length(bins))
            vDSP_vsq(imagOut, 1, &imagSq, 1, vDSP_Length(bins))
            vDSP_vadd(magnitude, 1, imagSq, 1, &magnitude, 1, vDSP_Length(bins))
            var count = Int32(bins)
            vvsqrtf(&magnitude, magnitude, &count)

            melFilterbankFlat.withUnsafeBufferPointer { fb in
                magnitude.withUnsafeBufferPointer { mag in
                    melFrame.withUnsafeMutableBufferPointer { out in
                        vDSP_mmul(
                            fb.baseAddress!, 1,
                            mag.baseAddress!, 1,
                            out.baseAddress!, 1,
                            vDSP_Length(nMels), 1, vDSP_Length(bins))
                    }
                }
            }

            // log(max(mel, floor)) vectorized: clamp to the floor (vDSP_vthr)
            // then natural log (vvlogf) into the reused logMel buffer.
            vDSP_vthr(melFrame, 1, &floor, &logMel, 1, vDSP_Length(nMels))
            vvlogf(&logMel, logMel, &melCount)
            frames.append(logMel)
        }

        // lhotse alignment: replicate the last frame if the STFT produced
        // fewer frames than compute_num_frames asks for.
        while frames.count < targetFrames, let last = frames.last {
            frames.append(last)
        }
        return frames
    }

    // MARK: - Static tables

    /// Call-independent tables shared across every extractor instance.
    private struct Tables: Sendable {
        let hannWindow: [Float]
        let melFilterbankFlat: [Float]
    }

    /// Built once per process (pure `[Float]` values, immutable → Sendable).
    private static let sharedTables: Tables = {
        let nFFT = LuxTtsConstants.nFFT
        return Tables(
            hannWindow: periodicHannWindow(length: nFFT),
            melFilterbankFlat: htkMelFilterbank(
                nFFT: nFFT, nMels: LuxTtsConstants.nMels,
                sampleRate: LuxTtsConstants.melSampleRate))
    }()

    private static func periodicHannWindow(length: Int) -> [Float] {
        (0..<length).map { i in
            0.5 * (1.0 - cos(2.0 * Float.pi * Float(i) / Float(length)))
        }
    }

    /// torchaudio `melscale_fbanks(norm: nil, mel_scale: "htk")`, flattened
    /// row-major `[nMels x bins]` for `vDSP_mmul`.
    private static func htkMelFilterbank(nFFT: Int, nMels: Int, sampleRate: Int) -> [Float] {
        let bins = nFFT / 2 + 1
        let fMax = Double(sampleRate) / 2.0

        func hzToMel(_ hz: Double) -> Double { 2595.0 * log10(1.0 + hz / 700.0) }
        func melToHz(_ mel: Double) -> Double { 700.0 * (pow(10.0, mel / 2595.0) - 1.0) }

        let melMin = hzToMel(0)
        let melMax = hzToMel(fMax)
        let melPoints = (0..<(nMels + 2)).map { i in
            melToHz(melMin + Double(i) * (melMax - melMin) / Double(nMels + 1))
        }
        // all_freqs = linspace(0, sr/2, bins)
        let freqs = (0..<bins).map { Double($0) * fMax / Double(bins - 1) }

        var flat = [Float](repeating: 0, count: nMels * bins)
        for m in 0..<nMels {
            let fLeft = melPoints[m]
            let fCenter = melPoints[m + 1]
            let fRight = melPoints[m + 2]
            for b in 0..<bins {
                let up = (freqs[b] - fLeft) / (fCenter - fLeft)
                let down = (fRight - freqs[b]) / (fRight - fCenter)
                flat[m * bins + b] = Float(max(0.0, min(up, down)))
            }
        }
        return flat
    }
}
