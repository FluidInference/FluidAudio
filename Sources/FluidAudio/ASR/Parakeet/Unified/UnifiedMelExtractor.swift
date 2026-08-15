@preconcurrency import CoreML
import Foundation

/// Native-Swift log-mel features for Parakeet Unified, a drop-in replacement for
/// the CoreML `parakeet_unified_preprocessor` bundle.
///
/// Reproduces NeMo's `AudioToMelSpectrogramPreprocessor` exactly as configured in
/// `nvidia/parakeet-unified-en-0.6b` (`model_config.yaml`):
/// `n_fft=512`, `window_size=0.025` (400), `window_stride=0.01` (160),
/// `features=128`, `window=hann` (symmetric), `preemph=0.97`,
/// `log_zero_guard='add'/2^-24`, and crucially **`normalize: per_feature`** — the
/// per-mel-bin mean/variance standardization that the EOU config (`normalize:
/// "NA"`) omits. `AudioMelSpectrogram` supplies the spectrogram; this type adds
/// the per-feature normalization and packs the fixed-shape encoder input.
struct UnifiedMelExtractor {
    private let mel: AudioMelSpectrogram
    private let nMels: Int
    private let hopLength = 160

    /// Total samples in the (zero-padded) encoder window.
    let windowSamples: Int
    /// Fixed mel frame count the encoder expects for the window (center-padded:
    /// `windowSamples / hop + 1`). Matches the CoreML `mel_shape` last dim.
    let totalFrames: Int

    init(windowSamples: Int, nMels: Int = 128) {
        self.windowSamples = windowSamples
        self.nMels = nMels
        self.totalFrames = windowSamples / hopLength + 1
        self.mel = AudioMelSpectrogram(
            sampleRate: 16000,
            nMels: nMels,
            nFFT: 512,
            hopLength: 160,
            winLength: 400,
            preemph: 0.97,
            padTo: 0,
            windowPeriodic: false
        )
    }

    /// Compute `(mel, melLength)` for one encoder window.
    ///
    /// - Parameters:
    ///   - window: zero-padded buffer of length `windowSamples`; the first
    ///     `validCount` samples are real audio (the rest is padding), matching how
    ///     the managers fill the CoreML preprocessor's `audio_signal` input.
    ///   - validCount: number of real samples at the head of `window`.
    /// - Returns: `mel` shaped `[1, nMels, totalFrames]` and `melLength` (`[1]`,
    ///   int32) holding the number of valid mel frames — identical contract to the
    ///   CoreML preprocessor outputs.
    func features(window: [Float], validCount: Int) throws -> (mel: MLMultiArray, length: MLMultiArray) {
        // Time-major [totalFrames * nMels] raw log-mel over the full window.
        var flat = mel.computeFlatTransposed(
            audio: window,
            lastAudioSample: 0,
            paddingMode: .center,
            expectedFrameCount: totalFrames
        ).mel

        // NeMo `get_seq_len` (center padding, stft_pad_amount=0):
        // floor((L + n_fft - n_fft) / hop) = floor(L / hop). Note: NO `+1` — the
        // mel *tensor* has `floor(L/hop)+1` frames, but the valid length reported
        // to the encoder (and used for per_feature stats + masking) is one less,
        // so the final tensor frame is always normalized to 0. Getting this off by
        // one only nudges WER at large chunks but balloons it at tiny chunks,
        // whose decoded frames sit right at this boundary.
        let validFrames = min(validCount / hopLength, totalFrames)
        normalizePerFeature(&flat, frames: totalFrames, validFrames: validFrames)

        let melArray = try MLMultiArray(
            shape: [1, NSNumber(value: nMels), NSNumber(value: totalFrames)], dataType: .float32)
        melArray.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            // Contiguous [1, nMels, T]: element [0, m, t] is at offset m*T + t.
            for t in 0..<totalFrames {
                let base = t * nMels
                for m in 0..<nMels {
                    ptr[m * totalFrames + t] = flat[base + m]
                }
            }
        }

        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray[0] = NSNumber(value: validFrames)
        return (melArray, lengthArray)
    }

    /// NeMo `normalize_batch(..., normalize_type="per_feature")`: for each mel bin,
    /// subtract the mean and divide by the unbiased std (`+1e-5`) computed over the
    /// valid frames, applied to all valid frames; pad frames are zeroed.
    private func normalizePerFeature(_ x: inout [Float], frames: Int, validFrames: Int) {
        guard validFrames > 0 else {
            for i in x.indices { x[i] = 0 }
            return
        }
        let denom = Float(validFrames > 1 ? validFrames - 1 : 1)
        for m in 0..<nMels {
            var mean: Float = 0
            for t in 0..<validFrames { mean += x[t * nMels + m] }
            mean /= Float(validFrames)

            var varSum: Float = 0
            for t in 0..<validFrames {
                let d = x[t * nMels + m] - mean
                varSum += d * d
            }
            let std = (varSum / denom).squareRoot() + 1e-5

            for t in 0..<frames {
                x[t * nMels + m] = t < validFrames ? (x[t * nMels + m] - mean) / std : 0
            }
        }
    }
}
