@preconcurrency import CoreML
import Foundation

/// Native-Swift log-mel features for Nemotron streaming, a drop-in replacement
/// for the CoreML `preprocessor` model.
///
/// Reproduces NeMo's `AudioToMelSpectrogramPreprocessor` as configured in
/// `nvidia/nemotron-speech-streaming-en-0.6b`: `n_fft=512`,
/// `window_size=0.025` (400), `window_stride=0.01` (160), `features=128`,
/// `window=hann` (symmetric), `preemph=0.97`, `log` (`2^-24` additive guard),
/// and crucially **`normalize: NA`** — i.e. *no* per-feature standardization.
/// That is exactly `AudioMelSpectrogram`'s default front-end with the
/// normalization step omitted, unlike `UnifiedMelExtractor` whose model uses
/// `normalize: per_feature`.
///
/// Removing the CoreML preprocessor avoids its flexible-shape `RangeDim` audio
/// input, whose ANE `default_function` was built against a 1-sample lower bound
/// and raised `ios17.slice_by_index: zero shape error` on iPadOS cold starts
/// (issue #739).
struct NemotronMelExtractor {
    private let mel: AudioMelSpectrogram
    private let nMels: Int
    private let hopLength = 160

    init(nMels: Int = 128) {
        self.nMels = nMels
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

    /// Raw (unnormalized) log-mel for one chunk of audio, shaped
    /// `[1, nMels, T]` with `T = floor((count + n_fft - win) / hop) + 1` (NeMo
    /// center padding) — the same `mel` tensor the CoreML preprocessor produced,
    /// frame for frame. `mel_length` was ignored by the pipeline (it sets the
    /// encoder's `mel_length` to `config.totalMelFrames`), so it is not returned.
    func melSpectrogram(samples: [Float]) throws -> MLMultiArray {
        let result = mel.computeFlatTransposed(
            audio: samples,
            lastAudioSample: 0,
            paddingMode: .center,
            expectedFrameCount: nil
        )
        let flat = result.mel
        let totalFrames = result.numFrames

        let melArray = try MLMultiArray(
            shape: [1, NSNumber(value: nMels), NSNumber(value: totalFrames)], dataType: .float32)
        melArray.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            // Contiguous [1, nMels, T]: element [0, m, t] is at offset m*T + t,
            // sourced from the time-major flat buffer at t*nMels + m.
            for t in 0..<totalFrames {
                let base = t * nMels
                for m in 0..<nMels {
                    ptr[m * totalFrames + t] = flat[base + m]
                }
            }
        }
        return melArray
    }
}
