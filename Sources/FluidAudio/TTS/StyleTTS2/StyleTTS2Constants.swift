import Foundation

/// Compile-time constants for the StyleTTS2 LibriTTS (iteration_3) pipeline.
///
/// Mirrors the values baked into the CoreML packages — every number here
/// is dictated by the trace-time shapes / hyperparameters in
/// `mobius/models/tts/styletts2/coreml/inference.py`.
public enum StyleTTS2Constants {

    // MARK: - Audio
    /// Output sample rate. 24 kHz mono — the rate the HiFi-GAN generator
    /// was trained at.
    public static let sampleRate: Int = 24_000
    /// HiFi-GAN per-frame hop in samples (`24 kHz × 12.5 ms`). Used to
    /// pre-allocate output buffers from the predicted frame count.
    public static let hopSamples: Int = 300
    /// Mirror of `_runtime`'s tail trim — the decoder leaks 50 samples of
    /// generator-noise tail past the intended end-of-utterance.
    public static let tailTrimSamples: Int = 50

    // MARK: - Tokenizer / token axis buckets
    /// Default fixed token axis baked into `bert_fp16.mlmodelc` and
    /// `fused_diffusion_sampler_fp16.mlmodelc`. Anything longer needs a
    /// bucket variant.
    public static let defaultBertTokens: Int = 57
    /// Token axis sizes for the bucket variants. Ordered ascending so the
    /// loader can pick the smallest bucket that fits.
    public static let bucketTokenSizes: [Int] = [64, 128, 256]

    // MARK: - Reference-audio mel filterbank
    /// `torchaudio.transforms.MelSpectrogram` argument set used at training
    /// time. Note: the upstream `make_preprocess()` does **not** override
    /// `sample_rate`, so torchaudio defaults to 16 000 even though the audio
    /// is loaded at 24 kHz. The Swift extractor must replicate that quirk
    /// so the mel bins line up with what the model saw during training.
    public static let melFilterSampleRate: Int = 16_000
    public static let melNFFT: Int = 2_048
    public static let melWinLength: Int = 1_200
    public static let melHopLength: Int = 300
    public static let melNMels: Int = 80
    /// `(log(mel + 1e-5) - mean) / std` normalization constants from
    /// `make_preprocess()`.
    public static let melLogEpsilon: Float = 1e-5
    public static let melLogMean: Float = -4.0
    public static let melLogStd: Float = 4.0

    // MARK: - Diffusion / Karras schedule
    /// ADPM2 step count baked into `fused_diffusion_sampler_fp16.mlmodelc`.
    /// The fused graph requires exactly `noiseSteps - 1` auxiliary noise
    /// vectors (4 for the default 5-step schedule).
    public static let diffusionSteps: Int = 5
    public static let sigmaMin: Double = 0.0001
    public static let sigmaMax: Double = 3.0
    public static let rhoSchedule: Double = 9.0

    // MARK: - Style blend (α/β)
    /// Mixing weights between the diffusion-sampled style and the reference
    /// encoder output. Defaults match the upstream demo (`alpha=0.3`,
    /// `beta=0.7` → 30 % diffusion / 70 % reference for ref slot,
    /// 70 % diffusion / 30 % reference for prosody slot).
    public static let defaultAlpha: Float = 0.3
    public static let defaultBeta: Float = 0.7

    // MARK: - Style vector dimensions
    /// `s_pred` is `[1, 256]` after `squeeze(1)`. The first 128 channels
    /// are the AdaIN reference (`ref_diff`), the second 128 channels are
    /// the prosody (`s_diff`). `ref_encoder` returns the same 256-dim
    /// vector with the same split.
    public static let styleDim: Int = 256
    public static let refSplit: Int = 128

    // MARK: - Decoder type
    /// LibriTTS uses HiFi-GAN — the Python orchestrator gates the
    /// causal `_hifigan_shift` on this and the iteration_3 build is
    /// HiFi-GAN-only, so the Swift port hard-codes the shift on. If you
    /// ever swap in an istftnet checkpoint, set this to `false` and skip
    /// the shift.
    public static let applyHifiganAsrShift: Bool = true
}
