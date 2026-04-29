import Foundation

/// Constants for the StyleTTS2 4-stage diffusion TTS backend.
///
/// Reference: `FluidInference/StyleTTS-2-coreml` (LibriTTS multi-speaker
/// checkpoint). The companion `config.json` shipped with the repo carries
/// the same numbers in machine-readable form; values here mirror that
/// contract so the framework can run without parsing the bundle config
/// before model load.
public enum StyleTTS2Constants {

    // MARK: - Audio

    public static let audioSampleRate: Int = 24_000
    /// HiFi-GAN hop size — `mel_frames * hopSize == samples`.
    public static let hopSize: Int = 300
    public static let melChannels: Int = 80
    public static let nFFT: Int = 2_048
    public static let winLength: Int = 1_200

    // MARK: - Tokenizer

    /// 178-token espeak-ng IPA + stress vocabulary (mirrors
    /// `text_utils.TextCleaner` from upstream StyleTTS2).
    public static let vocabSize: Int = 178
    /// Pad token id (id 0 in the upstream cleaner table).
    public static let padTokenId: Int = 0

    // MARK: - Model dimensions

    /// Style/reference vector channels per branch (acoustic, prosody).
    public static let styleDim: Int = 128
    /// Concat of acoustic + prosody style vectors (`ref_s` input dim).
    public static let refStyleDim: Int = 256
    /// BERT/text predictor hidden size.
    public static let hiddenDim: Int = 512

    // MARK: - Sampler (ADPM2 + Karras schedule + CFG)

    /// Default number of diffusion sampler steps (5× per utterance).
    public static let defaultDiffusionSteps: Int = 5
    /// Karras schedule rho (controls sigma curvature). Matches the upstream
    /// e2e reference (`99b_e2e_coreml.py`) which uses rho=9.0 — *not* the
    /// k-diffusion default of 7.0.
    public static let karrasRho: Float = 9.0
    public static let karrasSigmaMin: Float = 0.0001
    public static let karrasSigmaMax: Float = 3.0
    /// Classifier-free guidance scale applied during the diffusion step.
    public static let cfgScale: Float = 1.0
    /// Diffusion step model is shipped only at this `bert_dur` bucket.
    public static let diffusionBucket: Int = 512

    // MARK: - Bucket selection

    /// Token-length buckets shipped for `text_predictor`.
    public static let textPredictorBuckets: [Int] = [32, 64, 128, 256, 512]
    /// Mel-frame buckets shipped for the HiFi-GAN decoder.
    public static let decoderBuckets: [Int] = [256, 512, 1024, 2048, 4096]

    // MARK: - Repository

    public static let defaultModelsSubdirectory: String = "Models"
}
