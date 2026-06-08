import Foundation

/// Compile-time constants for the Supertonic-3 multilingual TTS pipeline.
///
/// Mirrors the hyperparameters published in the upstream `tts.json` and the
/// reference inference flow in
/// `https://github.com/supertone-inc/supertonic/blob/main/swift/Sources/Helper.swift`.
///
/// Supertonic-3 ships four ONNX models (text_encoder, duration_predictor,
/// vector_estimator, vocoder) totalling ~398 MB. FluidAudio re-publishes those
/// models as `.mlmodelc` bundles under
/// `FluidInference/supertonic-3-coreml`; see
/// `Scripts/convert_supertonic3_to_coreml.py` for the conversion recipe.
public enum Supertonic3Constants {

    // MARK: - Audio

    /// Vocoder output sample rate. 44.1 kHz mono Float32.
    public static let sampleRate: Int = 44_100

    // MARK: - Latent / chunking

    /// Base chunk size (samples) of the acoustic autoencoder. Drives the
    /// `latent_len = ceil(wav_len / (base_chunk_size * chunk_compress_factor))`
    /// calculation for the denoising loop. Matches `ae.base_chunk_size` in
    /// the published `tts.json` (Supertonic-3 v1.7.3).
    public static let baseChunkSize: Int = 512

    /// Chunk-compress factor used by the text-to-latent module. The flattened
    /// latent dimension passed to `vector_estimator` is
    /// `latent_dim * chunk_compress_factor`. Matches `ttl.chunk_compress_factor`.
    public static let chunkCompressFactor: Int = 6

    /// Per-chunk latent dimensionality before applying `chunk_compress_factor`.
    /// Matches `ttl.latent_dim` (== `ae.ldim`) in the published config.
    public static let latentDim: Int = 24

    /// Style token count expected by the text-to-latent style encoder
    /// (`style_ttl` shape is `[bsz, ttlStyleTokens, ttlStyleDim]`).
    public static let ttlStyleTokens: Int = 50

    /// Style embedding dim for `style_ttl` (matches `style_value_dim`).
    public static let ttlStyleDim: Int = 256

    /// Style token count expected by the duration-predictor style encoder
    /// (`style_dp` shape is `[bsz, dpStyleTokens, dpStyleDim]`).
    public static let dpStyleTokens: Int = 8

    /// Style embedding dim for `style_dp` (matches `dp.style_token_layer.style_value_dim`).
    public static let dpStyleDim: Int = 16

    /// Text-encoder output channel count fed into `vector_estimator.text_emb`.
    public static let textEmbDim: Int = 256

    /// Pinned text-token sequence length expected by `text_encoder` and
    /// `duration_predictor`. The CoreML conversion fixes the T axis at 128;
    /// the unicode processor pads/truncates inputs to match. Mirrors
    /// `TEXT_T_FIXED = 128` in the reference Python driver.
    public static let textTFixed: Int = 128

    // MARK: - Inference

    /// Default number of denoising steps for the vector_estimator loop. The
    /// reference CLI ships 8; lower values trade quality for latency.
    public static let defaultTotalSteps: Int = 8

    /// Default global speed factor applied to the predicted duration vector
    /// (`duration /= speed`). The reference CLI ships 1.05.
    public static let defaultSpeed: Float = 1.05

    /// Default silence inserted between text chunks when synthesizing long
    /// utterances. 0.3 s mirrors the reference CLI default.
    public static let defaultSilenceDuration: Float = 0.3

    /// Max characters per chunk when synthesizing long English/Latin text.
    /// Although `textTFixed = 128` would *fit* ~110 chars, the model's output
    /// degrades as a chunk approaches that token window (a 105-char single
    /// chunk scored 17.6% WER vs 0% at 71 chars), so the cap is held at 70 to
    /// keep every chunk in the clean regime. Longer text is split by
    /// `Supertonic3TextChunker` before reaching the encoder. See #669.
    public static let maxChunkLengthLatin: Int = 70

    /// Tighter chunk cap for Korean / Japanese. CJK expands to more codepoints
    /// per visible character after NFKD, so the same token-window budget holds
    /// fewer CJK characters — kept proportionally below `maxChunkLengthLatin`
    /// (same ~0.82 ratio as the original 90/110) to stay in the clean regime
    /// and tighter than Latin. See #669.
    public static let maxChunkLengthCJK: Int = 57

    // MARK: - Language whitelist (matches AVAILABLE_LANGS in the reference)

    /// 31 supported languages plus "na" (language-agnostic / numeric).
    public static let availableLanguages: [String] = [
        "en", "ko", "ja", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi",
        "fr", "hi", "hr", "hu", "id", "it", "lt", "lv", "nl", "pl", "pt", "ro",
        "ru", "sk", "sl", "sv", "tr", "uk", "vi", "na",
    ]

    /// Languages that should use the tighter `maxChunkLengthCJK` (57-char) chunker.
    public static let cjkLanguages: Set<String> = ["ko", "ja"]
}
