import Foundation

/// Compile-time constants for the Chatterbox Nano backend (beta model
/// conversion — artifacts and defaults may change).
///
/// Pipeline: T3 (GPT2-small, 110M, batch 1 — no CFG, no alignment analyzer)
/// emits S3 speech tokens at 25 Hz, the S3Gen meanflow decoder maps them to
/// 50 Hz mel frames in 2 plain Euler steps, and the HiFT vocoder renders
/// 24 kHz audio. Models are converted in mobius
/// (`models/tts/chatterbox/coreml`) and published to
/// `FluidInference/chatterbox-nano-coreml`.
///
/// Note: upstream applies a Perth watermark to generated audio in the host
/// app; that postprocessing is not implemented here.
public enum ChatterboxNanoConstants {
    public static let sampleRate = 24_000
    /// Audio samples per mel frame (50 mel frames/s at 24 kHz).
    public static let samplesPerMelFrame = 480

    // ---- T3 (token generator) ----
    /// Static prefill window baked into the prefill model.
    public static let prefillLength = 512
    /// KV-cache capacity baked into the decode model.
    public static let maxContext = 1536
    public static let hiddenSize = 768
    public static let layerCount = 12
    public static let kvHeads = 12
    public static let headDim = 64

    public static let startSpeechToken = 6561
    public static let stopSpeechToken = 6562
    /// Valid S3 speech-token range for the flow decoder; generated ids at or
    /// above this (BOS/EOS) are dropped before vocoding.
    public static let speechVocabSize = 6561
    /// Full T3 output vocabulary (speech tokens + BOS + EOS).
    public static let outputVocabSize = 6563

    // ---- S3Gen ----
    /// Upstream appends three silence tokens before vocoding (`S3GEN_SIL`).
    public static let silenceToken = 4299
    public static let silenceTokenCount = 3
    /// Flow token bucket (prompt + generated) baked into `FlowMean-N500`.
    public static let flowTokenBucket = 500
    /// Mel frames produced by the flow bucket (2 per token) = HiFT bucket.
    public static let melFrameBucket = 1000
    /// Harmonic channels in the HiFT source module (harmonics + fundamental).
    public static let hiftHarmonics = 9

    // ---- Upstream sampling defaults (tts_turbo.generate) ----
    public static let temperature: Float = 0.8
    public static let topK = 1000
    public static let topP: Float = 0.95
    public static let repetitionPenalty: Float = 1.2
    public static let maxNewTokens = 1000

    public static let defaultVoice = "default"
}
