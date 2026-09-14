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
    /// Static prefill window baked into the prefill model. The window holds
    /// `[voice conditioning, text BPE tokens, BOS]`, and the built-in voice's
    /// conditioning is 376 rows — so the *usable text budget* is
    /// 512 − 376 − 1 = **135 BPE tokens** (roughly 500–550 characters), not
    /// 512 (#924).
    public static let prefillLength = 512
    /// KV-cache capacity baked into the decode model.
    public static let maxContext = 1536
    public static let hiddenSize = 768
    public static let layerCount = 12
    public static let kvHeads = 12
    public static let headDim = 64

    /// GPT2 BPE vocabulary size incl. added tokens (text table rows).
    public static let textVocabSize = 50276
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
    /// Flow token bucket (prompt + generated) of the `.standard` capacity
    /// (`FlowMean-N500`). The built-in voice's 250 prompt tokens and the 3
    /// appended silence tokens live inside this bucket, so the *usable
    /// generation budget* is 500 − 250 − 3 = **247 speech tokens ≈ 9.9 s of
    /// audio** (#924). Use `ChatterboxNanoOutputCapacity.extended` for ~3×
    /// that. Prefer `ChatterboxNanoOutputCapacity.flowTokenBucket`.
    public static let flowTokenBucket = 500
    /// Mel frames produced by the `.standard` flow bucket (2 per token) =
    /// HiFT bucket. Prefer `ChatterboxNanoOutputCapacity.melFrameBucket`.
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

/// Which S3Gen flow/vocoder bucket pair to download and load. The flow
/// bucket holds `voice prompt tokens + generated speech tokens + 3 silence
/// tokens`, so the audio each capacity can generate depends on the voice:
/// with the built-in voice (250 prompt tokens) `.standard` yields ≤247
/// generated tokens ≈ 9.9 s per call and `.extended` ≤747 ≈ 29.9 s.
///
/// `.extended` is a separate ~270 MB download (`FlowMean-N1000` +
/// `HiFT-T2000`) and roughly doubles the flow/vocoder latency per call —
/// the buckets are static shapes, so short outputs pay the full bucket.
public enum ChatterboxNanoOutputCapacity: String, CaseIterable, Sendable {
    /// `FlowMean-N500` + `HiFT-T1000` — ≈9.9 s of generated audio with the
    /// built-in voice.
    case standard
    /// `FlowMean-N1000` + `HiFT-T2000` — ≈29.9 s of generated audio with
    /// the built-in voice.
    case extended

    /// Flow token bucket (prompt + generated + silence) baked into the
    /// capacity's `FlowMean` model.
    public var flowTokenBucket: Int {
        switch self {
        case .standard: return 500
        case .extended: return 1000
        }
    }

    /// Mel frames produced by the flow bucket (2 per token) = HiFT bucket.
    public var melFrameBucket: Int { 2 * flowTokenBucket }

    /// Speech tokens available for generation once `promptTokens` (the
    /// loaded voice's prompt) and the appended silence tokens are counted.
    public func generationBudget(promptTokens: Int) -> Int {
        max(0, flowTokenBucket - promptTokens - ChatterboxNanoConstants.silenceTokenCount)
    }
}
