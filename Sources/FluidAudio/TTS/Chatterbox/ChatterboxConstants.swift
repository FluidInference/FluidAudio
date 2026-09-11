import Foundation

/// Compile-time constants for the Chatterbox Multilingual backend.
///
/// Pipeline: T3 (Llama-520M, CFG batch 2) emits S3 speech tokens at 25 Hz,
/// the S3Gen flow decoder maps them to 50 Hz mel frames, and the HiFT
/// vocoder renders 24 kHz audio. Models are converted in mobius
/// (`models/tts/chatterbox/coreml`) and published to
/// `FluidInference/chatterbox-multilingual-coreml`.
///
/// Note: upstream applies a Perth watermark to generated audio in the host
/// app; that postprocessing is not implemented here.
public enum ChatterboxConstants {
    public static let sampleRate = 24_000
    /// Audio samples per mel frame (50 mel frames/s at 24 kHz).
    public static let samplesPerMelFrame = 480

    // ---- T3 (token generator) ----
    /// Static prefill window baked into the prefill model.
    public static let prefillLength = 256
    /// KV-cache capacity baked into the decode model.
    public static let maxContext = 1024
    public static let hiddenSize = 1024
    public static let layerCount = 30
    public static let kvHeads = 16
    public static let headDim = 64
    /// Conditioning block length (speaker + 32 perceiver latents + emotion).
    public static let condLength = 34

    public static let startTextToken = 255
    public static let stopTextToken = 0
    public static let startSpeechToken = 6561
    public static let stopSpeechToken = 6562
    /// Valid S3 speech-token range for the flow decoder; generated ids at or
    /// above this (EOS) are dropped before vocoding.
    public static let speechVocabSize = 6561
    /// Full T3 output vocabulary (speech tokens + specials).
    public static let outputVocabSize = 8194

    // ---- S3Gen ----
    /// Flow token bucket (prompt + generated) baked into `Flow-N500`.
    public static let flowTokenBucket = 500
    /// Mel frames produced by the flow bucket (2 per token) = HiFT bucket.
    public static let melFrameBucket = 1000
    /// Harmonic channels in the HiFT source module (harmonics + fundamental).
    public static let hiftHarmonics = 9

    // ---- Upstream sampling defaults (mtl_tts.generate) ----
    public static let cfgWeight: Float = 0.5
    public static let temperature: Float = 0.8
    public static let repetitionPenalty: Float = 2.0
    public static let minP: Float = 0.05
    public static let topP: Float = 1.0
    public static let maxNewTokens = 1000

    /// Languages the Swift text frontend supports. The remaining upstream
    /// languages (zh, ja, he, ko, ru) require language-specific text
    /// transforms (Cangjie codes, kana normalization, diacritization) that
    /// are not yet ported.
    public static let supportedLanguages: Set<String> = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "hi", "it",
        "ms", "nl", "no", "pl", "pt", "sv", "sw", "tr",
    ]
    public static let unsupportedLanguages: Set<String> = ["zh", "ja", "he", "ko", "ru"]

    public static let defaultLanguage = "en"
    public static let defaultVoice = "default"
}
