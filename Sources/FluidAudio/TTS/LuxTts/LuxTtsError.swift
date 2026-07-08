import Foundation

/// Errors surfaced by the LuxTTS backend.
public enum LuxTtsError: Error, LocalizedError {
    case notInitialized
    case downloadFailed(String)
    case modelFileNotFound(String)
    case corruptedModel(String, underlying: String)
    case tokenizerFailed(String)
    case invalidPromptAudio(String)
    case inputTooLong(String)
    case inferenceFailed(stage: String, underlying: String)
    /// Phase 1 ships no in-process text→espeak-IPA frontend; callers must
    /// supply pre-phonemized input via `synthesize(phonemes:...)`.
    case g2pUnavailable

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "LuxTTS is not initialized. Call initialize() first."
        case .downloadFailed(let detail):
            return "LuxTTS model download failed: \(detail)"
        case .modelFileNotFound(let name):
            return "LuxTTS model file not found: \(name)"
        case .corruptedModel(let name, let underlying):
            return "LuxTTS model \(name) failed to load: \(underlying)"
        case .tokenizerFailed(let detail):
            return "LuxTTS tokenizer failure: \(detail)"
        case .invalidPromptAudio(let detail):
            return "LuxTTS prompt audio invalid: \(detail)"
        case .inputTooLong(let detail):
            return "LuxTTS input exceeds the fixed CoreML shape bucket: \(detail)"
        case .inferenceFailed(let stage, let underlying):
            return "LuxTTS inference failed at \(stage): \(underlying)"
        case .g2pUnavailable:
            return
                "LuxTTS has no built-in text→phoneme frontend yet; pass espeak-style "
                + "IPA via synthesize(phonemes:promptPhonemes:...) (tokens.txt token set)."
        }
    }
}
