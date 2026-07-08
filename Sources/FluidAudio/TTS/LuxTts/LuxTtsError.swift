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
    case degenerateDuration(featuresLength: Int, tokensCount: Int)
    case inferenceFailed(stage: String, underlying: String)

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
        case .degenerateDuration(let featuresLength, let tokensCount):
            return
                "LuxTTS duration estimate is degenerate: features length \(featuresLength) < "
                + "token count \(tokensCount) yields < 1 frame per token, which would collapse "
                + "every frame onto the pad slot (prompt too short or too many tokens)"
        case .inferenceFailed(let stage, let underlying):
            return "LuxTTS inference failed at \(stage): \(underlying)"
        }
    }
}
