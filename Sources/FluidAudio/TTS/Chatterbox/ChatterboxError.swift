import Foundation

/// Errors thrown by the Chatterbox Multilingual backend.
public enum ChatterboxError: Error, LocalizedError {
    case downloadFailed(String)
    case modelNotFound(String)
    case malformedAsset(String)
    case unsupportedLanguage(String)
    case textTooLong(tokens: Int, max: Int)
    case generationTooLong(tokens: Int, max: Int)
    case processingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let detail):
            return "Chatterbox download failed: \(detail)"
        case .modelNotFound(let name):
            return "Chatterbox model not found: \(name)"
        case .malformedAsset(let detail):
            return "Chatterbox asset is malformed: \(detail)"
        case .unsupportedLanguage(let lang):
            return
                "Chatterbox language '\(lang)' is not supported by the Swift frontend yet "
                + "(needs language-specific text preprocessing). Supported: "
                + ChatterboxConstants.supportedLanguages.sorted().joined(separator: ", ")
        case .textTooLong(let tokens, let max):
            return "Text tokenizes to \(tokens) tokens; the prefill window holds \(max)"
        case .generationTooLong(let tokens, let max):
            return "Generated \(tokens) speech tokens; the flow bucket holds \(max)"
        case .processingFailed(let detail):
            return "Chatterbox synthesis failed: \(detail)"
        }
    }
}
