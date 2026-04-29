import Foundation

/// Errors that can occur during StyleTTS2 synthesis.
public enum StyleTTS2Error: LocalizedError {
    case downloadFailed(String)
    case corruptedModel(String)
    case modelNotFound(String)
    case processingFailed(String)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let message):
            return "StyleTTS2 download failed: \(message)"
        case .corruptedModel(let name):
            return "StyleTTS2 model \(name) is corrupted"
        case .modelNotFound(let name):
            return "StyleTTS2 model \(name) not found"
        case .processingFailed(let message):
            return "StyleTTS2 processing failed: \(message)"
        case .invalidConfiguration(let message):
            return "StyleTTS2 invalid configuration: \(message)"
        }
    }
}
