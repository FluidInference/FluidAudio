import Foundation

/// Errors that can occur during VoxCPM synthesis.
public enum VoxCpmError: LocalizedError {
    case downloadFailed(String)
    case corruptedModel(String)
    case modelNotFound(String)
    case processingFailed(String)
    case tokenizerFailed(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let message):
            return "Download failed: \(message)"
        case .corruptedModel(let name):
            return "Model \(name) is corrupted"
        case .modelNotFound(let name):
            return "Model \(name) not found"
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .tokenizerFailed(let message):
            return "Tokenizer failed: \(message)"
        }
    }
}
