import Foundation

/// Errors that can surface during StyleTTS2 initialization or synthesis.
public enum StyleTTS2Error: Error, LocalizedError, Sendable {
    case notInitialized
    case modelFileNotFound(String)
    case corruptedModel(String, underlying: String)
    case downloadFailed(String)
    case unsupportedReferenceAudio(String)
    case textTooLong(tokenCount: Int, maxLength: Int)
    case noBucketAvailable(tokenCount: Int)
    case phonemizationFailed(String)
    case inferenceFailed(stage: String, underlying: String)
    case invalidTensorShape(stage: String, expected: String, got: String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "StyleTTS2 manager has not been initialized. Call initialize() first."
        case .modelFileNotFound(let name):
            return "StyleTTS2 model file not found: \(name)"
        case .corruptedModel(let name, let underlying):
            return "StyleTTS2 model appears corrupted: \(name) (\(underlying))"
        case .downloadFailed(let message):
            return "StyleTTS2 download failed: \(message)"
        case .unsupportedReferenceAudio(let message):
            return "StyleTTS2 reference audio could not be loaded: \(message)"
        case .textTooLong(let tokenCount, let maxLength):
            return "Text produced \(tokenCount) tokens; StyleTTS2 max bucket holds \(maxLength)."
        case .noBucketAvailable(let tokenCount):
            return
                "No StyleTTS2 token-axis bucket fits \(tokenCount) tokens (max bucket = \(StyleTTS2Constants.bucketTokenSizes.max() ?? 0))."
        case .phonemizationFailed(let message):
            return "StyleTTS2 phonemization failed: \(message)"
        case .inferenceFailed(let stage, let underlying):
            return "StyleTTS2 \(stage) inference failed: \(underlying)"
        case .invalidTensorShape(let stage, let expected, let got):
            return "StyleTTS2 \(stage) tensor shape mismatch: expected \(expected), got \(got)"
        }
    }
}
