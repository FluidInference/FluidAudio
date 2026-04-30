import Foundation

/// Errors that can surface during Magpie TTS initialization or synthesis.
public enum MagpieError: Error, LocalizedError, Sendable {
    case notInitialized
    case modelFileNotFound(String)
    case corruptedModel(String, underlying: String)
    case downloadFailed(String)
    case invalidConstants(String)
    case unsupportedLanguage(String)
    case tokenizerDataMissing(language: String, file: String)
    case textTooLong(tokenCount: Int, maxLength: Int)
    case invalidNpyFile(path: String, reason: String)
    case inferenceFailed(stage: String, underlying: String)
    case invalidSpeakerIndex(Int)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Magpie TTS manager has not been initialized. Call initialize() first."
        case .modelFileNotFound(let name):
            return "Magpie model file not found: \(name)"
        case .corruptedModel(let name, let underlying):
            return "Magpie model appears corrupted: \(name) (\(underlying))"
        case .downloadFailed(let message):
            return "Magpie download failed: \(message)"
        case .invalidConstants(let message):
            return "Magpie constants invalid: \(message)"
        case .unsupportedLanguage(let code):
            return "Magpie does not support language code: \(code)"
        case .tokenizerDataMissing(let language, let file):
            return "Tokenizer data missing for \(language): \(file)"
        case .textTooLong(let tokenCount, let maxLength):
            return "Text produced \(tokenCount) tokens; Magpie accepts at most \(maxLength)."
        case .invalidNpyFile(let path, let reason):
            return "Invalid .npy file at \(path): \(reason)"
        case .inferenceFailed(let stage, let underlying):
            return "Magpie \(stage) inference failed: \(underlying)"
        case .invalidSpeakerIndex(let index):
            return "Invalid Magpie speaker index \(index) (valid range: 0..<\(MagpieConstants.numSpeakers))."
        }
    }
}
