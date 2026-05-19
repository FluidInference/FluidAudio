import Foundation

/// Errors that can surface during Supertonic-3 initialization or synthesis.
public enum Supertonic3Error: Error, LocalizedError, Sendable {
    case notInitialized
    case modelFileNotFound(String)
    case corruptedModel(String, underlying: String)
    case downloadFailed(String)
    case unsupportedLanguage(String)
    case voiceStyleLoadFailed(path: String, underlying: String)
    case voiceStyleShapeMismatch(component: String, expected: [Int], got: [Int])
    case unicodeIndexerLoadFailed(String)
    case configLoadFailed(String)
    case textTooLong(charCount: Int, maxLength: Int)
    case inferenceFailed(stage: String, underlying: String)
    case invalidTensorShape(stage: String, expected: String, got: String)
    case emptyText

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Supertonic3 manager has not been initialized. Call initialize() first."
        case .modelFileNotFound(let name):
            return "Supertonic3 model file not found: \(name)"
        case .corruptedModel(let name, let underlying):
            return "Supertonic3 model appears corrupted: \(name) (\(underlying))"
        case .downloadFailed(let message):
            return "Supertonic3 download failed: \(message)"
        case .unsupportedLanguage(let lang):
            let available = Supertonic3Constants.availableLanguages.joined(separator: ", ")
            return "Supertonic3 unsupported language '\(lang)'. Available: \(available)"
        case .voiceStyleLoadFailed(let path, let underlying):
            return "Supertonic3 voice style load failed at \(path): \(underlying)"
        case .voiceStyleShapeMismatch(let component, let expected, let got):
            return "Supertonic3 voice style \(component) shape mismatch: expected \(expected), got \(got)"
        case .unicodeIndexerLoadFailed(let message):
            return "Supertonic3 unicode_indexer.json load failed: \(message)"
        case .configLoadFailed(let message):
            return "Supertonic3 tts.json load failed: \(message)"
        case .textTooLong(let charCount, let maxLength):
            return "Supertonic3 input chunk has \(charCount) chars; max chunk length is \(maxLength)."
        case .inferenceFailed(let stage, let underlying):
            return "Supertonic3 \(stage) inference failed: \(underlying)"
        case .invalidTensorShape(let stage, let expected, let got):
            return "Supertonic3 \(stage) tensor shape mismatch: expected \(expected), got \(got)"
        case .emptyText:
            return "Supertonic3 received empty text after normalization."
        }
    }
}
