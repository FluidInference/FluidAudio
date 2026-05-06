import Foundation

/// Errors emitted by the StyleTTS2 7-stage CoreML chain.
public enum StyleTTS2Error: Error, LocalizedError {
    case downloadFailed(String)
    case corruptedModel(String)
    case modelNotFound(String)
    case modelNotLoaded(String)
    case processingFailed(String)
    case invalidConfiguration(String)
    case voiceFileMissing(URL)
    case invalidVoiceFile(String)
    case phonemeSequenceTooLong(Int)
    case acousticFramesExceedCap(have: Int, cap: Int)
    case predictionFailed(stage: String, underlying: Error)
    case unexpectedOutputShape(stage: String, expected: String, got: String)
    case audioConversionFailed(String)
    case inputProcessingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .downloadFailed(let detail):
            return "StyleTTS2 download failed: \(detail)"
        case .corruptedModel(let name):
            return "StyleTTS2 model \(name) is corrupted"
        case .modelNotFound(let name):
            return "StyleTTS2 model \(name) not found"
        case .modelNotLoaded(let name):
            return "StyleTTS2 model '\(name)' not loaded. Call initialize() first."
        case .processingFailed(let message):
            return "StyleTTS2 processing failed: \(message)"
        case .invalidConfiguration(let message):
            return "StyleTTS2 invalid configuration: \(message)"
        case .voiceFileMissing(let url):
            return "StyleTTS2 ref_s voice file not found at \(url.path)."
        case .invalidVoiceFile(let detail):
            return "StyleTTS2 ref_s voice file is invalid: \(detail)"
        case .phonemeSequenceTooLong(let n):
            return "StyleTTS2 phoneme sequence has \(n) tokens (max \(StyleTTS2Constants.maxInputTokens))."
        case .acousticFramesExceedCap(let have, let cap):
            return "StyleTTS2 produced T_a=\(have) frames > MAX_FRAMES=\(cap). Chunk the input."
        case .predictionFailed(let stage, let err):
            return "StyleTTS2 stage '\(stage)' failed: \(err.localizedDescription)"
        case .unexpectedOutputShape(let stage, let expected, let got):
            return
                "StyleTTS2 stage '\(stage)' returned unexpected shape (expected \(expected), got \(got))."
        case .audioConversionFailed(let detail):
            return "StyleTTS2 audio conversion failed: \(detail)"
        case .inputProcessingFailed(let detail):
            return "StyleTTS2 input processing failed: \(detail)"
        }
    }
}
