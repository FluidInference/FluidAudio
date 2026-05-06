import Foundation

/// Errors emitted by the StyleTTS2-ANE 7-stage CoreML chain.
public enum StyleTTS2AneError: Error, LocalizedError {
    case modelNotLoaded(String)
    case downloadFailed(String)
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
        case .modelNotLoaded(let name):
            return "StyleTTS2-ANE model '\(name)' not loaded. Call initialize() first."
        case .downloadFailed(let detail):
            return "StyleTTS2-ANE download failed: \(detail)"
        case .voiceFileMissing(let url):
            return "StyleTTS2-ANE ref_s voice file not found at \(url.path)."
        case .invalidVoiceFile(let detail):
            return "StyleTTS2-ANE ref_s voice file is invalid: \(detail)"
        case .phonemeSequenceTooLong(let n):
            return "StyleTTS2-ANE phoneme sequence has \(n) tokens (max \(StyleTTS2AneConstants.maxInputTokens))."
        case .acousticFramesExceedCap(let have, let cap):
            return "StyleTTS2-ANE produced T_a=\(have) frames > MAX_FRAMES=\(cap). Chunk the input."
        case .predictionFailed(let stage, let err):
            return "StyleTTS2-ANE stage '\(stage)' failed: \(err.localizedDescription)"
        case .unexpectedOutputShape(let stage, let expected, let got):
            return
                "StyleTTS2-ANE stage '\(stage)' returned unexpected shape (expected \(expected), got \(got))."
        case .audioConversionFailed(let detail):
            return "StyleTTS2-ANE audio conversion failed: \(detail)"
        case .inputProcessingFailed(let detail):
            return "StyleTTS2-ANE input processing failed: \(detail)"
        }
    }
}
