import Foundation

/// Errors emitted by the KokoroLai TTS chain.
public enum KokoroLaiError: Error, LocalizedError {
    case modelNotLoaded(String)
    case downloadFailed(String)
    case vocabMissing(URL)
    case voicePackMissing(URL)
    case invalidVoicePack(String)
    case unsupportedPhoneme(Character)
    case phonemeSequenceTooLong(Int)
    case inputProcessingFailed(String)
    case acousticFramesExceedCap(have: Int, cap: Int)
    case predictionFailed(stage: String, underlying: Error)
    case unexpectedOutputShape(stage: String, expected: String, got: String)
    case audioConversionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotLoaded(let name):
            return "KokoroLai model '\(name)' not loaded. Call initialize() first."
        case .downloadFailed(let detail):
            return "KokoroLai download failed: \(detail)"
        case .vocabMissing(let url):
            return "KokoroLai vocab.json not found at \(url.path)."
        case .voicePackMissing(let url):
            return "KokoroLai voice pack not found at \(url.path)."
        case .invalidVoicePack(let detail):
            return "KokoroLai voice pack is invalid: \(detail)"
        case .unsupportedPhoneme(let ch):
            return "KokoroLai vocab does not contain phoneme '\(ch)'."
        case .phonemeSequenceTooLong(let n):
            return "KokoroLai phoneme sequence has \(n) characters (max \(KokoroLaiConstants.maxPhonemeLength))."
        case .inputProcessingFailed(let detail):
            return "KokoroLai input processing failed: \(detail)"
        case .acousticFramesExceedCap(let have, let cap):
            return "KokoroLai PostAlbert produced T_a=\(have) frames > MAX_FRAMES=\(cap). Chunk the input."
        case .predictionFailed(let stage, let err):
            return "KokoroLai stage '\(stage)' failed: \(err.localizedDescription)"
        case .unexpectedOutputShape(let stage, let expected, let got):
            return "KokoroLai stage '\(stage)' returned unexpected shape (expected \(expected), got \(got))."
        case .audioConversionFailed(let detail):
            return "KokoroLai audio conversion failed: \(detail)"
        }
    }
}
