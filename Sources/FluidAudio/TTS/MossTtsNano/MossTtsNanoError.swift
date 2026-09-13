import Foundation

/// Errors surfaced by the MOSS-TTS-Nano backend.
public enum MossTtsNanoError: Error, LocalizedError, Sendable {
    case notInitialized
    case modelFileNotFound(String)
    case corruptedModel(String, underlying: String)
    case downloadFailed(String)
    case configLoadFailed(String)
    case tokenizerLoadFailed(String)
    case voiceLoadFailed(path: String, underlying: String)
    case invalidVoice(String)
    case emptyText
    case promptTooLong(rows: Int, capacity: Int)
    case inferenceFailed(stage: String, underlying: String)
    case invalidTensorShape(stage: String, expected: String, got: String)
    case audioLoadFailed(path: String, underlying: String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "MossTtsNano manager has not been initialized. Call initialize() first."
        case .modelFileNotFound(let name):
            return "MossTtsNano model file not found: \(name)"
        case .corruptedModel(let name, let underlying):
            return "MossTtsNano model appears corrupted: \(name) (\(underlying))"
        case .downloadFailed(let message):
            return "MossTtsNano download failed: \(message)"
        case .configLoadFailed(let message):
            return "MossTtsNano config.json load failed: \(message)"
        case .tokenizerLoadFailed(let message):
            return "MossTtsNano tokenizer.model load failed: \(message)"
        case .voiceLoadFailed(let path, let underlying):
            return "MossTtsNano voice load failed at \(path): \(underlying)"
        case .invalidVoice(let message):
            return "MossTtsNano invalid voice: \(message)"
        case .emptyText:
            return "MossTtsNano received empty text after normalization."
        case .promptTooLong(let rows, let capacity):
            return
                "MossTtsNano prompt has \(rows) rows but the prefill graph holds \(capacity); use a shorter reference clip."
        case .inferenceFailed(let stage, let underlying):
            return "MossTtsNano \(stage) inference failed: \(underlying)"
        case .invalidTensorShape(let stage, let expected, let got):
            return "MossTtsNano \(stage) tensor shape mismatch: expected \(expected), got \(got)"
        case .audioLoadFailed(let path, let underlying):
            return "MossTtsNano could not load reference audio at \(path): \(underlying)"
        }
    }
}
