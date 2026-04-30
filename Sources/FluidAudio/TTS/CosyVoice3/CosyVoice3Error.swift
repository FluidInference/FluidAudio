import Foundation

/// Errors surfaced by the CosyVoice3 Swift pipeline.
public enum CosyVoice3Error: LocalizedError, Sendable {
    case notInitialized
    case modelFileNotFound(String)
    case invalidFixture(String)
    case invalidSafetensors(String)
    case prefillTooLong(Int)
    case sequenceTooLong(Int)
    case predictionFailed(String)
    case embeddingTableMissing(String)
    case invalidShape(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "CosyVoice3 pipeline not initialized — call loadIfNeeded() first."
        case .modelFileNotFound(let path):
            return "CosyVoice3 model file not found at: \(path)"
        case .invalidFixture(let reason):
            return "Invalid CosyVoice3 fixture: \(reason)"
        case .invalidSafetensors(let reason):
            return "Invalid safetensors file: \(reason)"
        case .prefillTooLong(let length):
            return "Prefill sequence length \(length) exceeds max \(CosyVoice3Constants.prefillLength)"
        case .sequenceTooLong(let length):
            return "KV cache length \(length) exceeds max \(CosyVoice3Constants.kvMaxLength)"
        case .predictionFailed(let stage):
            return "CosyVoice3 prediction failed at stage: \(stage)"
        case .embeddingTableMissing(let name):
            return "CosyVoice3 embedding table missing: \(name)"
        case .invalidShape(let detail):
            return "CosyVoice3 shape mismatch: \(detail)"
        }
    }
}
