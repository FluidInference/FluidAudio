import Foundation

/// Errors raised by ``MultilingualG2PModel``.
public enum MultilingualG2PError: Error, LocalizedError {
    case modelLoadFailed(String)
    case encoderPredictionFailed
    case decoderPredictionFailed

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let detail):
            return "Failed to load multilingual G2P CoreML model: \(detail)"
        case .encoderPredictionFailed:
            return "Multilingual G2P encoder prediction failed."
        case .decoderPredictionFailed:
            return "Multilingual G2P decoder prediction failed."
        }
    }
}
