import Foundation

/// Errors thrown by the Inflect v2 TTS backend.
public enum InflectError: Error, LocalizedError {
    case notInitialized
    case downloadFailed(String)
    case modelFileNotFound(String)
    case corruptedModel(String, underlying: String)
    case phonemizationFailed(String)
    case inputProcessingFailed(String)
    /// A chunk's predicted duration exceeded the largest synthesizer bucket.
    case durationOverflow(frames: Int, maxFrames: Int)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Inflect backend is not initialized; call initialize() first."
        case .downloadFailed(let detail):
            return "Inflect model download failed: \(detail)"
        case .modelFileNotFound(let name):
            return "Inflect model file not found: \(name)"
        case .corruptedModel(let name, let underlying):
            return "Inflect model \(name) failed to load: \(underlying)"
        case .phonemizationFailed(let detail):
            return "Inflect phonemization failed: \(detail)"
        case .inputProcessingFailed(let detail):
            return "Inflect input processing failed: \(detail)"
        case .durationOverflow(let frames, let maxFrames):
            return
                "Predicted duration \(frames) frames exceeds the largest bucket "
                + "(\(maxFrames)); shorten the input chunk."
        case .predictionFailed(let detail):
            return "Inflect CoreML prediction failed: \(detail)"
        }
    }
}
