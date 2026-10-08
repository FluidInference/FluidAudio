import Foundation

/// Errors thrown by the Paradee TTS backend.
public enum ParadeeError: Error, LocalizedError {
    case notInitialized
    case downloadFailed(String)
    case modelFileNotFound(String)
    case corruptedModel(String, underlying: String)
    /// `.all` / `.cpuAndGPU` abort in MPSGraph on the LSTMs.
    case unsupportedComputeUnits(String)
    case inputProcessingFailed(String)
    /// A chunk's predicted duration exceeded the acoustic model's frame axis.
    case durationOverflow(frames: Int, maxFrames: Int)
    case predictionFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Paradee backend is not initialized; call initialize() first."
        case .downloadFailed(let detail):
            return "Paradee model download failed: \(detail)"
        case .modelFileNotFound(let name):
            return "Paradee model file not found: \(name)"
        case .corruptedModel(let name, let underlying):
            return "Paradee model \(name) failed to load: \(underlying)"
        case .unsupportedComputeUnits(let units):
            return
                "Paradee does not support compute units \(units): the LSTMs abort on the GPU. "
                + "Use .cpuOnly or .cpuAndNeuralEngine."
        case .inputProcessingFailed(let detail):
            return "Paradee input processing failed: \(detail)"
        case .durationOverflow(let frames, let maxFrames):
            return "Predicted duration \(frames) frames exceeds the model limit (\(maxFrames)); shorten the input."
        case .predictionFailed(let detail):
            return "Paradee CoreML prediction failed: \(detail)"
        }
    }
}
