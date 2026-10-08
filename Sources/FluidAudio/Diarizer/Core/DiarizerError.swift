import Foundation

// Outside the `Diarizer` trait guard: Shared/ANEMemoryOptimizer throws it.
public enum DiarizerError: Error, LocalizedError {
    case notInitialized
    case modelDownloadFailed
    case modelCompilationFailed
    case embeddingExtractionFailed
    case invalidAudioData
    case processingFailed(String)
    case memoryAllocationFailed
    case invalidArrayBounds

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "Diarization system not initialized. Call initialize() first."
        case .modelDownloadFailed:
            return "Failed to download required models."
        case .modelCompilationFailed:
            return "Failed to compile CoreML models."
        case .embeddingExtractionFailed:
            return "Failed to extract speaker embedding from audio."
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .memoryAllocationFailed:
            return "Failed to allocate ANE-aligned memory."
        case .invalidArrayBounds:
            return "Array bounds exceeded for zero-copy view."
        }
    }
}
