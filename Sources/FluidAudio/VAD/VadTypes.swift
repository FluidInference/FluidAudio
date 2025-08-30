import CoreML
import Foundation

public struct VadConfig: Sendable {
    public var threshold: Float
    public var debugMode: Bool
    public var computeUnits: MLComputeUnits

    public static let `default` = VadConfig()

    public init(
        threshold: Float = 0.5,
        debugMode: Bool = false,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.threshold = threshold
        self.debugMode = debugMode
        self.computeUnits = computeUnits
    }
}

public struct VadResult: Sendable {
    public let probability: Float
    public let isVoiceActive: Bool
    public let processingTime: TimeInterval

    public init(
        probability: Float,
        isVoiceActive: Bool,
        processingTime: TimeInterval
    ) {
        self.probability = probability
        self.isVoiceActive = isVoiceActive
        self.processingTime = processingTime
    }
}

// Internal struct for VadAudioProcessor compatibility
internal struct SpectralFeatures {
    let spectralFlux: Float
    let mfccFeatures: [Float]
}

public enum VadError: Error, LocalizedError {
    case notInitialized
    case modelLoadingFailed
    case modelProcessingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "VAD system not initialized"
        case .modelLoadingFailed:
            return "Failed to load VAD model"
        case .modelProcessingFailed(let message):
            return "Model processing failed: \(message)"
        }
    }
}
