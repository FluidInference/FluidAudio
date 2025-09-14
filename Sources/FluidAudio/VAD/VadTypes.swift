import CoreML
import Foundation

public struct VadConfig: Sendable {
    public var threshold: Float
    public var debugMode: Bool
    public var computeUnits: MLComputeUnits

    public static let `default` = VadConfig()

    public init(
        threshold: Float = 0.85,
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

public struct VadSegment: Sendable {
    public let startTime: TimeInterval
    public let endTime: TimeInterval
    public let startSample: Int
    public let endSample: Int

    public var duration: TimeInterval {
        return endTime - startTime
    }

    public var sampleCount: Int {
        return endSample - startSample
    }

    public init(
        startTime: TimeInterval,
        endTime: TimeInterval,
        startSample: Int,
        endSample: Int
    ) {
        self.startTime = startTime
        self.endTime = endTime
        self.startSample = startSample
        self.endSample = endSample
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
