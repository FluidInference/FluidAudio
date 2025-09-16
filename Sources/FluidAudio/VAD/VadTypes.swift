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

public struct VadSegmentationConfig: Sendable {
    public var minSpeechDuration: TimeInterval
    public var minSilenceDuration: TimeInterval
    public var maxSpeechDuration: TimeInterval
    public var speechPadding: TimeInterval
    public var silenceThresholdForSplit: Float

    public static let `default` = VadSegmentationConfig()

    public init(
        minSpeechDuration: TimeInterval = 0.15,
        minSilenceDuration: TimeInterval = 0.75,
        // ASR model by default is 15s, for other models you may want to adjust this
        maxSpeechDuration: TimeInterval = 14.0,
        speechPadding: TimeInterval = 0.1,
        silenceThresholdForSplit: Float = 0.3
    ) {
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.maxSpeechDuration = maxSpeechDuration
        self.speechPadding = speechPadding
        self.silenceThresholdForSplit = silenceThresholdForSplit
    }
}

public struct VadState: Sendable {
    public static let contextLength = 64

    public let hiddenState: [Float]
    public let cellState: [Float]
    public let context: [Float]

    public init(
        hiddenState: [Float], cellState: [Float], context: [Float] = Array(repeating: 0.0, count: contextLength)
    ) {
        self.hiddenState = hiddenState
        self.cellState = cellState
        self.context = context
    }

    /// Create initial zero states for the first chunk
    public static func initial() -> VadState {
        return VadState(
            hiddenState: Array(repeating: 0.0, count: 128),
            cellState: Array(repeating: 0.0, count: 128),
            context: Array(repeating: 0.0, count: contextLength)
        )
    }
}

public struct VadResult: Sendable {
    public let probability: Float
    public let isVoiceActive: Bool
    public let processingTime: TimeInterval
    public let outputState: VadState

    public init(
        probability: Float,
        isVoiceActive: Bool,
        processingTime: TimeInterval,
        outputState: VadState
    ) {
        self.probability = probability
        self.isVoiceActive = isVoiceActive
        self.processingTime = processingTime
        self.outputState = outputState
    }
}

public struct VadSegment: Sendable {
    public let startTime: TimeInterval
    public let endTime: TimeInterval

    public var duration: TimeInterval {
        return endTime - startTime
    }

    public func startSample(sampleRate: Int) -> Int {
        return Int(startTime * Double(sampleRate))
    }

    public func endSample(sampleRate: Int) -> Int {
        return Int(endTime * Double(sampleRate))
    }

    public func sampleCount(sampleRate: Int) -> Int {
        return endSample(sampleRate: sampleRate) - startSample(sampleRate: sampleRate)
    }

    public init(
        startTime: TimeInterval,
        endTime: TimeInterval
    ) {
        self.startTime = startTime
        self.endTime = endTime
    }
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
