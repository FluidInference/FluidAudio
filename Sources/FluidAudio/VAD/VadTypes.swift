import CoreML
import Foundation

public struct VadConfig: Sendable {
    /// Baseline model probability threshold used when no segmentation override is provided.
    public var defaultThreshold: Float
    public var debugMode: Bool
    public var computeUnits: MLComputeUnits

    public static let `default` = VadConfig()

    public init(
        defaultThreshold: Float = 0.85,
        debugMode: Bool = false,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) {
        self.defaultThreshold = defaultThreshold
        self.debugMode = debugMode
        self.computeUnits = computeUnits
    }
}

// See Documentation/VAD/Segmentation.md for details
public struct VadSegmentationConfig: Sendable {
    public var minSpeechDuration: TimeInterval
    public var minSilenceDuration: TimeInterval
    public var maxSpeechDuration: TimeInterval
    public var speechPadding: TimeInterval
    public var silenceThresholdForSplit: Float
    public var negativeThreshold: Float?
    public var negativeThresholdOffset: Float
    public var minSilenceAtMaxSpeech: TimeInterval
    public var useMaxPossibleSilenceAtMaxSpeech: Bool

    public static let `default` = VadSegmentationConfig()

    public init(
        minSpeechDuration: TimeInterval = 0.15,
        minSilenceDuration: TimeInterval = 0.75,
        // ASR model by default is 15s, for other models you may want to adjust this
        maxSpeechDuration: TimeInterval = 14.0,
        speechPadding: TimeInterval = 0.1,
        silenceThresholdForSplit: Float = 0.3,
        negativeThreshold: Float? = nil,
        negativeThresholdOffset: Float = 0.15,
        minSilenceAtMaxSpeech: TimeInterval = 0.098,
        useMaxPossibleSilenceAtMaxSpeech: Bool = true
    ) {
        // Hard runtime guarantees (will trap in debug & release if violated)
        precondition(minSpeechDuration >= 0, "minSpeechDuration must be non-negative")
        precondition(minSilenceDuration >= 0, "minSilenceDuration must be non-negative")
        precondition(maxSpeechDuration > 0, "maxSpeechDuration must be positive")
        precondition(speechPadding >= 0, "speechPadding must be non-negative")
        precondition(
            silenceThresholdForSplit >= 0 && silenceThresholdForSplit <= 1,
            "silenceThresholdForSplit must be in [0, 1]")
        precondition(negativeThresholdOffset >= 0, "negativeThresholdOffset must be non-negative")
        precondition(minSilenceAtMaxSpeech >= 0, "minSilenceAtMaxSpeech must be non-negative")

        // Debug-only assertions for logical consistency
        assert(minSpeechDuration <= maxSpeechDuration, "minSpeechDuration should not exceed maxSpeechDuration")
        assert(minSilenceDuration <= maxSpeechDuration, "minSilenceDuration should not exceed maxSpeechDuration")
        assert(speechPadding <= minSpeechDuration, "speechPadding is typically <= minSpeechDuration")

        if let negative = negativeThreshold {
            precondition(negative >= 0 && negative <= 1, "negativeThreshold must be in [0, 1]")
            assert(
                negative <= silenceThresholdForSplit,
                "negativeThreshold is typically <= silenceThresholdForSplit to preserve hysteresis behavior")
        }

        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.maxSpeechDuration = maxSpeechDuration
        self.speechPadding = speechPadding
        self.silenceThresholdForSplit = silenceThresholdForSplit
        self.negativeThreshold = negativeThreshold
        self.negativeThresholdOffset = negativeThresholdOffset
        self.minSilenceAtMaxSpeech = minSilenceAtMaxSpeech
        self.useMaxPossibleSilenceAtMaxSpeech = useMaxPossibleSilenceAtMaxSpeech
    }

    /// Resolve the working negative threshold used for hysteresis when we only know the positive threshold.
    /// Mirrors Silero's heuristic of subtracting a fixed offset, but keeps an override escape hatch.
    public func effectiveNegativeThreshold(baseThreshold: Float) -> Float {
        if let override = negativeThreshold {
            return override
        }
        return max(baseThreshold - negativeThresholdOffset, 0.01)
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

public struct VadStreamState: Sendable {
    public var modelState: VadState
    public var triggered: Bool
    public var tempEndSample: Int?
    public var processedSamples: Int

    public init(
        modelState: VadState = .initial(),
        triggered: Bool = false,
        tempEndSample: Int? = nil,
        processedSamples: Int = 0
    ) {
        self.modelState = modelState
        self.triggered = triggered
        self.tempEndSample = tempEndSample
        self.processedSamples = processedSamples
    }

    public static func initial() -> VadStreamState {
        VadStreamState()
    }
}

public struct VadStreamEvent: Sendable {
    public enum Kind: Sendable {
        case speechStart
        case speechEnd
    }

    public let kind: Kind
    public let sampleIndex: Int
    public let time: TimeInterval?

    public init(kind: Kind, sampleIndex: Int, time: TimeInterval? = nil) {
        self.kind = kind
        self.sampleIndex = sampleIndex
        self.time = time
    }

    public var isStart: Bool { kind == .speechStart }
    public var isEnd: Bool { kind == .speechEnd }
}

public struct VadStreamResult: Sendable {
    public let state: VadStreamState
    public let event: VadStreamEvent?
    public let probability: Float

    public init(state: VadStreamState, event: VadStreamEvent?, probability: Float) {
        self.state = state
        self.event = event
        self.probability = probability
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
