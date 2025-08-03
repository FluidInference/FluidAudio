import CoreML
import Foundation

public struct VadConfig: Sendable {
    public var threshold: Float
    public var chunkSize: Int
    public var sampleRate: Int
    public var modelCacheDirectory: URL?
    public var debugMode: Bool
    public var adaptiveThreshold: Bool
    public var minThreshold: Float
    public var maxThreshold: Float
    public var computeUnits: MLComputeUnits

    public var enableSNRFiltering: Bool
    public var minSNRThreshold: Float
    public var noiseFloorWindow: Int
    public var spectralRolloffThreshold: Float
    public var spectralCentroidRange: (min: Float, max: Float)

    public static let `default` = VadConfig()

    public init(
        threshold: Float = 0.445,  // Optimal: 98% accuracy on MUSAN dataset
        chunkSize: Int = 512,
        sampleRate: Int = 16000,
        modelCacheDirectory: URL? = nil,
        debugMode: Bool = false,
        adaptiveThreshold: Bool = true,  // Helps with varying audio conditions
        minThreshold: Float = 0.1,
        maxThreshold: Float = 0.7,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
        enableSNRFiltering: Bool = true,
        minSNRThreshold: Float = 6.0,
        noiseFloorWindow: Int = 100,
        spectralRolloffThreshold: Float = 0.85,
        spectralCentroidRange: (min: Float, max: Float) = (200.0, 8000.0)
    ) {
        self.threshold = threshold
        self.chunkSize = chunkSize
        self.sampleRate = sampleRate
        self.modelCacheDirectory = modelCacheDirectory
        self.debugMode = debugMode
        self.adaptiveThreshold = adaptiveThreshold
        self.minThreshold = minThreshold
        self.maxThreshold = maxThreshold
        self.computeUnits = computeUnits
        self.enableSNRFiltering = enableSNRFiltering
        self.minSNRThreshold = minSNRThreshold
        self.noiseFloorWindow = noiseFloorWindow
        self.spectralRolloffThreshold = spectralRolloffThreshold
        self.spectralCentroidRange = spectralCentroidRange
    }
}

public struct VadResult: Sendable {
    public let probability: Float
    public let isVoiceActive: Bool
    public let processingTime: TimeInterval
    public let snrValue: Float?
    public let spectralFeatures: SpectralFeatures?

    public init(
        probability: Float, isVoiceActive: Bool, processingTime: TimeInterval, snrValue: Float? = nil,
        spectralFeatures: SpectralFeatures? = nil
    ) {
        self.probability = probability
        self.isVoiceActive = isVoiceActive
        self.processingTime = processingTime
        self.snrValue = snrValue
        self.spectralFeatures = spectralFeatures
    }
}

public struct SpectralFeatures: Sendable {
    public let spectralCentroid: Float
    public let spectralRolloff: Float
    public let spectralFlux: Float
    public let mfccFeatures: [Float]
    public let zeroCrossingRate: Float
    public let spectralEntropy: Float

    public init(
        spectralCentroid: Float, spectralRolloff: Float, spectralFlux: Float, mfccFeatures: [Float],
        zeroCrossingRate: Float, spectralEntropy: Float
    ) {
        self.spectralCentroid = spectralCentroid
        self.spectralRolloff = spectralRolloff
        self.spectralFlux = spectralFlux
        self.mfccFeatures = mfccFeatures
        self.zeroCrossingRate = zeroCrossingRate
        self.spectralEntropy = spectralEntropy
    }
}

public enum VadError: Error, LocalizedError {
    case notInitialized
    case modelLoadingFailed
    case modelProcessingFailed(String)
    case invalidAudioData
    case invalidModelPath
    case modelDownloadFailed
    case modelCompilationFailed

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "VAD system not initialized. Call initialize() first."
        case .modelLoadingFailed:
            return "Failed to load VAD models."
        case .modelProcessingFailed(let message):
            return "Model processing failed: \(message)"
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .invalidModelPath:
            return "Invalid model path provided."
        case .modelDownloadFailed:
            return "Failed to download VAD models from Hugging Face."
        case .modelCompilationFailed:
            return "Failed to compile VAD models after multiple attempts."
        }
    }
}
