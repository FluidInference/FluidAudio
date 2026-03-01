import Foundation
import CoreML
import Accelerate

public struct TitaNetEmbeddingExtractor {
    public let config: EmbeddingConfig
    public let model: MLModel
    private let memoryOptimizer: ANEMemoryOptimizer
    private let processedSignalArray: MLMultiArray
    private let lengthArray: MLMultiArray
    private let sampleRate: Float = 16_000
    
    public init(config: EmbeddingConfig) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        
        self.config = config
        self.model = try TitaNet_large_2_48s(configuration: configuration).model
        self.memoryOptimizer = ANEMemoryOptimizer()
        self.processedSignalArray = try memoryOptimizer.createAlignedArray(
            shape: [
                1,
                NSNumber(value: IndexUtils.nextMultiple(of: config.melPadTo, for: config.maxMelLength)),
                NSNumber(value: config.melFeatures)
            ],
            dataType: .float32
        )
        self.lengthArray = try memoryOptimizer.createAlignedArray(
            shape: [1],
            dataType: .int32
        )
    }
    
    /// Extract a speaker embedding from an audio signal
    /// - Parameter audioSignal: A raw 16kHz audio signal. Must fit within the configured input size limits
    /// - Returns: A 192D speaker identity embedding vector
    public func extractEmbedding<C>(
        mels: C, melLength: Int
    ) throws -> [Float] where C: AccelerateBuffer & Collection, C.Element == Float, C.Index == Int {
        // Ensure input size is within bounds
        guard melLength >= config.minMelLength else {
            throw TitaNetError.invalidAudioInput(
                "Audio is too short to extract speaker identity (\(melLength) < \(config.minMelLength))")
        }
        
        guard melLength <= config.maxMelLength else {
            throw TitaNetError.invalidAudioInput(
                "Audio too long for TitaNet model (\(melLength) > \(config.maxMelLength))")
        }
        
        // Copy inputs to MLMultiArrays
        memoryOptimizer.optimizedCopy(
            from: mels,
            to: processedSignalArray,
            pad: true
        )
        
        lengthArray[0] = NSNumber(value: Int32(melLength))
        
        // Build input
        let input: MLDictionaryFeatureProvider
        do {
            input = try MLDictionaryFeatureProvider(dictionary: [
                "processed_signal": MLFeatureValue(multiArray: processedSignalArray),
                "length": MLFeatureValue(multiArray: lengthArray)
            ])
        } catch {
            throw TitaNetError.predictionFailed("Failed to create input features: \(error)")
        }
        
        // Get prediction
        let output: MLFeatureProvider
        do {
//            let start = Date()
            output = try model.prediction(from: input)
//            let end = Date()
//            print("embeddings extracted in \(end.timeIntervalSince(start))s")
        } catch {
            throw TitaNetError.predictionFailed("CoreML prediction failed (melLength=\(melLength), shape=\(processedSignalArray.shape)): \(error)")
        }
        
        guard let embedding = output.featureValue(for: "embedding")?.shapedArrayValue(of: Float.self)?.scalars else {
            throw TitaNetError.predictionFailed("Missing embedding output")
        }
        
        return embedding
    }
}

public enum TitaNetError: Error, LocalizedError {
    case invalidAudioInput(String)
    case predictionFailed(String)
}

public enum TitaNetVariant {
    case large2_48s
    case large3_04s
    case small2_48s
    case small3_04s
    
    public func model(configuration: MLModelConfiguration) throws -> MLModel {
        switch self {
        case .large2_48s: return try TitaNet_large_2_48s(configuration: configuration).model
        case .small2_48s: return try TitaNet_small_2_48s(configuration: configuration).model
        case .large3_04s: return try TitaNet_large_3_04s(configuration: configuration).model
        case .small3_04s: return try TitaNet_small_3_04s(configuration: configuration).model
        }
    }
    
    public var name: String {
        switch self {
        case .large2_48s: return "TitaNet_large_2_48s"
        case .small2_48s: return "TitaNet_small_2_48s"
        case .large3_04s: return "TitaNet_large_3_04s"
        case .small3_04s: return "TitaNet_small_3_04s"
        }
    }
    
    public var maxInputFrames: Int {
        switch self {
        case .large2_48s: return 31
        case .large3_04s: return 38
        case .small2_48s: return 31
        case .small3_04s: return 38
        }
    }
}
