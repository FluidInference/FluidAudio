import Accelerate
import CoreML
import Foundation
import Metal

/// Neural Engine optimization utilities for ASR pipeline
public enum ANEOptimizer {

    // Use shared ANE constants
    public static let aneAlignment = ANEMemoryUtils.aneAlignment
    public static let aneTileSize = ANEMemoryUtils.aneTileSize

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public static func createANEAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        do {
            return try ANEMemoryUtils.createAlignedArray(
                shape: shape,
                dataType: dataType,
                zeroClear: false  // ASR doesn't need zero-cleared memory
            )
        } catch ANEMemoryUtils.ANEMemoryError.allocationFailed {
            throw NSError(
                domain: "ANEOptimizer", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to allocate ANE-aligned memory"])
        } catch {
            throw NSError(
                domain: "ANEOptimizer", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "ANE memory allocation error: \(error)"])
        }
    }

    /// Calculate optimal strides for ANE tile processing
    public static func calculateOptimalStrides(
        for shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) -> [NSNumber] {
        return ANEMemoryUtils.calculateOptimalStrides(for: shape)
    }

    /// Configure optimal compute units for each model type
    public static func optimalComputeUnits(for modelType: ModelType) -> MLComputeUnits {
        return .cpuAndNeuralEngine
    }

    /// Create zero-copy memory view between models
    public static func createZeroCopyView(
        from sourceArray: MLMultiArray,
        shape: [NSNumber],
        offset: Int = 0
    ) throws -> MLMultiArray {
        // Ensure we have enough data
        let sourceElements = sourceArray.shape.map { $0.intValue }.reduce(1, *)
        let viewElements = shape.map { $0.intValue }.reduce(1, *)

        guard offset + viewElements <= sourceElements else {
            throw NSError(
                domain: "ANEOptimizer", code: -2,
                userInfo: [NSLocalizedDescriptionKey: "View exceeds source array bounds"])
        }

        // Calculate byte offset
        let elementSize = ANEMemoryUtils.getElementSize(for: sourceArray.dataType)

        let byteOffset = offset * elementSize
        let offsetPointer = sourceArray.dataPointer.advanced(by: byteOffset)

        // Create view with same data but new shape
        return try MLMultiArray(
            dataPointer: offsetPointer,
            shape: shape,
            dataType: sourceArray.dataType,
            strides: calculateOptimalStrides(for: shape, dataType: sourceArray.dataType),
            deallocator: nil  // No deallocation since it's a view
        )
    }

    /// Prefetch data to Neural Engine
    public static func prefetchToNeuralEngine(_ array: MLMultiArray) {
        // Trigger ANE prefetch by accessing first and last elements
        // This causes the ANE to initiate DMA transfer
        if array.count > 0 {
            _ = array[0]
            _ = array[array.count - 1]
        }
    }

    /// Convert float32 array to float16 for ANE efficiency
    public static func convertToFloat16(_ input: MLMultiArray) throws -> MLMultiArray {
        guard input.dataType == .float32 else {
            throw NSError(
                domain: "ANEOptimizer", code: -3,
                userInfo: [NSLocalizedDescriptionKey: "Input must be float32"])
        }

        // Create float16 array with ANE alignment
        let float16Array = try createANEAlignedArray(
            shape: input.shape,
            dataType: .float16
        )

        // Convert using Accelerate with platform-specific handling
        let sourcePtr = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)

        var sourceBuffer = vImage_Buffer(
            data: sourcePtr,
            height: 1,
            width: vImagePixelCount(input.count),
            rowBytes: input.count * MemoryLayout<Float>.stride
        )

        // Use UInt16 as storage type for cross-platform compatibility
        let destPtr = float16Array.dataPointer.bindMemory(to: UInt16.self, capacity: input.count)

        var destBuffer = vImage_Buffer(
            data: destPtr,
            height: 1,
            width: vImagePixelCount(input.count),
            rowBytes: input.count * MemoryLayout<UInt16>.stride
        )

        vImageConvert_PlanarFtoPlanar16F(&sourceBuffer, &destBuffer, 0)

        return float16Array
    }

    /// Model type enumeration for compute unit selection
    public enum ModelType {
        case encoder
        case decoder
        case joint
    }
}

/// Extension for MLFeatureProvider to enable zero-copy chaining
public class ZeroCopyFeatureProvider: NSObject, MLFeatureProvider {
    private let features: [String: MLFeatureValue]

    public init(features: [String: MLFeatureValue]) {
        self.features = features
        super.init()
    }

    public var featureNames: Set<String> {
        Set(features.keys)
    }

    public func featureValue(for featureName: String) -> MLFeatureValue? {
        features[featureName]
    }

    /// Create a provider that chains output from one model to input of another
    public static func chain(
        from outputProvider: MLFeatureProvider,
        outputName: String,
        to inputName: String
    ) -> ZeroCopyFeatureProvider? {
        guard let outputValue = outputProvider.featureValue(for: outputName) else {
            return nil
        }

        return ZeroCopyFeatureProvider(features: [inputName: outputValue])
    }
}
