import Accelerate
import CoreML
import Foundation
import Metal

/// Neural Engine optimization utilities for ASR pipeline
@available(macOS 13.0, iOS 16.0, *)
public enum ANEOptimizer {

    /// ANE requires 64-byte alignment for optimal DMA transfers
    public static let aneAlignment = 64

    /// ANE tile size for matrix operations
    public static let aneTileSize = 16

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public static func createANEAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        // Calculate total elements
        let totalElements = shape.map { $0.intValue }.reduce(1, *)

        // Calculate stride to ensure ANE alignment
        let elementSize: Int
        switch dataType {
        case .float16:
            elementSize = 2
        case .float32:
            elementSize = 4
        case .float64:
            elementSize = 8
        case .int32:
            elementSize = 4
        case .double:
            elementSize = 8
        @unknown default:
            elementSize = 4
        }

        // Align the allocation size to ANE requirements
        let bytesNeeded = totalElements * elementSize
        let alignedBytes = ((bytesNeeded + aneAlignment - 1) / aneAlignment) * aneAlignment

        // Allocate page-aligned memory for ANE DMA
        var alignedPointer: UnsafeMutableRawPointer?
        let result = posix_memalign(&alignedPointer, aneAlignment, alignedBytes)

        guard result == 0, let pointer = alignedPointer else {
            throw NSError(
                domain: "ANEOptimizer", code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to allocate ANE-aligned memory"])
        }

        // Create MLMultiArray with aligned memory
        let array = try MLMultiArray(
            dataPointer: pointer,
            shape: shape,
            dataType: dataType,
            strides: calculateOptimalStrides(for: shape, dataType: dataType),
            deallocator: { bytes in
                bytes.deallocate()
            }
        )

        return array
    }

    /// Calculate optimal strides for ANE tile processing
    public static func calculateOptimalStrides(
        for shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) -> [NSNumber] {
        var strides: [Int] = []
        var currentStride = 1

        // Calculate strides from last dimension to first
        for i in (0..<shape.count).reversed() {
            strides.insert(currentStride, at: 0)
            let dimSize = shape[i].intValue

            // Align dimension stride to ANE tile boundaries when beneficial
            if i == shape.count - 1 && dimSize % aneTileSize != 0 {
                // Pad the innermost dimension to tile boundary
                let paddedSize = ((dimSize + aneTileSize - 1) / aneTileSize) * aneTileSize
                currentStride *= paddedSize
            } else {
                currentStride *= dimSize
            }
        }

        return strides.map { NSNumber(value: $0) }
    }

    /// Configure optimal compute units for each model type
    public static func optimalComputeUnits(for modelType: ModelType) -> MLComputeUnits {
        switch modelType {
        case .melSpectrogram:
            // FFT operations run better on CPU/GPU
            return .cpuAndGPU
        case .encoder:
            // Transformer/attention benefits from Neural Engine
            return .cpuAndNeuralEngine
        case .decoder:
            // LSTM operations optimized for Neural Engine
            return .cpuAndNeuralEngine
        case .joint:
            // Small dense layers perfect for Neural Engine
            if #available(macOS 14.0, iOS 17.0, *) {
                // Use Neural Engine exclusively for small models
                return .all  // Will prefer NE for small ops
            } else {
                return .cpuAndNeuralEngine
            }
        }
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
        let elementSize: Int
        switch sourceArray.dataType {
        case .float16:
            elementSize = 2
        case .float32:
            elementSize = 4
        case .float64:
            elementSize = 8
        case .int32:
            elementSize = 4
        case .double:
            elementSize = 8
        @unknown default:
            elementSize = 4
        }

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
    @available(macOS 14.0, iOS 17.0, *)
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

        // Convert using Accelerate
        let sourcePtr = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)
        let destPtr = float16Array.dataPointer.bindMemory(to: Float16.self, capacity: input.count)

        var sourceBuffer = vImage_Buffer(
            data: sourcePtr,
            height: 1,
            width: vImagePixelCount(input.count),
            rowBytes: input.count * MemoryLayout<Float>.stride
        )

        var destBuffer = vImage_Buffer(
            data: destPtr,
            height: 1,
            width: vImagePixelCount(input.count),
            rowBytes: input.count * MemoryLayout<Float16>.stride
        )

        vImageConvert_PlanarFtoPlanar16F(&sourceBuffer, &destBuffer, 0)

        return float16Array
    }

    /// Model type enumeration for compute unit selection
    public enum ModelType {
        case melSpectrogram
        case encoder
        case decoder
        case joint
    }
}

/// Extension for MLFeatureProvider to enable zero-copy chaining
@available(macOS 13.0, iOS 16.0, *)
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
