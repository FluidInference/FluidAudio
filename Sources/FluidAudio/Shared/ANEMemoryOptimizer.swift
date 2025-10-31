import Accelerate
import CoreML
import Dispatch
import Foundation
import Metal

private final class ANEBufferPool {

    private struct Entry {
        var array: MLMultiArray
        var leased: Bool
    }

    private var storage: [String: [Entry]] = [:]
    private let queue = DispatchQueue(label: "com.fluidaudio.ane-buffer-pool")

    func sharedBuffer(
        key: String,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType,
        allocator: () throws -> MLMultiArray
    ) throws -> MLMultiArray {
        try queue.sync {
            if let entries = storage[key],
                let index = entries.firstIndex(where: {
                    $0.array.shape == shape && $0.array.dataType == dataType
                })
            {
                return entries[index].array
            }

            let array = try allocator()
            var entries = storage[key] ?? []
            entries.append(Entry(array: array, leased: false))
            storage[key] = entries
            return array
        }
    }

    func leaseBuffer(
        key: String,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType,
        allocator: () throws -> MLMultiArray
    ) throws -> (index: Int, array: MLMultiArray) {
        try queue.sync {
            if var entries = storage[key],
                let index = entries.firstIndex(where: {
                    !$0.leased && $0.array.shape == shape && $0.array.dataType == dataType
                })
            {
                entries[index].leased = true
                storage[key] = entries
                return (index, entries[index].array)
            }

            let array = try allocator()
            var entries = storage[key] ?? []
            entries.append(Entry(array: array, leased: true))
            storage[key] = entries
            return (entries.count - 1, array)
        }
    }

    func releaseLease(key: String, index: Int) {
        queue.sync {
            guard var entries = storage[key], entries.indices.contains(index) else { return }
            entries[index].leased = false
            storage[key] = entries
        }
    }

    func clear() {
        queue.sync {
            storage.removeAll()
        }
    }
}

/// ANE-optimized memory management for speaker diarization pipeline
public final class ANEMemoryOptimizer {
    // Use shared ANE constants
    public static let aneAlignment = ANEMemoryUtils.aneAlignment
    public static let aneTileSize = ANEMemoryUtils.aneTileSize

    private let pool = ANEBufferPool()

    public init() {}

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public nonisolated func createAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        do {
            return try ANEMemoryUtils.createAlignedArray(
                shape: shape,
                dataType: dataType,
                zeroClear: true
            )
        } catch ANEMemoryUtils.ANEMemoryError.allocationFailed {
            throw DiarizerError.memoryAllocationFailed
        } catch {
            throw DiarizerError.memoryAllocationFailed
        }
    }

    /// Calculate optimal strides for ANE tile processing
    private nonisolated func calculateOptimalStrides(
        for shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) -> [NSNumber] {
        return ANEMemoryUtils.calculateOptimalStrides(for: shape)
    }

    /// Get or create a reusable buffer from pool
    public func getPooledBuffer(
        key: String,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        try pool.sharedBuffer(
            key: key,
            shape: shape,
            dataType: dataType,
            allocator: { try self.createAlignedArray(shape: shape, dataType: dataType) }
        )
    }

    /// Clear buffer pool to free memory
    public func clearBufferPool() {
        pool.clear()
    }

    /// Create zero-copy memory view between models
    public nonisolated func createZeroCopyView(
        from sourceArray: MLMultiArray,
        shape: [NSNumber],
        offset: Int = 0
    ) throws -> MLMultiArray {
        // Ensure we have enough data
        let sourceElements = sourceArray.shape.map { $0.intValue }.reduce(1, *)
        let viewElements = shape.map { $0.intValue }.reduce(1, *)

        guard offset + viewElements <= sourceElements else {
            throw DiarizerError.invalidArrayBounds
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

    /// Copy data using ANE-optimized memory operations
    public nonisolated func optimizedCopy<C>(
        from source: C,
        to destination: MLMultiArray,
        offset: Int = 0
    ) where C: Collection, C.Element == Float {
        guard destination.dataType == .float32 else { return }

        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        let count = min(source.count, destination.count - offset)

        // If source is contiguous array, use optimized copy
        if let array = source as? [Float] {
            array.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!,
                    destPtr.advanced(by: offset),
                    vDSP_Length(count),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(count)
                )
            }
        } else if let slice = source as? ArraySlice<Float> {
            // ArraySlice may share contiguous memory
            slice.withUnsafeBufferPointer { srcBuffer in
                vDSP_mmov(
                    srcBuffer.baseAddress!,
                    destPtr.advanced(by: offset),
                    vDSP_Length(count),
                    vDSP_Length(1),
                    vDSP_Length(1),
                    vDSP_Length(count)
                )
            }
        } else {
            // Fallback for other collections - copy element by element
            var destIndex = offset
            for element in source.prefix(count) {
                destPtr[destIndex] = element
                destIndex += 1
            }
        }
    }

}

/// Extension for MLMultiArray to enable zero-copy operations
extension MLMultiArray {
    /// Prefetch data to Neural Engine (iOS 17+/macOS 14+)
    public func prefetchToNeuralEngine() {
        // Trigger ANE prefetch by accessing first and last elements
        // This causes the ANE to initiate DMA transfer
        if count > 0 {
            _ = self[0]
            _ = self[count - 1]
        }
    }
}

/// Zero-copy feature provider for chaining models
public final class ZeroCopyDiarizerFeatureProvider: NSObject, MLFeatureProvider {
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
        to inputName: String,
        additionalFeatures: [String: MLFeatureValue] = [:]
    ) -> ZeroCopyDiarizerFeatureProvider? {
        guard let outputValue = outputProvider.featureValue(for: outputName) else {
            return nil
        }

        var features = additionalFeatures
        features[inputName] = outputValue

        return ZeroCopyDiarizerFeatureProvider(features: features)
    }

    /// Create a provider for batch processing multiple inputs
    public static func batchProvider(
        waveforms: [MLMultiArray],
        masks: [MLMultiArray]
    ) -> [ZeroCopyDiarizerFeatureProvider] {
        zip(waveforms, masks).map { waveform, mask in
            ZeroCopyDiarizerFeatureProvider(features: [
                "waveform": MLFeatureValue(multiArray: waveform),
                "mask": MLFeatureValue(multiArray: mask),
            ])
        }
    }
}
