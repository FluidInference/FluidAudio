import Accelerate
import CoreML
import Foundation
import Metal

/// ANE-optimized memory management for speaker diarization pipeline
public final class ANEMemoryOptimizer {
    // Use shared ANE constants
    public static let aneAlignment = ANEMemoryUtils.aneAlignment
    public static let aneTileSize = ANEMemoryUtils.aneTileSize

    private var bufferPool: [String: MLMultiArray] = [:]
    private let bufferLock = NSLock()

    public init() {}

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public func createAlignedArray(
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
    private func calculateOptimalStrides(
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
        bufferLock.lock()
        defer { bufferLock.unlock() }

        if let existing = bufferPool[key] {
            // Verify shape matches
            if existing.shape == shape && existing.dataType == dataType {
                return existing
            }
        }

        // Create new buffer
        let buffer = try createAlignedArray(shape: shape, dataType: dataType)
        bufferPool[key] = buffer
        return buffer
    }

    /// Initialize a reusable buffer in the pool
    public func initializePooledBuffer(
        key: String,
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws {
        bufferLock.lock()
        defer { bufferLock.unlock() }
        // Create new buffer
        bufferPool[key] = try createAlignedArray(shape: shape, dataType: dataType)
    }

    /// Clear buffer pool to free memory
    public func clearBufferPool() {
        bufferLock.lock()
        defer { bufferLock.unlock() }

        bufferPool.removeAll()
    }

    /// Create zero-copy memory view between models
    public func createZeroCopyView(
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
    public func optimizedCopy<C>(
        from source: C,
        to destination: MLMultiArray,
        offset: Int = 0,
        pad: Bool = false
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
        } else if let contiguous = source as? ContiguousArray<Float> {
            // ContiguousArray is also contiguous
            contiguous.withUnsafeBufferPointer { srcBuffer in
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

        if pad {
            let padStart = offset + count
            let remaining = destination.count - padStart
            if remaining > 0 {
                vDSP_vclr(destPtr.advanced(by: padStart), 1, vDSP_Length(remaining))
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
        return zip(waveforms, masks).map { waveform, mask in
            ZeroCopyDiarizerFeatureProvider(features: [
                "waveform": MLFeatureValue(multiArray: waveform),
                "mask": MLFeatureValue(multiArray: mask),
            ])
        }
    }
}
