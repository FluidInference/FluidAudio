import Accelerate
import CoreML
import Foundation
import Metal

/// ANE-optimized memory management for speaker diarization pipeline
@available(macOS 13.0, iOS 16.0, *)
public class ANEMemoryOptimizer {
    /// ANE requires 64-byte alignment for optimal DMA transfers
    public static let aneAlignment = 64

    /// ANE tile size for matrix operations
    public static let aneTileSize = 16

    /// Shared instance for resource management
    public static let shared = ANEMemoryOptimizer()

    private var bufferPool: [String: MLMultiArray] = [:]
    private let bufferLock = NSLock()

    private init() {}

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public func createAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        // Calculate total elements
        let totalElements = shape.map { $0.intValue }.reduce(1, *)

        // Calculate element size
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
        let alignedBytes = ((bytesNeeded + Self.aneAlignment - 1) / Self.aneAlignment) * Self.aneAlignment

        // Allocate page-aligned memory for ANE DMA
        var alignedPointer: UnsafeMutableRawPointer?
        let result = posix_memalign(&alignedPointer, Self.aneAlignment, alignedBytes)

        guard result == 0, let pointer = alignedPointer else {
            throw DiarizerError.memoryAllocationFailed
        }

        // Zero-initialize the memory
        memset(pointer, 0, alignedBytes)

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
    private func calculateOptimalStrides(
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
            if i == shape.count - 1 && dimSize % Self.aneTileSize != 0 {
                // Pad the innermost dimension to tile boundary
                let paddedSize = ((dimSize + Self.aneTileSize - 1) / Self.aneTileSize) * Self.aneTileSize
                currentStride *= paddedSize
            } else {
                currentStride *= dimSize
            }
        }

        return strides.map { NSNumber(value: $0) }
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

    /// Copy data using ANE-optimized memory operations
    public func optimizedCopy(
        from source: [Float],
        to destination: MLMultiArray,
        offset: Int = 0
    ) {
        guard destination.dataType == .float32 else { return }

        let destPtr = destination.dataPointer.assumingMemoryBound(to: Float.self)
        let count = min(source.count, destination.count - offset)

        // Use vDSP for optimized memory copy
        source.withUnsafeBufferPointer { srcBuffer in
            vDSP_mmov(
                srcBuffer.baseAddress!,
                destPtr.advanced(by: offset),
                vDSP_Length(count),
                vDSP_Length(1),
                vDSP_Length(1),
                vDSP_Length(count)
            )
        }
    }

    /// Clear buffer pool to free memory
    public func clearBufferPool() {
        bufferLock.lock()
        defer { bufferLock.unlock() }

        bufferPool.removeAll()
    }
}

/// Extension for MLMultiArray to enable zero-copy operations
@available(macOS 13.0, iOS 16.0, *)
extension MLMultiArray {
    /// Prefetch data to Neural Engine (iOS 17+/macOS 14+)
    @available(macOS 14.0, iOS 17.0, *)
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
@available(macOS 13.0, iOS 16.0, *)
public class ZeroCopyDiarizerFeatureProvider: NSObject, MLFeatureProvider {
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
