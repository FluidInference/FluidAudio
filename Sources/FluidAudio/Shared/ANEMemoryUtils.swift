import Accelerate
import CoreML
import Foundation
import Metal

/// Shared ANE optimization utilities for all ML pipelines
@available(macOS 13.0, iOS 16.0, *)
public enum ANEMemoryUtils {

    /// ANE requires 64-byte alignment for optimal DMA transfers
    public static let aneAlignment = 64

    /// ANE tile size for matrix operations
    public static let aneTileSize = 16

    /// Errors that can occur during ANE memory operations
    public enum ANEMemoryError: Error {
        case allocationFailed
        case invalidShape
        case unsupportedDataType
    }

    /// Create ANE-aligned MLMultiArray with optimized memory layout
    public static func createAlignedArray(
        shape: [NSNumber],
        dataType: MLMultiArrayDataType,
        zeroClear: Bool = true
    ) throws -> MLMultiArray {
        // Calculate element size
        let elementSize = getElementSize(for: dataType)

        // Calculate total elements from shape
        let totalElements = shape.map { $0.intValue }.reduce(1, *)

        // For very large arrays, use standard strides to avoid excessive memory overhead
        // ANE optimization is beneficial for smaller tensors but counterproductive for large test arrays
        let strides: [NSNumber]
        if totalElements > 100_000 {
            // Use standard row-major strides for large arrays
            var standardStrides: [Int] = []
            var currentStride = 1
            for i in (0..<shape.count).reversed() {
                standardStrides.insert(currentStride, at: 0)
                currentStride *= shape[i].intValue
            }
            strides = standardStrides.map { NSNumber(value: $0) }
        } else {
            // Use ANE-optimized strides for smaller arrays
            strides = calculateOptimalStrides(for: shape)
        }

        let actualElements = totalElements

        // Align the allocation size to ANE requirements
        let bytesNeeded = actualElements * elementSize
        // Ensure at least one alignment unit is allocated even for empty arrays
        let alignedBytes = max(aneAlignment, ((bytesNeeded + aneAlignment - 1) / aneAlignment) * aneAlignment)

        // Allocate page-aligned memory for ANE DMA
        var alignedPointer: UnsafeMutableRawPointer?
        let result = posix_memalign(&alignedPointer, aneAlignment, alignedBytes)

        guard result == 0, let pointer = alignedPointer else {
            throw ANEMemoryError.allocationFailed
        }

        // Zero-initialize the memory if requested
        if zeroClear {
            memset(pointer, 0, alignedBytes)
        }

        // Create MLMultiArray with aligned memory
        let array = try MLMultiArray(
            dataPointer: pointer,
            shape: shape,
            dataType: dataType,
            strides: strides,
            deallocator: { bytes in
                bytes.deallocate()
            }
        )

        return array
    }

    /// Calculate optimal strides for ANE tile processing
    public static func calculateOptimalStrides(for shape: [NSNumber]) -> [NSNumber] {
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

    /// Get element size in bytes for a given data type
    public static func getElementSize(for dataType: MLMultiArrayDataType) -> Int {
        switch dataType {
        case .float16:
            return 2
        case .float32:
            return 4
        case .float64, .double:
            return 8
        case .int32:
            return 4
        @unknown default:
            return 4
        }
    }

    /// Create a zero-copy view of an MLMultiArray slice
    public static func createZeroCopyView(
        from array: MLMultiArray,
        offset: Int,
        shape: [NSNumber],
        strides: [NSNumber]? = nil
    ) throws -> MLMultiArray {
        // Validate bounds
        let elementSize = getElementSize(for: array.dataType)
        let totalElements = shape.map { $0.intValue }.reduce(1, *)
        let bytesNeeded = totalElements * elementSize
        let byteOffset = offset * elementSize

        guard byteOffset + bytesNeeded <= array.count * elementSize else {
            throw ANEMemoryError.invalidShape
        }

        // Create view with offset pointer
        let offsetPointer = array.dataPointer.advanced(by: byteOffset)

        return try MLMultiArray(
            dataPointer: offsetPointer,
            shape: shape,
            dataType: array.dataType,
            strides: strides ?? calculateOptimalStrides(for: shape),
            deallocator: nil  // No deallocation for views
        )
    }

    /// Prefetch memory pages for ANE processing
    public static func prefetchForANE(_ array: MLMultiArray) {
        let dataPointer = array.dataPointer
        let elementSize = getElementSize(for: array.dataType)
        let totalBytes = array.count * elementSize

        // Touch first and last cache lines to trigger ANE DMA prefetch
        if totalBytes > 0 {
            _ = dataPointer.load(as: UInt8.self)
            if totalBytes > 1 {
                _ = dataPointer.advanced(by: totalBytes - 1).load(as: UInt8.self)
            }
        }
    }
}

/// Extension for MLMultiArray to add ANE optimization methods
@available(macOS 13.0, iOS 16.0, *)
extension MLMultiArray {

    /// Check if this array is ANE-aligned
    public var isANEAligned: Bool {
        let address = Int(bitPattern: self.dataPointer)
        return address % ANEMemoryUtils.aneAlignment == 0
    }

    /// Prefetch this array for ANE processing
    public func prefetchForANE() {
        ANEMemoryUtils.prefetchForANE(self)
    }
}
