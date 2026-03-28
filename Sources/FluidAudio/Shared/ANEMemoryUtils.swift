import Accelerate
import CoreML
import Darwin
import Foundation

/// Shared ANE optimization utilities for all ML pipelines
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

        // Calculate optimal strides for ANE
        let strides = calculateOptimalStrides(for: shape)

        // Calculate actual elements needed based on strides (accounts for padding)
        // The total elements needed is the stride of the first dimension times the first dimension size
        let totalElementsNeeded: Int
        if !shape.isEmpty {
            totalElementsNeeded = strides[0].intValue * shape[0].intValue
        } else {
            totalElementsNeeded = 0
        }

        // Align the allocation size to ANE requirements
        let bytesNeeded = totalElementsNeeded * elementSize
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
                // `posix_memalign` requires `free` for cleanup; `deallocate()` would trap.
                Darwin.free(bytes)
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
        case .float32, .float:
            return 4
        case .float64, .double:
            return 8
        case .int32:
            return MemoryLayout<Int32>.stride
        @unknown default:
            return MemoryLayout<Float>.stride
        }
    }

    /// Create a zero-copy view of an MLMultiArray slice
    public static func createZeroCopyView(
        from array: MLMultiArray,
        offset: Int,
        shape: [NSNumber],
        strides: [NSNumber]? = nil
    ) throws -> MLMultiArray {
        // Validate bounds using stride-aware backing size (accounts for ANE padding)
        let elementSize = getElementSize(for: array.dataType)
        let viewStrides = strides ?? calculateOptimalStrides(for: shape)
        let viewBackingElements =
            shape.isEmpty ? 0 : viewStrides[0].intValue * shape[0].intValue
        let sourceBackingElements =
            array.shape.isEmpty ? 0 : array.strides[0].intValue * array.shape[0].intValue
        let byteOffset = offset * elementSize

        guard byteOffset + viewBackingElements * elementSize <= sourceBackingElements * elementSize else {
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

    /// Convert a float32 MLMultiArray to float16 with ANE-aligned memory.
    public static func convertToFloat16(_ input: MLMultiArray) throws -> MLMultiArray {
        guard input.dataType == .float32 else {
            throw ANEMemoryError.unsupportedDataType
        }

        let float16Array = try createAlignedArray(
            shape: input.shape,
            dataType: .float16,
            zeroClear: false
        )

        let sourcePtr = input.dataPointer.bindMemory(to: Float.self, capacity: input.count)

        var sourceBuffer = vImage_Buffer(
            data: sourcePtr,
            height: 1,
            width: vImagePixelCount(input.count),
            rowBytes: input.count * MemoryLayout<Float>.stride
        )

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

    /// Stride-aware copy between two MLMultiArrays that may have different stride layouts.
    ///
    /// Copies all logical elements from `source` to `destination` (which must have the same shape
    /// and data type). The innermost dimension must have stride 1 in both arrays. Outer dimensions
    /// are iterated respecting each array's strides.
    public static func strideAwareCopy(from source: MLMultiArray, to destination: MLMultiArray) {
        let shape = source.shape.map { $0.intValue }
        let srcStrides = source.strides.map { $0.intValue }
        let dstStrides = destination.strides.map { $0.intValue }

        let ndim = shape.count
        guard ndim > 0 else { return }

        // Validate shapes and types match
        precondition(
            source.shape == destination.shape,
            "strideAwareCopy: shape mismatch \(source.shape) vs \(destination.shape)"
        )
        precondition(
            source.dataType == destination.dataType,
            "strideAwareCopy: dataType mismatch"
        )
        precondition(
            srcStrides[ndim - 1] == 1 && dstStrides[ndim - 1] == 1,
            "strideAwareCopy: innermost stride must be 1"
        )

        let elementSize = getElementSize(for: source.dataType)
        let srcPtr = source.dataPointer
        let dstPtr = destination.dataPointer

        // If strides match, a single memcpy suffices (fast path).
        if srcStrides == dstStrides {
            let totalBytes = srcStrides[0] * shape[0] * elementSize
            memcpy(dstPtr, srcPtr, totalBytes)
            return
        }

        // Innermost dimension byte count
        let innerBytes = shape[ndim - 1] * elementSize

        if ndim == 1 {
            memcpy(dstPtr, srcPtr, innerBytes)
            return
        }

        // Number of outer "rows" = product of all dimensions except the last
        let outerCount = shape.dropLast().reduce(1, *)

        // Multi-index iteration over outer dimensions
        var indices = [Int](repeating: 0, count: ndim - 1)

        for _ in 0..<outerCount {
            // Compute flat byte offset for source and destination
            var srcByteOffset = 0
            var dstByteOffset = 0
            for d in 0..<(ndim - 1) {
                srcByteOffset += indices[d] * srcStrides[d] * elementSize
                dstByteOffset += indices[d] * dstStrides[d] * elementSize
            }

            // Copy innermost dimension as contiguous block
            memcpy(dstPtr + dstByteOffset, srcPtr + srcByteOffset, innerBytes)

            // Increment multi-index (odometer style)
            var carry = ndim - 2
            while carry >= 0 {
                indices[carry] += 1
                if indices[carry] < shape[carry] {
                    break
                }
                indices[carry] = 0
                carry -= 1
            }
        }
    }

}
