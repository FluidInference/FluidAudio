import CoreML
import Foundation
import os

/// Thread-safe cache for MLMultiArray instances to reduce allocation overhead
actor MLArrayCache {
    private var cache: [CacheKey: [MLMultiArray]] = [:]
    private let maxCacheSize: Int
    private let logger = AppLogger(category: "MLArrayCache")

    struct CacheKey: Hashable {
        let shape: [Int]
        let dataType: MLMultiArrayDataType
    }

    init(maxCacheSize: Int = 100) {
        self.maxCacheSize = maxCacheSize
    }

    /// Get a cached array or create a new one
    func getArray(shape: [NSNumber], dataType: MLMultiArrayDataType) throws -> MLMultiArray {
        let key = CacheKey(
            shape: shape.map { $0.intValue },
            dataType: dataType
        )

        // Check if we have a cached array
        if var arrays = cache[key], !arrays.isEmpty {
            // Never return the same buffer twice while it is still in use; keep the trimmed bucket so we only
            // hand out arrays that callers have explicitly returned to the cache.
            let array = arrays.removeLast()
            cache[key] = arrays
            return array
        }

        return try ANEOptimizer.createANEAlignedArray(shape: shape, dataType: dataType)
    }

    /// Return an array to the cache for reuse
    func returnArray(_ array: MLMultiArray) {
        let key = CacheKey(
            shape: array.shape.map { $0.intValue },
            dataType: array.dataType
        )

        var arrays = cache[key] ?? []

        // Limit cache size per key
        if arrays.count < maxCacheSize / max(cache.count, 1) {
            // Reset the array data before caching
            if array.dataType == .float32 {
                array.resetData(to: 0)
            }
            arrays.append(array)
            cache[key] = arrays
            logger.debug("Returned array to cache for shape: \(array.shape)")
        }
    }

    /// Pre-warm the cache with commonly used shapes
    func prewarm(shapes: [(shape: [NSNumber], dataType: MLMultiArrayDataType)]) async {
        logger.info("Pre-warming cache with \(shapes.count) shapes")

        for (shape, dataType) in shapes {
            do {
                var arrays: [MLMultiArray] = []
                let prewarmCount = min(5, maxCacheSize / max(shapes.count, 1))

                for _ in 0..<prewarmCount {
                    let array = try ANEOptimizer.createANEAlignedArray(shape: shape, dataType: dataType)
                    arrays.append(array)
                }

                let key = CacheKey(shape: shape.map { $0.intValue }, dataType: dataType)
                cache[key] = arrays
            } catch {
                logger.error("Failed to pre-warm shape \(shape): \(error)")
            }
        }
    }

    /// Get a Float16 array (converting from Float32 if needed)
    func getFloat16Array(shape: [NSNumber], from float32Array: MLMultiArray? = nil) throws -> MLMultiArray {
        if let float32Array = float32Array {
            // Convert existing array to Float16
            return try ANEOptimizer.convertToFloat16(float32Array)
        } else {
            // Get new Float16 array from cache
            return try getArray(shape: shape, dataType: .float16)
        }
    }

    /// Clear the cache
    func clear() {
        cache.removeAll()
        logger.info("Cache cleared")
    }
}

/// Global shared cache instance
let sharedMLArrayCache = MLArrayCache()
