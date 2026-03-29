import CoreML
import Foundation

/// Thread-safe cache for MLMultiArray instances to reduce allocation overhead
actor MLArrayCache {
    private var cache: [CacheKey: [MLMultiArray]] = [:]
    private let maxCacheSize: Int

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

        if var arrays = cache[key], !arrays.isEmpty {
            let array = arrays.removeLast()
            cache[key] = arrays
            return array
        }

        return try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: dataType)
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
            array.resetData(to: 0)
            arrays.append(array)
            cache[key] = arrays
        }
    }

    /// Pre-warm the cache with commonly used shapes
    func prewarm(shapes: [(shape: [NSNumber], dataType: MLMultiArrayDataType)]) {
        for (shape, dataType) in shapes {
            do {
                var arrays: [MLMultiArray] = []
                let prewarmCount = min(5, maxCacheSize / max(shapes.count, 1))

                for _ in 0..<prewarmCount {
                    let array = try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: dataType)
                    arrays.append(array)
                }

                let key = CacheKey(shape: shape.map { $0.intValue }, dataType: dataType)
                cache[key] = arrays
            } catch {
                // Silently skip shapes that fail to allocate during pre-warm
            }
        }
    }

    /// Clear the cache
    func clear() {
        cache.removeAll()
    }
}

/// Global shared cache instance
let sharedMLArrayCache = MLArrayCache()
