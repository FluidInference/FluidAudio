import CoreML
import Foundation
import os

/// Thread-safe cache for MLMultiArray instances to reduce allocation overhead
@available(macOS 13.0, iOS 16.0, *)
final class MLArrayCache: Sendable {
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
        // For Swift 6 compatibility, bypass caching temporarily
        // TODO: Implement proper Sendable caching mechanism
        logger.debug("Creating ANE-aligned array for shape: \(shape)")
        return try ANEOptimizer.createANEAlignedArray(shape: shape, dataType: dataType)
    }

    /// Return an array to the cache for reuse
    func returnArray(_ array: MLMultiArray) {
        // For Swift 6 compatibility, bypass caching temporarily
        // TODO: Implement proper Sendable caching mechanism
    }

    /// Pre-warm the cache with commonly used shapes
    func prewarm(shapes: [(shape: [NSNumber], dataType: MLMultiArrayDataType)]) async {
        // For Swift 6 compatibility, bypass prewarming temporarily
        logger.info("Cache prewarming skipped for Swift 6 compatibility")
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
        // For Swift 6 compatibility, no cache to clear
        logger.info("Cache clear skipped (no active cache)")
    }
}

/// Global shared cache instance
@available(macOS 13.0, iOS 16.0, *)
let sharedMLArrayCache = MLArrayCache()
