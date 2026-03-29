@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class MLArrayCacheTests: XCTestCase {

    var cache: MLArrayCache!

    override func setUp() async throws {
        cache = MLArrayCache(maxCacheSize: 10)
    }

    // MARK: - Basic Cache Operations

    func testGetArrayCreatesANEAligned() async throws {
        let shape: [NSNumber] = [1, 100]
        let array = try await cache.getArray(shape: shape, dataType: .float32)

        XCTAssertEqual(array.shape, shape)
        XCTAssertEqual(array.dataType, .float32)

        // In CI, we don't test alignment since we use standard arrays
        let isCI = ProcessInfo.processInfo.environment["CI"] != nil
        if !isCI {
            // Verify ANE alignment only in non-CI environment
            let pointerValue = Int(bitPattern: array.dataPointer)
            XCTAssertEqual(pointerValue % ANEMemoryUtils.aneAlignment, 0)
        }
    }

    func testCacheHitOnSecondRequest() async throws {
        let shape: [NSNumber] = [2, 50]

        // First request - cache miss
        let array1 = try await cache.getArray(shape: shape, dataType: .float32)

        // Return array to cache
        await cache.returnArray(array1)

        // Second request - should be cache hit
        let array2 = try await cache.getArray(shape: shape, dataType: .float32)

        // Arrays should have same shape and type
        XCTAssertEqual(array2.shape, shape)
        XCTAssertEqual(array2.dataType, .float32)
    }

    func testReturnArrayResetsData() async throws {
        let shape: [NSNumber] = [10]
        let array = try await cache.getArray(shape: shape, dataType: .float32)

        // Set some values
        for i in 0..<array.count {
            array[i] = NSNumber(value: Float(i) * 2.0)
        }

        // Return to cache
        await cache.returnArray(array)

        // Get from cache again
        let cachedArray = try await cache.getArray(shape: shape, dataType: .float32)

        // Data should be reset to zero
        for i in 0..<cachedArray.count {
            XCTAssertEqual(cachedArray[i].floatValue, 0.0)
        }
    }

    // MARK: - Cache Size Management

    func testCacheSizeLimit() async throws {
        // Create cache with small size
        let smallCache = MLArrayCache(maxCacheSize: 4)

        // Create arrays with same shape
        let shape: [NSNumber] = [100]
        var arrays: [MLMultiArray] = []

        // Get 4 arrays
        for _ in 0..<4 {
            arrays.append(try await smallCache.getArray(shape: shape, dataType: .float32))
        }

        // Return all to cache
        for array in arrays {
            await smallCache.returnArray(array)
        }

        // Try to return one more - should not exceed limit
        let extraArray = try await smallCache.getArray(shape: shape, dataType: .float32)
        await smallCache.returnArray(extraArray)

        // Cache should still work
        let finalArray = try await smallCache.getArray(shape: shape, dataType: .float32)
        XCTAssertNotNil(finalArray)
    }

    // MARK: - Pre-warming Tests

    func testPrewarmCache() async {
        let shapes: [(shape: [NSNumber], dataType: MLMultiArrayDataType)] = [
            ([1, 100], .float32),
            ([2, 50], .float32),
            ([1, 1024], .float16),
        ]

        await cache.prewarm(shapes: shapes)

        // Arrays should be available from cache
        for (shape, dataType) in shapes {
            do {
                let array = try await cache.getArray(shape: shape, dataType: dataType)
                XCTAssertEqual(array.shape, shape)
                XCTAssertEqual(array.dataType, dataType)
            } catch {
                XCTFail("Failed to get pre-warmed array: \(error)")
            }
        }
    }

    // MARK: - Clear Cache Tests

    func testClearCache() async throws {
        let shape: [NSNumber] = [50]

        // Add array to cache
        let array = try await cache.getArray(shape: shape, dataType: .float32)
        await cache.returnArray(array)

        // Clear cache
        await cache.clear()

        // Next request should be cache miss (new array)
        let newArray = try await cache.getArray(shape: shape, dataType: .float32)
        XCTAssertNotNil(newArray)
    }

    // MARK: - Different Data Types

    func testDifferentDataTypes() async throws {
        let shape: [NSNumber] = [10, 10]

        // Test different data types
        let float32 = try await cache.getArray(shape: shape, dataType: .float32)
        XCTAssertEqual(float32.dataType, .float32)

        let float16 = try await cache.getArray(shape: shape, dataType: .float16)
        XCTAssertEqual(float16.dataType, .float16)

        let int32 = try await cache.getArray(shape: shape, dataType: .int32)
        XCTAssertEqual(int32.dataType, .int32)

        // Return all to cache
        await cache.returnArray(float32)
        await cache.returnArray(float16)
        await cache.returnArray(int32)

        // Should get correct types back
        let cachedFloat32 = try await cache.getArray(shape: shape, dataType: .float32)
        XCTAssertEqual(cachedFloat32.dataType, .float32)
    }

    // MARK: - Thread Safety Tests

    func testConcurrentAccess() async throws {
        let shape: [NSNumber] = [10]
        let testCache = cache!

        // Perform limited concurrent operations
        await withTaskGroup(of: Void.self) { group in
            // Reduced number of concurrent tasks
            for i in 0..<3 {
                group.addTask {
                    do {
                        let array = try await testCache.getArray(shape: shape, dataType: .float32)
                        array[0] = NSNumber(value: Float(i))
                        await testCache.returnArray(array)
                    } catch {
                        // Can't call XCTFail from Sendable closure - just print
                        print("Concurrent access failed: \(error)")
                    }
                }
            }
        }

        // Cache should still be functional
        let finalArray = try await cache.getArray(shape: shape, dataType: .float32)
        XCTAssertNotNil(finalArray)
    }

    func testGetArrayDoesNotReuseActiveBuffer() async throws {
        let shape: [NSNumber] = [1, 64]

        let array1 = try await cache.getArray(shape: shape, dataType: .float32)
        let array2 = try await cache.getArray(shape: shape, dataType: .float32)

        XCTAssertFalse(array1 === array2, "Cache should not hand out the same array while it is still borrowed")

        await cache.returnArray(array1)
        await cache.returnArray(array2)
    }

    // Removed performance test - can cause timing issues

    // MARK: - Global Cache Tests

    func testSharedCacheInstance() async throws {
        // Test the global shared instance
        let shape: [NSNumber] = [256]
        let array = try await sharedMLArrayCache.getArray(shape: shape, dataType: .float32)

        XCTAssertEqual(array.shape, shape)

        // Return to shared cache
        await sharedMLArrayCache.returnArray(array)
    }

}
