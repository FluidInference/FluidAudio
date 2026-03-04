@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class ANEMemoryOptimizerTests: XCTestCase {

    var optimizer: ANEMemoryOptimizer!

    override func setUp() {
        super.setUp()
        optimizer = ANEMemoryOptimizer()
    }

    override func tearDown() {
        // Clean up any pooled buffers
        optimizer.clearBufferPool()
        super.tearDown()
    }

    // MARK: - Buffer Creation Tests

    func testCreateAlignedArray() throws {
        let shape = [3, 240000] as [NSNumber]
        let array = try optimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )

        XCTAssertEqual(array.shape, shape)
        XCTAssertEqual(array.dataType, .float32)

        // Verify alignment
        let dataPointer = array.dataPointer
        let address = Int(bitPattern: dataPointer)
        XCTAssertEqual(
            address % ANEMemoryOptimizer.aneAlignment, 0,
            "Array should be 64-byte aligned")
    }

    func testCreateSmallAlignedArray() throws {
        // Test with very small array
        let shape = [1, 10] as [NSNumber]
        let array = try optimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )

        XCTAssertEqual(array.shape, shape)
        XCTAssertEqual(array.count, 10)
    }

    func testCreateLargeAlignedArray() throws {
        // Test with large array (10MB)
        let shape = [1, 2_500_000] as [NSNumber]
        let array = try optimizer.createAlignedArray(
            shape: shape,
            dataType: .float32
        )

        XCTAssertEqual(array.shape, shape)
        XCTAssertEqual(array.count, 2_500_000)
    }

    // MARK: - Buffer Pooling Tests

    func testBufferPoolReuse() throws {
        let shape = [3, 1000] as [NSNumber]
        let key = "test_buffer"

        // Get buffer from pool
        let buffer1 = try optimizer.getPooledBuffer(
            key: key,
            shape: shape,
            dataType: .float32
        )

        // Get same buffer again
        let buffer2 = try optimizer.getPooledBuffer(
            key: key,
            shape: shape,
            dataType: .float32
        )

        // Should be the same instance
        XCTAssertTrue(buffer1 === buffer2)
    }

    func testBufferPoolDifferentKeys() throws {
        let shape = [3, 1000] as [NSNumber]

        let buffer1 = try optimizer.getPooledBuffer(
            key: "buffer1",
            shape: shape,
            dataType: .float32
        )

        let buffer2 = try optimizer.getPooledBuffer(
            key: "buffer2",
            shape: shape,
            dataType: .float32
        )

        // Should be different instances
        XCTAssertFalse(buffer1 === buffer2)
    }

    func testBufferPoolShapeChange() throws {
        let key = "changing_buffer"

        // Create with one shape
        let buffer1 = try optimizer.getPooledBuffer(
            key: key,
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )
        XCTAssertEqual(buffer1.count, 3000)

        // Request with different shape
        let buffer2 = try optimizer.getPooledBuffer(
            key: key,
            shape: [5, 2000] as [NSNumber],
            dataType: .float32
        )
        XCTAssertEqual(buffer2.count, 10000)

        // Should be different buffers
        XCTAssertFalse(buffer1 === buffer2)
    }

    func testClearPool() throws {
        // Create multiple buffers
        _ = try optimizer.getPooledBuffer(
            key: "buffer1",
            shape: [100] as [NSNumber],
            dataType: .float32
        )
        _ = try optimizer.getPooledBuffer(
            key: "buffer2",
            shape: [200] as [NSNumber],
            dataType: .float32
        )

        // Clear pool
        optimizer.clearBufferPool()

        // Get buffer again - should be new instance
        let newBuffer = try optimizer.getPooledBuffer(
            key: "buffer1",
            shape: [100] as [NSNumber],
            dataType: .float32
        )

        // Verify it's properly initialized
        XCTAssertEqual(newBuffer.count, 100)
    }

    // MARK: - Zero-Copy Feature Provider Tests

    func testZeroCopyFeatureProvider() throws {
        let waveform = try optimizer.createAlignedArray(
            shape: [3, 240000] as [NSNumber],
            dataType: .float32
        )
        let mask = try optimizer.createAlignedArray(
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )

        // Fill with test data
        for i in 0..<waveform.count {
            waveform[i] = NSNumber(value: Float(i) / Float(waveform.count))
        }

        let features: [String: MLFeatureValue] = [
            "logmel": MLFeatureValue(multiArray: waveform),
            "mask": MLFeatureValue(multiArray: mask),
        ]

        let provider = ZeroCopyDiarizerFeatureProvider(features: features)

        XCTAssertEqual(provider.featureNames, ["logmel", "mask"])

        // Verify features
        let waveformFeature = provider.featureValue(for: "logmel")
        XCTAssertNotNil(waveformFeature)
        XCTAssertEqual(waveformFeature?.multiArrayValue?.count, waveform.count)

        let maskFeature = provider.featureValue(for: "mask")
        XCTAssertNotNil(maskFeature)
        XCTAssertEqual(maskFeature?.multiArrayValue?.count, mask.count)
    }

    // MARK: - Memory Pressure Tests

    func testMemoryPressureHandling() throws {
        // Create many buffers to simulate memory pressure
        var buffers: [MLMultiArray] = []

        for i in 0..<100 {
            let buffer = try optimizer.getPooledBuffer(
                key: "pressure_test_\(i)",
                shape: [1000, 1000] as [NSNumber],  // 4MB each
                dataType: .float32
            )
            buffers.append(buffer)
        }

        // Clear pool should release references
        optimizer.clearBufferPool()

        // Verify we can still create new buffers
        let newBuffer = try optimizer.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float32
        )
        XCTAssertEqual(newBuffer.count, 100)
    }

    // MARK: - Thread Safety Tests

    func testConcurrentBufferAccess() {
        let expectation = XCTestExpectation(description: "Concurrent access")
        expectation.expectedFulfillmentCount = 10

        DispatchQueue.concurrentPerform(iterations: 10) { index in
            do {
                let buffer = try optimizer.getPooledBuffer(
                    key: "concurrent_\(index % 3)",  // Share some keys
                    shape: [100] as [NSNumber],
                    dataType: .float32
                )
                XCTAssertEqual(buffer.count, 100)
                expectation.fulfill()
            } catch {
                XCTFail("Failed to get buffer: \(error)")
            }
        }

        wait(for: [expectation], timeout: 5.0)
    }

    // MARK: - Performance Tests

    func testBufferCreationPerformance() {
        measure {
            do {
                for _ in 0..<100 {
                    _ = try optimizer.createAlignedArray(
                        shape: [3, 16000] as [NSNumber],
                        dataType: .float32
                    )
                }
            } catch {
                XCTFail("Performance test failed: \(error)")
            }
        }
    }

    func testBufferPoolingPerformance() {
        measure {
            do {
                for i in 0..<1000 {
                    _ = try optimizer.getPooledBuffer(
                        key: "perf_test_\(i % 10)",  // Reuse 10 buffers
                        shape: [3, 16000] as [NSNumber],
                        dataType: .float32
                    )
                }
            } catch {
                XCTFail("Performance test failed: \(error)")
            }
        }
    }
}
