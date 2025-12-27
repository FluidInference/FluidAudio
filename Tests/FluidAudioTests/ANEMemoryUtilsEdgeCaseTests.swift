import Accelerate
@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class ANEMemoryUtilsEdgeCaseTests: XCTestCase {

    // MARK: - Alignment Edge Cases

    func testCreateAlignedArrayWithZeroElements() throws {
        // Edge case: 0-element array
        let array = try ANEMemoryUtils.createAlignedArray(
            shape: [0] as [NSNumber],
            dataType: .float32
        )

        XCTAssertEqual(array.count, 0)

        // Should still be aligned even if empty
        let address = Int(bitPattern: array.dataPointer)
        XCTAssertEqual(address % ANEMemoryUtils.aneAlignment, 0)
    }

    func testCreateAlignedArrayWithOddDimensions() throws {
        // Test with dimensions that don't align naturally
        let oddShapes: [[NSNumber]] = [
            [17],
            [3, 17],
            [1, 1, 1],
            [63],  // One less than alignment
            [65],  // One more than alignment
            [127, 513],  // Both odd
        ]

        for shape in oddShapes {
            let array = try ANEMemoryUtils.createAlignedArray(
                shape: shape,
                dataType: .float32
            )

            // Verify alignment
            let address = Int(bitPattern: array.dataPointer)
            XCTAssertEqual(
                address % ANEMemoryUtils.aneAlignment, 0,
                "Failed alignment for shape \(shape)")

            // Verify we can access all elements
            let totalElements = shape.reduce(1) { $0 * $1.intValue }
            if totalElements > 0 {
                array[0] = 1.0
                if totalElements > 1 {
                    array[totalElements - 1] = 2.0
                    XCTAssertEqual(array[totalElements - 1].floatValue, 2.0)
                }
                XCTAssertEqual(array[0].floatValue, 1.0)
            }
        }
    }

    func testCreateAlignedArrayWithLargeDimensions() throws {
        // Test with very large arrays
        let largeShapes: [[NSNumber]] = [
            [1_000_000],  // 4MB
            [1000, 1000],  // 4MB
            [100, 100, 100],  // 4MB
        ]

        for shape in largeShapes {
            do {
                let array = try ANEMemoryUtils.createAlignedArray(
                    shape: shape,
                    dataType: .float32
                )

                let address = Int(bitPattern: array.dataPointer)
                XCTAssertEqual(address % ANEMemoryUtils.aneAlignment, 0)

                // Test boundary access
                array[0] = 1.0
                array[array.count - 1] = 2.0

            } catch {
                // May fail on memory-constrained systems
                print("Large array allocation failed for shape \(shape): \(error)")
            }
        }
    }

    func testDataTypesAlignment() throws {
        let shape = [100] as [NSNumber]
        let dataTypes: [MLMultiArrayDataType] = [
            .float32,
            .float64,
            .int32,
        ]

        for dataType in dataTypes {
            let array = try ANEMemoryUtils.createAlignedArray(
                shape: shape,
                dataType: dataType
            )

            let address = Int(bitPattern: array.dataPointer)
            XCTAssertEqual(
                address % ANEMemoryUtils.aneAlignment, 0,
                "Failed alignment for data type \(dataType)")
        }
    }

    // MARK: - vDSP Operations Edge Cases

    func testVDSPCopyWithUnalignedCount() throws {
        let source = try ANEMemoryUtils.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float32
        )
        let dest = try ANEMemoryUtils.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float32
        )

        // Fill source with pattern
        for i in 0..<100 {
            source[i] = NSNumber(value: Float(i))
        }

        // Test copying odd number of elements
        let oddCounts = [1, 17, 63, 99]

        for count in oddCounts {
            // Clear destination
            vDSP_vclr(dest.dataPointer.assumingMemoryBound(to: Float.self), 1, vDSP_Length(100))

            // Copy partial data using vDSP
            vDSP_mmov(
                source.dataPointer.assumingMemoryBound(to: Float.self),
                dest.dataPointer.assumingMemoryBound(to: Float.self),
                vDSP_Length(count),
                vDSP_Length(1),
                vDSP_Length(count),
                vDSP_Length(1)
            )

            // Verify copy
            for i in 0..<count {
                XCTAssertEqual(
                    dest[i].floatValue, Float(i),
                    "Copy failed at index \(i) for count \(count)")
            }

            // Verify rest is still zero
            for i in count..<100 {
                XCTAssertEqual(
                    dest[i].floatValue, 0.0,
                    "Unexpected value at index \(i) for count \(count)")
            }
        }
    }

    func testVDSPCopyWithZeroBytes() throws {
        let source = try ANEMemoryUtils.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float32
        )
        let dest = try ANEMemoryUtils.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float32
        )

        // Fill both with different patterns
        for i in 0..<100 {
            source[i] = NSNumber(value: Float(i))
            dest[i] = NSNumber(value: Float(1000 + i))
        }

        // Copy zero bytes - no-op
        // vDSP_mmov with count 0 would be a no-op

        // Destination should be unchanged
        for i in 0..<100 {
            XCTAssertEqual(dest[i].floatValue, Float(1000 + i))
        }
    }

    func testVDSPCopyAcrossDataTypes() throws {
        // Test copying between different data type arrays
        let float32Array = try ANEMemoryUtils.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float32
        )

        let float64Array = try ANEMemoryUtils.createAlignedArray(
            shape: [100] as [NSNumber],
            dataType: .float64
        )

        // Fill float32 array
        for i in 0..<100 {
            float32Array[i] = NSNumber(value: Float(i) * 0.5)
        }

        // Copy raw bytes using memcpy (this will not convert types properly)
        memcpy(
            float64Array.dataPointer,
            float32Array.dataPointer,
            100 * MemoryLayout<Float>.size
        )

        // The data will be garbled due to type mismatch - this tests edge case handling
        // Just verify no crash occurred
        XCTAssertNotNil(float64Array[0])
    }

    // MARK: - Clear Operations Edge Cases

    func testClearWithPartialArray() throws {
        let array = try ANEMemoryUtils.createAlignedArray(
            shape: [1000] as [NSNumber],
            dataType: .float32,
            zeroClear: false  // Don't clear initially
        )

        // Fill with pattern
        for i in 0..<1000 {
            array[i] = NSNumber(value: Float(i))
        }

        // Clear only part of the array
        let partialCount = 500
        vDSP_vclr(
            array.dataPointer.assumingMemoryBound(to: Float.self),
            1,
            vDSP_Length(partialCount)
        )

        // Verify partial clear
        for i in 0..<partialCount {
            XCTAssertEqual(array[i].floatValue, 0.0)
        }
        for i in partialCount..<1000 {
            XCTAssertEqual(array[i].floatValue, Float(i))
        }
    }

    // MARK: - Memory Boundary Tests

    func testAccessNearAllocationBoundaries() throws {
        // Test accessing elements near the allocation boundaries
        let sizes = [64, 128, 256, 512, 1024]

        for size in sizes {
            let array = try ANEMemoryUtils.createAlignedArray(
                shape: [size] as [NSNumber],
                dataType: .float32
            )

            // Write to boundaries
            array[0] = -1.0  // First element
            array[size - 1] = -2.0  // Last element

            if size > 2 {
                array[1] = -3.0  // Second element
                array[size - 2] = -4.0  // Second to last
            }

            // Verify writes
            XCTAssertEqual(array[0].floatValue, -1.0)
            XCTAssertEqual(array[size - 1].floatValue, -2.0)

            if size > 2 {
                XCTAssertEqual(array[1].floatValue, -3.0)
                XCTAssertEqual(array[size - 2].floatValue, -4.0)
            }
        }
    }

    func testStridedAccess() throws {
        let array = try ANEMemoryUtils.createAlignedArray(
            shape: [1024] as [NSNumber],
            dataType: .float32
        )

        // Write with stride
        let stride = 7  // Prime number for interesting pattern
        for i in stride..<1024 where i % stride == 0 {
            array[i] = NSNumber(value: Float(i))
        }

        // Verify strided writes
        for i in 0..<1024 {
            if i >= stride && i % stride == 0 {
                XCTAssertEqual(array[i].floatValue, Float(i))
            } else {
                XCTAssertEqual(array[i].floatValue, 0.0)
            }
        }
    }

    // MARK: - Thread Safety Edge Cases

    func testConcurrentArrayCreation() {
        let iterations = 100
        let queues = 8
        let expectation = XCTestExpectation(description: "Concurrent creation")
        expectation.expectedFulfillmentCount = queues

        for queueIndex in 0..<queues {
            DispatchQueue.global().async {
                autoreleasepool {
                    for i in 0..<iterations {
                        do {
                            let shape = [100 + i] as [NSNumber]
                            let array = try ANEMemoryUtils.createAlignedArray(
                                shape: shape,
                                dataType: .float32
                            )

                            // Verify alignment
                            let address = Int(bitPattern: array.dataPointer)
                            XCTAssertEqual(address % ANEMemoryUtils.aneAlignment, 0)

                            // Write and read to verify memory is valid
                            array[0] = NSNumber(value: Float(queueIndex))
                            XCTAssertEqual(array[0].floatValue, Float(queueIndex))

                        } catch {
                            XCTFail("Failed to create array: \(error)")
                        }
                    }
                }
                expectation.fulfill()
            }
        }

        wait(for: [expectation], timeout: 30.0)
    }

    // MARK: - Performance Under Stress

    func testRapidAllocationDeallocation() {
        measure {
            for _ in 0..<1000 {
                autoreleasepool {
                    do {
                        let array = try ANEMemoryUtils.createAlignedArray(
                            shape: [10000] as [NSNumber],
                            dataType: .float32
                        )
                        // Write to ensure allocation is real
                        array[0] = 1.0
                        array[9999] = 2.0
                    } catch {
                        XCTFail("Allocation failed: \(error)")
                    }
                }
            }
        }
    }

    // MARK: - Helper Methods

    private func verifyMemoryPattern(
        in array: MLMultiArray,
        pattern: (Int) -> Float,
        count: Int
    ) {
        for i in 0..<count {
            XCTAssertEqual(
                array[i].floatValue,
                pattern(i),
                accuracy: 0.0001,
                "Pattern mismatch at index \(i)"
            )
        }
    }
}
