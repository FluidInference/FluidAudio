@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class EmbeddingExtractorMemoryTests: XCTestCase {

    // MARK: - Memory Tests Without Real Models

    func testEmbeddingExtractorInitialization() throws {
        // Test that EmbeddingExtractor can be initialized without crashing
        // This tests the pre-allocation of ANE-aligned buffers

        // Since we can't create a real MLModel in tests, we'll skip this
        throw XCTSkip("Cannot test without real MLModel")
    }

    func testANEBufferAlignment() throws {
        // Test that buffers created for embedding extraction are ANE-aligned
        let optimizer = ANEMemoryOptimizer()

        // Test waveform buffer creation
        let waveformBuffer = try optimizer.createAlignedArray(
            shape: [3, 240000] as [NSNumber],
            dataType: .float32
        )

        let address = Int(bitPattern: waveformBuffer.dataPointer)
        XCTAssertEqual(
            address % ANEMemoryOptimizer.aneAlignment, 0,
            "Waveform buffer should be ANE-aligned")

        // Test mask buffer creation
        let maskBuffer = try optimizer.createAlignedArray(
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )

        let maskAddress = Int(bitPattern: maskBuffer.dataPointer)
        XCTAssertEqual(
            maskAddress % ANEMemoryOptimizer.aneAlignment, 0,
            "Mask buffer should be ANE-aligned")
    }

    func testBufferPoolingForMasks() throws {
        let optimizer = ANEMemoryOptimizer()

        // Test that mask buffers of different sizes are pooled correctly
        let sizes = [100, 500, 1000, 2000]

        for size in sizes {
            let key = "wespeaker_mask_\(size)"
            let shape = [3, size] as [NSNumber]

            // Get buffer from pool
            let buffer1 = try optimizer.getPooledBuffer(
                key: key,
                shape: shape,
                dataType: .float32
            )

            // Get same buffer again - should be identical
            let buffer2 = try optimizer.getPooledBuffer(
                key: key,
                shape: shape,
                dataType: .float32
            )

            XCTAssertTrue(
                buffer1 === buffer2,
                "Same mask buffer should be reused from pool")
        }
    }

    func testMemoryStressWithMultipleMasks() throws {
        let optimizer = ANEMemoryOptimizer()

        // Simulate processing many speakers with different mask sizes
        autoreleasepool {
            for iteration in 0..<100 {
                for speakerCount in [1, 2, 3, 4] {
                    let maskSize = 100 + iteration * 10
                    let key = "stress_test_\(speakerCount)_\(maskSize)"

                    _ = try? optimizer.getPooledBuffer(
                        key: key,
                        shape: [3, maskSize] as [NSNumber],
                        dataType: .float32
                    )
                }
            }
        }

        // Clear pool and verify we can still allocate
        optimizer.clearBufferPool()

        let finalBuffer = try optimizer.getPooledBuffer(
            key: "after_stress",
            shape: [3, 100] as [NSNumber],
            dataType: .float32
        )

        XCTAssertEqual(finalBuffer.count, 300)
    }

    func testZeroCopyFeatureProvider() throws {
        let optimizer = ANEMemoryOptimizer()

        // Create aligned buffers
        let waveformBuffer = try optimizer.createAlignedArray(
            shape: [3, 240000] as [NSNumber],
            dataType: .float32
        )

        let maskBuffer = try optimizer.createAlignedArray(
            shape: [3, 1000] as [NSNumber],
            dataType: .float32
        )

        // Fill with test data
        for i in 0..<min(1000, waveformBuffer.count) {
            waveformBuffer[i] = NSNumber(value: Float(i) * 0.001)
        }

        for i in 0..<maskBuffer.count {
            maskBuffer[i] = NSNumber(value: Float(i % 2))
        }

        // Create zero-copy feature provider
        let features: [String: MLFeatureValue] = [
            "waveform": MLFeatureValue(multiArray: waveformBuffer),
            "mask": MLFeatureValue(multiArray: maskBuffer),
        ]

        let provider = ZeroCopyDiarizerFeatureProvider(features: features)

        // Verify features
        XCTAssertEqual(provider.featureNames.sorted(), ["mask", "waveform"])

        let waveformFeature = provider.featureValue(for: "waveform")
        XCTAssertNotNil(waveformFeature)
        XCTAssertTrue(
            waveformFeature?.multiArrayValue === waveformBuffer,
            "Should use same buffer instance")

        let maskFeature = provider.featureValue(for: "mask")
        XCTAssertNotNil(maskFeature)
        XCTAssertTrue(
            maskFeature?.multiArrayValue === maskBuffer,
            "Should use same buffer instance")
    }

    func testConcurrentBufferAccess() {
        // Test concurrent buffer access by running operations sequentially
        // Swift 6 strict concurrency prevents capturing ANEMemoryOptimizer in concurrent closures
        let optimizer = ANEMemoryOptimizer()

        for index in 0..<10 {
            autoreleasepool {
                do {
                    // Each iteration gets its own waveform buffer
                    let waveformBuffer = try optimizer.getPooledBuffer(
                        key: "concurrent_waveform_\(index)",
                        shape: [3, 240000] as [NSNumber],
                        dataType: .float32
                    )

                    // Shared mask buffers (simulating speaker masks)
                    let maskBuffer = try optimizer.getPooledBuffer(
                        key: "concurrent_mask_\(index % 3)",
                        shape: [3, 1000] as [NSNumber],
                        dataType: .float32
                    )

                    // Simulate some processing
                    waveformBuffer[0] = NSNumber(value: Float(index))
                    maskBuffer[0] = NSNumber(value: Float(index))

                    XCTAssertEqual(waveformBuffer[0].floatValue, Float(index))
                } catch {
                    XCTFail("Buffer access failed: \(error)")
                }
            }
        }
    }
}
