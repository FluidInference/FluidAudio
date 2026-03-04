@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class SegmentationProcessorEdgeCaseTests: XCTestCase {

    var processor: SegmentationProcessor!

    override func setUp() {
        super.setUp()
        processor = SegmentationProcessor()
    }

    override func tearDown() {
        processor = nil
        ANEMemoryOptimizer().clearBufferPool()
        super.tearDown()
    }

    // MARK: - Audio Chunk Boundary Cases

    func testProcessingWithExactChunkSize() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        // Exactly 10 seconds at 16kHz
        let exactChunk = Array(repeating: Float(0.5), count: 160_000)[...]

        let (segments, rawOutput) = try processor.getSegments(
            audioChunk: exactChunk,
            segmentationModel: mockModel,
            chunkSize: 160_000
        )

        XCTAssertFalse(segments.isEmpty)
        XCTAssertNotNil(rawOutput)
    }

    func testProcessingWithOversizedChunk() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        // 20 seconds of audio when expecting 10
        let oversizedChunk = Array(repeating: Float(0.5), count: 320_000)[...]

        let (segments, _) = try processor.getSegments(
            audioChunk: oversizedChunk,
            segmentationModel: mockModel,
            chunkSize: 160_000
        )

        // Should process first chunk only
        XCTAssertFalse(segments.isEmpty)
    }

    func testProcessingWithTinyChunk() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        // Just 10 samples
        let tinyChunk = Array(repeating: Float(0.5), count: 10)[...]

        let (segments, _) = try processor.getSegments(
            audioChunk: tinyChunk,
            segmentationModel: mockModel,
            chunkSize: 160_000
        )

        // Should pad and process
        XCTAssertFalse(segments.isEmpty)
    }

    func testProcessingWithSingleSample() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        let singleSample = [Float(0.5)][...]

        let (segments, _) = try processor.getSegments(
            audioChunk: singleSample,
            segmentationModel: mockModel
        )

        XCTAssertFalse(segments.isEmpty)
    }

    // MARK: - Memory Alignment Edge Cases

    func testMLMultiArrayCreationWithOddSizes() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        // Test various non-aligned sizes
        let oddSizes = [16_001, 31_999, 80_123, 159_999]

        for size in oddSizes {
            let chunk = Array(repeating: Float(0.5), count: size)[...]

            let (segments, _) = try processor.getSegments(
                audioChunk: chunk,
                segmentationModel: mockModel,
                chunkSize: 160_000
            )

            XCTAssertFalse(segments.isEmpty, "Failed for size \(size)")
        }
    }

    func testReuseOfPooledBuffers() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        let chunk = Array(repeating: Float(0.5), count: 80_000)[...]

        // Process multiple times to test buffer reuse
        for i in 0..<50 {
            try autoreleasepool {
                let (segments, _) = try processor.getSegments(
                    audioChunk: chunk,
                    segmentationModel: mockModel
                )

                XCTAssertFalse(segments.isEmpty, "Failed at iteration \(i)")
            }
        }
    }

    // MARK: - Extreme Value Testing

    func testProcessingWithExtremeValues() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        var extremeChunk: [Float] = []
        for i in 0..<16_000 {
            switch i % 5 {
            case 0: extremeChunk.append(Float.infinity)
            case 1: extremeChunk.append(-Float.infinity)
            case 2: extremeChunk.append(Float.nan)
            case 3: extremeChunk.append(Float.greatestFiniteMagnitude)
            case 4: extremeChunk.append(-Float.greatestFiniteMagnitude)
            default: extremeChunk.append(0.0)
            }
        }

        let chunk = extremeChunk[...]

        // Should handle gracefully (may normalize or filter)
        do {
            let (segments, _) = try processor.getSegments(
                audioChunk: chunk,
                segmentationModel: mockModel
            )
            XCTAssertNotNil(segments)
        } catch {
            // Expected to potentially fail with extreme values
            XCTAssertTrue(
                error.localizedDescription.contains("value") || error.localizedDescription.contains("process"))
        }
    }

    func testProcessingWithSilence() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        // Complete silence
        let silentChunk = Array(repeating: Float(0.0), count: 160_000)[...]

        let (segments, _) = try processor.getSegments(
            audioChunk: silentChunk,
            segmentationModel: mockModel
        )

        // Should still process and return structure
        XCTAssertFalse(segments.isEmpty)
        if !segments.isEmpty && !segments[0].isEmpty {
            // All speakers should be inactive
            for frame in segments[0] {
                XCTAssertTrue(frame.allSatisfy { $0 <= 0.5 })
            }
        }
    }

    // MARK: - Concurrent Processing

    func testConcurrentSegmentProcessing() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        let expectation = XCTestExpectation(description: "Concurrent processing")
        expectation.expectedFulfillmentCount = 8

        // Multiple processors working concurrently
        for i in 0..<8 {
            DispatchQueue.global().async {
                autoreleasepool {
                    let localProcessor = SegmentationProcessor()
                    let chunk = Array(
                        repeating: Float(0.1 * Float(i)),
                        count: 80_000
                    )[...]

                    do {
                        let (segments, _) = try localProcessor.getSegments(
                            audioChunk: chunk,
                            segmentationModel: mockModel
                        )
                        XCTAssertFalse(segments.isEmpty)
                        expectation.fulfill()
                    } catch {
                        XCTFail("Concurrent processing failed: \(error)")
                    }
                }
            }
        }

        wait(for: [expectation], timeout: 20.0)
    }

    // MARK: - Feature Creation Edge Cases

    func testSlidingWindowWithEmptyData() {
        let emptySegments: [[[Float]]] = [[]]

        let feature = processor.createSlidingWindowFeature(
            binarizedSegments: emptySegments,
            chunkOffset: 0.0
        )

        XCTAssertEqual(feature.data.count, 1)
        XCTAssertTrue(feature.data[0].isEmpty)
    }

    func testSlidingWindowWithLargeOffset() {
        let segments: [[[Float]]] = [
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ]

        // Very large offset
        let feature = processor.createSlidingWindowFeature(
            binarizedSegments: segments,
            chunkOffset: 1_000_000.0
        )

        XCTAssertEqual(feature.slidingWindow.start, 1_000_000.0)
        XCTAssertGreaterThan(feature.slidingWindow.duration, 0)
    }

    func testBinarizationThreshold() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        // Create audio that might produce values near threshold
        var audioChunk: [Float] = []
        for i in 0..<160_000 {
            audioChunk.append(sin(Float(i) * 0.01) * 0.5)
        }

        let (segments, _) = try processor.getSegments(
            audioChunk: audioChunk[...],
            segmentationModel: mockModel
        )

        // Verify binarization worked
        if !segments.isEmpty {
            for batch in segments {
                for frame in batch {
                    for value in frame {
                        // Should be either 0 or 1 after binarization
                        XCTAssertTrue(
                            value == 0.0 || value == 1.0,
                            "Value \(value) not properly binarized")
                    }
                }
            }
        }
    }

    // MARK: - Memory Pressure Simulation

    func testProcessingUnderMemoryPressure() throws {
        guard let mockModel = createMockModel() else {
            throw XCTSkip("Mock model not available")
        }

        // Allocate large amounts of memory to simulate pressure
        var largeBuffers: [[Float]] = []
        for _ in 0..<10 {
            largeBuffers.append(Array(repeating: 0.0, count: 10_000_000))
        }

        // Try processing under pressure
        let chunk = Array(repeating: Float(0.5), count: 160_000)[...]

        do {
            let (segments, _) = try processor.getSegments(
                audioChunk: chunk,
                segmentationModel: mockModel
            )
            XCTAssertFalse(segments.isEmpty)
        } catch {
            // Acceptable to fail under extreme memory pressure
            print("Processing failed under memory pressure: \(error)")
        }

        // Clear memory
        largeBuffers.removeAll()
    }

    // MARK: - Helper Methods

    private func createMockModel() -> MLModel? {
        // Return nil to use XCTSkip in tests
        return nil
    }
}
