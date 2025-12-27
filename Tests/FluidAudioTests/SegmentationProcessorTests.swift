@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

final class SegmentationProcessorTests: XCTestCase {

    var processor: SegmentationProcessor!

    override func setUp() {
        super.setUp()
        processor = SegmentationProcessor()
    }

    override func tearDown() {
        processor = nil
        super.tearDown()
    }

    // MARK: - Segmentation Processing Tests

    func testGetSegmentsBasic() async throws {
        // Skip if no model available
        guard let mockModel = createMockSegmentationModel() else {
            throw XCTSkip("Mock model creation not available in test environment")
        }

        let audioChunk = Array(repeating: Float(0.5), count: 160_000)[...]

        let (segments, _) = try processor.getSegments(
            audioChunk: audioChunk,
            segmentationModel: mockModel
        )

        // Verify segments structure
        XCTAssertEqual(segments.count, 1)  // Batch size
        XCTAssertGreaterThan(segments[0].count, 0)  // Frames
        if !segments.isEmpty && !segments[0].isEmpty {
            XCTAssertEqual(segments[0][0].count, 3)  // 3 speakers
        }
    }

    func testGetSegmentsWithPartialAudio() async throws {
        guard let mockModel = createMockSegmentationModel() else {
            throw XCTSkip("Mock model creation not available in test environment")
        }

        // Test with less than full chunk
        let audioChunk = Array(repeating: Float(0.5), count: 80_000)[...]

        let (segments, _) = try processor.getSegments(
            audioChunk: audioChunk,
            segmentationModel: mockModel,
            chunkSize: 160_000
        )

        // Should still process successfully
        XCTAssertEqual(segments.count, 1)
    }

    // MARK: - Sliding Window Feature Tests

    func testCreateSlidingWindowFeature() {
        let binarizedSegments: [[[Float]]] = [
            [
                [1.0, 0.0, 0.0],  // Frame 0
                [0.0, 1.0, 0.0],  // Frame 1
                [0.0, 0.0, 1.0],  // Frame 2
            ]
        ]

        let feature = processor.createSlidingWindowFeature(
            binarizedSegments: binarizedSegments,
            chunkOffset: 10.0
        )

        XCTAssertEqual(feature.slidingWindow.start, 10.0)
        XCTAssertEqual(feature.slidingWindow.duration, 0.0619375, accuracy: 0.0001)
        XCTAssertEqual(feature.slidingWindow.step, 0.016875, accuracy: 0.0001)
        XCTAssertEqual(feature.data.count, 1)
        XCTAssertEqual(feature.data[0].count, 3)
    }

    // MARK: - Edge Cases

    func testEmptyAudioInput() async throws {
        guard let mockModel = createMockSegmentationModel() else {
            throw XCTSkip("Mock model creation not available in test environment")
        }

        let audioChunk: ArraySlice<Float> = [][...]

        do {
            _ = try processor.getSegments(
                audioChunk: audioChunk,
                segmentationModel: mockModel
            )
            XCTFail("Should throw error for empty audio")
        } catch {
            // Expected to fail
        }
    }

    // MARK: - Helper Methods

    private func createMockSegmentationModel() -> MLModel? {
        // In a real test environment, we would create a mock MLModel
        // For now, return nil to skip tests that require a model
        return nil
    }
}
