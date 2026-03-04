@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class AsrManagerExtensionTests: XCTestCase {

    var manager: AsrManager!

    override func setUp() {
        super.setUp()
        manager = AsrManager(config: ASRConfig.default)
    }

    override func tearDown() {
        manager = nil
        super.tearDown()
    }

    // MARK: - padAudioIfNeeded Tests

    func testPadAudioIfNeededNopadding() {
        let audioSamples: [Float] = [0.1, 0.2, 0.3, 0.4, 0.5]
        let targetLength = 3

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return original array when it's already longer than target
        XCTAssertEqual(result, audioSamples)
        XCTAssertEqual(result.count, 5)
    }

    func testPadAudioIfNeededWithPadding() {
        let audioSamples: [Float] = [0.1, 0.2, 0.3]
        let targetLength = 7

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should pad with zeros
        let expected: [Float] = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0]
        XCTAssertEqual(result, expected)
        XCTAssertEqual(result.count, targetLength)
    }

    func testPadAudioIfNeededExactLength() {
        let audioSamples: [Float] = [0.1, 0.2, 0.3]
        let targetLength = 3

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return original array when exactly the target length
        XCTAssertEqual(result, audioSamples)
        XCTAssertEqual(result.count, 3)
    }

    func testPadAudioIfNeededEmptyArray() {
        let audioSamples: [Float] = []
        let targetLength = 5

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return array of zeros
        let expected: [Float] = [0.0, 0.0, 0.0, 0.0, 0.0]
        XCTAssertEqual(result, expected)
        XCTAssertEqual(result.count, targetLength)
    }

    func testPadAudioIfNeededZeroTarget() {
        let audioSamples: [Float] = [0.1, 0.2]
        let targetLength = 0

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        // Should return original array when target is 0
        XCTAssertEqual(result, audioSamples)
        XCTAssertEqual(result.count, 2)
    }

    func testPadAudioIfNeededLargeArray() {
        let audioSamples: [Float] = Array(repeating: 0.5, count: 1000)
        let targetLength = 1500

        let result = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)

        XCTAssertEqual(result.count, targetLength)
        // First 1000 should be 0.5
        XCTAssertEqual(Array(result.prefix(1000)), audioSamples)
        // Last 500 should be 0.0
        XCTAssertEqual(Array(result.suffix(500)), Array(repeating: 0.0, count: 500))
    }

    // MARK: - calculateStartFrameOffset Tests

    func testCalculateStartFrameOffsetFirstSegment() {
        let offset = manager.calculateStartFrameOffset(segmentIndex: 0, leftContextSeconds: 2.0)

        // Method is deprecated - now always returns 0 (frame tracking handled by timeJump mechanism)
        XCTAssertEqual(offset, 0)
    }

    func testCalculateStartFrameOffsetSecondSegment() {
        let leftContext = 2.0
        let offset = manager.calculateStartFrameOffset(segmentIndex: 1, leftContextSeconds: leftContext)

        // Method is deprecated - now always returns 0 (frame tracking handled by timeJump mechanism)
        XCTAssertEqual(offset, 0)
    }

    func testCalculateStartFrameOffsetThirdSegment() {
        let leftContext = 1.5
        let offset = manager.calculateStartFrameOffset(segmentIndex: 2, leftContextSeconds: leftContext)

        // Method is deprecated - now always returns 0 (frame tracking handled by timeJump mechanism)
        XCTAssertEqual(offset, 0)
    }

    func testCalculateStartFrameOffsetVariousContexts() {
        // Method is deprecated - now always returns 0 (frame tracking handled by timeJump mechanism)
        let testCases: [(leftContext: Double, expected: Int)] = [
            (0.0, 0),  // No context
            (0.08, 0),  // Method always returns 0
            (0.16, 0),  // Method always returns 0
            (1.0, 0),  // Method always returns 0
            (3.2, 0),  // Method always returns 0
        ]

        for (leftContext, expected) in testCases {
            let offset = manager.calculateStartFrameOffset(segmentIndex: 1, leftContextSeconds: leftContext)
            XCTAssertEqual(offset, expected, "Failed for leftContext=\(leftContext)")
        }
    }

    func testCalculateStartFrameOffsetNegativeSegment() {
        let offset = manager.calculateStartFrameOffset(segmentIndex: -1, leftContextSeconds: 2.0)

        // Method is deprecated - now always returns 0 (frame tracking handled by timeJump mechanism)
        XCTAssertEqual(offset, 0)
    }

    // MARK: - Performance Tests

    func testPadAudioPerformance() {
        let audioSamples = Array(repeating: Float(0.5), count: 100_000)
        let targetLength = 240_000  // 15 seconds at 16kHz

        measure {
            for _ in 0..<100 {
                _ = manager.padAudioIfNeeded(audioSamples, targetLength: targetLength)
            }
        }
    }

    func testCalculateStartFrameOffsetPerformance() {
        measure {
            for i in 0..<10_000 {
                _ = manager.calculateStartFrameOffset(segmentIndex: i % 100, leftContextSeconds: 2.0)
            }
        }
    }
}
