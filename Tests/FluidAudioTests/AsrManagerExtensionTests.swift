import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
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

    // MARK: - removeDuplicateTokenSequence Tests

    func testRemoveDuplicateTokenSequenceNoDuplicates() {
        let previous = [1, 2, 3, 4, 5]
        let current = [6, 7, 8, 9, 10]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, current, "No duplicates should return original current")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testRemoveDuplicateTokenSequenceWithOverlap() {
        let previous = [1, 2, 3, 4, 5]
        let current = [3, 4, 5, 6, 7, 8]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should remove [3, 4, 5] from the beginning of current
        XCTAssertEqual(deduped, [6, 7, 8], "Should remove overlapping tokens")
        XCTAssertEqual(removedCount, 3, "Should remove 3 tokens")
    }

    func testRemoveDuplicateTokenSequencePunctuation() {
        let previous = [1, 2, 3, 7883]  // 7883 is period
        let current = [7883, 4, 5, 6]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should remove duplicate period
        XCTAssertEqual(deduped, [4, 5, 6], "Should remove duplicate punctuation")
        XCTAssertEqual(removedCount, 1, "Should remove 1 punctuation token")
    }

    func testRemoveDuplicateTokenSequenceQuestionMark() {
        let previous = [1, 2, 7952]  // 7952 is question mark
        let current = [7952, 3, 4]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, [3, 4], "Should remove duplicate question mark")
        XCTAssertEqual(removedCount, 1, "Should remove 1 token")
    }

    func testRemoveDuplicateTokenSequenceExclamation() {
        let previous = [1, 2, 7948]  // 7948 is exclamation
        let current = [7948, 3, 4]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, [3, 4], "Should remove duplicate exclamation")
        XCTAssertEqual(removedCount, 1, "Should remove 1 token")
    }

    func testRemoveDuplicateTokenSequenceEmptyPrevious() {
        let previous: [Int] = []
        let current = [1, 2, 3, 4]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, current, "Empty previous should return original current")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testRemoveDuplicateTokenSequenceEmptyCurrent() {
        let previous = [1, 2, 3, 4]
        let current: [Int] = []

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, current, "Empty current should return empty array")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testRemoveDuplicateTokenSequenceShortSequences() {
        let previous = [1, 2]
        let current = [2, 3]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Short sequences (< 3 tokens) should not be processed for overlap
        XCTAssertEqual(deduped, current, "Short sequences should return original")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testRemoveDuplicateTokenSequenceMaxOverlap() {
        // Test with overlap longer than default maxOverlap (12)
        let previous = Array(1...20)
        let current = Array(10...25)  // 10-21 overlaps with previous

        let (_, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find some overlap within maxOverlap limit
        XCTAssertGreaterThan(removedCount, 0, "Should find some overlap")
        XCTAssertLessThanOrEqual(removedCount, 12, "Should not exceed maxOverlap")
    }

    func testRemoveDuplicateTokenSequencePartialOverlap() {
        let previous = [10, 11, 12, 13, 14]
        let current = [12, 13, 14, 15, 16]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should remove [12, 13, 14] from current
        XCTAssertEqual(deduped, [15, 16], "Should remove partial overlap")
        XCTAssertEqual(removedCount, 3, "Should remove 3 overlapping tokens")
    }

    func testRemoveDuplicateTokenSequenceCustomMaxOverlap() {
        let previous = [1, 2, 3, 4, 5, 6]
        let current = [4, 5, 6, 7, 8, 9]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(
            previous: previous, current: current, maxOverlap: 2
        )

        // With maxOverlap=2, should not find the 3-token overlap but only 2-token overlap
        XCTAssertEqual(deduped, [6, 7, 8, 9], "Should not find overlap with small maxOverlap \(deduped)")
        XCTAssertEqual(removedCount, 2, "Should not remove any tokens with small maxOverlap")
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

    func testRemoveDuplicateTokenSequencePerformance() {
        let previous = Array(1...1000)
        let current = Array(990...1500)  // 10-token overlap

        measure {
            for _ in 0..<1000 {
                _ = manager.removeDuplicateTokenSequence(previous: previous, current: current)
            }
        }
    }
}
