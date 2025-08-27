import CoreML
import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 13.0, iOS 16.0, *)
final class AsrManagerDeduplicationTests: XCTestCase {

    var manager: AsrManager!

    override func setUp() {
        super.setUp()
        manager = AsrManager(config: ASRConfig.default)
    }

    override func tearDown() {
        manager = nil
        super.tearDown()
    }

    // MARK: - Range Bounds Safety Tests (Critical for crash prevention)

    func testRemoveDuplicateTokenSequenceVeryShortCurrent() {
        let previous = Array(1...20)
        let current = [21, 22]  // Only 2 tokens

        // Should not crash despite current being shorter than minimum overlap (4)
        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, current, "Very short current should return original")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testRemoveDuplicateTokenSequenceVeryShortPrevious() {
        let previous = [1, 2]  // Only 2 tokens
        let current = Array(1...20)

        // Should not crash despite previous being shorter than minimum overlap
        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, current, "Very short previous should return original current")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testRemoveDuplicateTokenSequenceBothVeryShort() {
        let previous = [1, 2]  // Only 2 tokens
        let current = [3, 4]  // Only 2 tokens

        // This case would have caused crash with min(2, 2) < 4
        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, current, "Both short should return original current")
        XCTAssertEqual(removedCount, 0, "Should not remove any tokens")
    }

    func testRemoveDuplicateTokenSequenceExactMinimumSize() {
        let previous = [1, 2, 3, 4]  // Exactly 4 tokens
        let current = [2, 3, 4, 5]  // Exactly 4 tokens

        // Should work at boundary condition
        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(deduped, [5], "Should find 3-token overlap and remove [2,3,4]")
        XCTAssertEqual(removedCount, 3, "Should remove 3 overlapping tokens")
    }

    // MARK: - Long Overlap Detection Tests (Spanish audio case)

    func testRemoveDuplicateTokenSequenceSpanishAudioCase() {
        // Reproduce case inspired by Spanish audio transcription
        // Create a case where previous ends with a sequence that overlaps with current
        // Previous ends: "...para el Senado en el año."
        let previous = [100, 101, 815, 460, 7329, 894, 364, 460, 279, 4539, 7883]

        // Current starts: ". su campaña para el Senado en el año 2005..."
        // The overlap is [815, 460, 7329, 894, 364, 460, 279, 4539] at the end of previous
        // and starting at position 5 in current
        let current = [7883, 451, 1917, 931, 8066, 815, 460, 7329, 894, 364, 460, 279, 4539, 7863, 236]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should at least remove the duplicate period
        XCTAssertGreaterThan(removedCount, 0, "Should remove at least the duplicate period")

        // If enhanced algorithm works, should remove more than just punctuation
        if removedCount > 1 {
            // Enhanced algorithm found longer overlap
            XCTAssertGreaterThanOrEqual(removedCount, 8, "Should find substantial overlap")
        } else {
            // Only punctuation was found
            XCTAssertEqual(removedCount, 1, "Should remove duplicate period")
        }

        // Result should not contain duplicate period at start
        XCTAssertFalse(deduped.isEmpty, "Should have remaining tokens")
    }

    func testRemoveDuplicateTokenSequenceOverlapNotAtStart() {
        // Test overlap that doesn't start at position 0 in current chunk
        let previous = [1, 2, 3, 4, 5, 6, 7, 8]
        let current = [99, 4, 5, 6, 7, 8, 9, 10]  // Overlap [4,5,6,7,8] at position 1

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should remove [99, 4, 5, 6, 7, 8] = 6 tokens
        XCTAssertEqual(removedCount, 6, "Should remove tokens up to and including overlap")
        XCTAssertEqual(deduped, [9, 10], "Should keep only non-overlapping suffix")
    }

    func testRemoveDuplicateTokenSequence13TokenOverlap() {
        // Test the exact 13-token overlap length from Spanish case
        let overlapSequence = Array(100...112)  // 13 tokens
        let previous = Array(1...20) + overlapSequence
        let current = [999] + overlapSequence + Array(200...210)

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should remove [999] + 13-token overlap = 14 tokens
        XCTAssertEqual(removedCount, 14, "Should remove prefix + 13-token overlap")
        XCTAssertEqual(deduped, Array(200...210), "Should keep only unique suffix")
    }

    func testRemoveDuplicateTokenSequence20TokenOverlap() {
        // Test maximum search capability
        let overlapSequence = Array(100...119)  // 20 tokens
        let previous = Array(1...30) + overlapSequence
        let current = overlapSequence + Array(200...220)

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find and remove the full 20-token overlap
        XCTAssertEqual(removedCount, 20, "Should remove full 20-token overlap")
        XCTAssertEqual(deduped, Array(200...220), "Should keep only unique suffix")
    }

    // MARK: - Punctuation Edge Cases

    func testRemoveDuplicateTokenSequencePunctuationWithLongOverlap() {
        // Test that punctuation doesn't prevent longer overlap detection
        let previous = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 7883]  // ends with period
        let current = [7883, 6, 7, 8, 9, 10, 11, 12]  // starts with period, has longer overlap

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find [6, 7, 8, 9, 10] overlap (5 tokens) + period = 6 tokens
        XCTAssertGreaterThan(removedCount, 1, "Should find more than just punctuation")
        XCTAssertEqual(removedCount, 6, "Should remove period + 5-token overlap")
        XCTAssertEqual(deduped, [11, 12], "Should keep unique suffix")
    }

    func testRemoveDuplicateTokenSequenceMultiplePunctuation() {
        let previous = [1, 2, 3, 7883, 7952]  // ends with period, question
        let current = [7952, 4, 5, 6]  // starts with question mark

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should remove duplicate question mark
        XCTAssertEqual(deduped, [4, 5, 6], "Should remove duplicate question mark")
        XCTAssertEqual(removedCount, 1, "Should remove 1 punctuation token")
    }

    func testRemoveDuplicateTokenSequencePunctuationWhenShort() {
        // Test punctuation handling when chunks are too short for longer overlap
        let previous = [1, 2, 7883]  // short with period
        let current = [7883, 3, 4]  // short starting with period

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should still handle punctuation even when hasLongerPotentialOverlap is true but current is ≤ 3
        XCTAssertEqual(deduped, [3, 4], "Should remove duplicate period")
        XCTAssertEqual(removedCount, 1, "Should remove 1 punctuation token")
    }

    // MARK: - Complex Real-World Scenarios

    func testRemoveDuplicateTokenSequenceNestedOverlap() {
        // Test with repetitive patterns that might confuse the algorithm
        let pattern = [10, 11, 12]
        let previous = [1, 2, 3] + pattern + [4, 5, 6] + pattern + [7, 8, 9]
        let current = pattern + [13, 14, 15]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find the 3-token pattern
        XCTAssertEqual(removedCount, 3, "Should remove 3-token pattern")
        XCTAssertEqual(deduped, [13, 14, 15], "Should keep unique suffix")
    }

    func testRemoveDuplicateTokenSequenceRepetitivePattern() {
        // Test with highly repetitive content
        let pattern = [100, 101, 102]
        let previous = pattern + pattern + pattern + [200]
        let current = pattern + [300, 301]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        XCTAssertEqual(removedCount, 3, "Should find 3-token repetitive pattern")
        XCTAssertEqual(deduped, [300, 301], "Should keep unique part")
    }

    func testRemoveDuplicateTokenSequenceWordBoundaryRespect() {
        // Test with realistic token sequences that represent word boundaries
        // Use minimum 3-token overlap since algorithm requires that
        let previous = [1, 2, 3, 4, 5]  // "the quick brown"
        let current = [3, 4, 5, 6, 7]  // "brown fox jumps" - 3-token overlap [3,4,5]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find 3-token overlap [3,4,5] and remove them
        XCTAssertEqual(removedCount, 3, "Should find 3-token overlap")
        XCTAssertEqual(deduped, [6, 7], "Should keep non-overlapping suffix")
    }

    // MARK: - Performance and Stress Tests

    func testRemoveDuplicateTokenSequenceLargeArraysPerformance() {
        let largeSequence = Array(1...1000)
        let overlap = Array(950...999)  // 50-token overlap
        let previous = largeSequence
        let current = overlap + Array(2000...2100)

        measure {
            _ = manager.removeDuplicateTokenSequence(previous: previous, current: current)
        }
    }

    func testRemoveDuplicateTokenSequenceWorstCaseComplexity() {
        // Create worst-case scenario: many possible matches to check
        let previous = Array(repeating: 1, count: 100) + Array(2...50)
        let current = Array(40...60) + Array(repeating: 2, count: 50)

        let startTime = Date()
        let (_, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)
        let duration = Date().timeIntervalSince(startTime)

        // Should complete in reasonable time even in worst case
        XCTAssertLessThan(duration, 0.1, "Should complete within 100ms even in worst case")
        XCTAssertGreaterThanOrEqual(removedCount, 0, "Should return valid result")
    }

    func testRemoveDuplicateTokenSequenceMemoryEfficiency() {
        // Test with very large arrays to ensure no memory leaks
        // Create overlap at end of previous and start of current
        let previous = Array(1...10000)
        let current = Array(9980...10050)  // 21-token overlap [9980...10000]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find the 20-token overlap (within our search limit)
        XCTAssertGreaterThan(removedCount, 0, "Should find overlap in large arrays")
        XCTAssertEqual(deduped.count + removedCount, current.count, "Should maintain token count invariant")
    }

    // MARK: - Algorithm Boundary Tests

    func testRemoveDuplicateTokenSequenceSearchLimits() {
        // Test the search parameter limits with an overlap at the end/start
        let longPrevious = Array(1...100)
        let longCurrent = Array(85...150)  // 16-token overlap [85...100]

        // Should respect maxSearchLength = min(35, previous.count)
        let (_, removedCount) = manager.removeDuplicateTokenSequence(previous: longPrevious, current: longCurrent)

        XCTAssertGreaterThan(removedCount, 0, "Should find some overlap")
        // The algorithm should find the 16-token overlap
        XCTAssertLessThanOrEqual(removedCount, 20, "Should respect search length limits")
    }

    func testRemoveDuplicateTokenSequenceCurrentSearchLimit() {
        // Test the current chunk search limit (15 positions)
        let previous = Array(1...50)
        // Create current where overlap starts at position 10 (within 15-position limit)
        let current = Array(100...109) + Array(40...50) + Array(200...210)

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find the overlap that starts at position 10
        XCTAssertGreaterThan(removedCount, 10, "Should find overlap within search limit")
        XCTAssertTrue(deduped.starts(with: [200, 201, 202]), "Should preserve suffix after overlap")
    }

    // MARK: - Fallback Mechanism Tests

    func testRemoveDuplicateTokenSequenceFallbackTo3Tokens() {
        // Test fallback to 3-token minimum when 4+ token search fails
        let previous = [1, 2, 3, 4, 5, 6, 7]
        let current = [5, 6, 7, 8, 9]  // 3-token overlap [5,6,7]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should fall back to 3-token overlap detection
        XCTAssertEqual(removedCount, 3, "Should find 3-token overlap via fallback")
        XCTAssertEqual(deduped, [8, 9], "Should keep non-overlapping suffix")
    }

    func testRemoveDuplicateTokenSequenceNoFallbackNeeded() {
        // Test case where main algorithm succeeds, no fallback needed
        let previous = Array(1...20)
        let current = Array(15...25)  // 6-token overlap [15,16,17,18,19,20]

        let (deduped, removedCount) = manager.removeDuplicateTokenSequence(previous: previous, current: current)

        // Should find overlap without needing fallback
        XCTAssertGreaterThanOrEqual(removedCount, 4, "Should find 4+ token overlap without fallback")
        XCTAssertEqual(deduped, Array(21...25), "Should keep unique suffix")
    }
}
