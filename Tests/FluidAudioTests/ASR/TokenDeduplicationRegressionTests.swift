import XCTest

@testable import FluidAudio

/// Regression tests for token deduplication algorithms.
/// These tests capture the behavior BEFORE refactoring to ensure the new SequenceMatcher
/// utilities preserve exact behavior after extraction.
final class TokenDeduplicationRegressionTests: XCTestCase {

    // MARK: - AsrManager Token Deduplication Tests

    /// Test punctuation deduplication (Stage 1)
    func testRemoveDuplicateTokenSequence_PunctuationDeduplication() async throws {
        let asrManager = AsrManager()

        // Test case: Punctuation token (7883 = period) duplicated at boundary
        let (deduped1, removed1) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101, 7883],
            current: [7883, 102, 103]
        )
        XCTAssertEqual(deduped1, [102, 103], "Should remove duplicate punctuation token")
        XCTAssertEqual(removed1, 1, "Should report 1 removed token")

        // Test case: Comma (7952) duplicated
        let (deduped2, removed2) = asrManager.removeDuplicateTokenSequence(
            previous: [200, 201, 7952],
            current: [7952, 202, 203]
        )
        XCTAssertEqual(deduped2, [202, 203], "Should remove duplicate comma")
        XCTAssertEqual(removed2, 1, "Should report 1 removed token")

        // Test case: Question mark (7948) duplicated
        let (deduped3, removed3) = asrManager.removeDuplicateTokenSequence(
            previous: [300, 301, 7948],
            current: [7948, 302, 303]
        )
        XCTAssertEqual(deduped3, [302, 303], "Should remove duplicate question mark")
        XCTAssertEqual(removed3, 1, "Should report 1 removed token")
    }

    /// Test suffix-prefix overlap (Stage 2)
    func testRemoveDuplicateTokenSequence_SuffixPrefixOverlap() async throws {
        let asrManager = AsrManager()

        // Test case: 2-token overlap
        let (deduped1, removed1) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101, 102],
            current: [101, 102, 103, 104]
        )
        XCTAssertEqual(deduped1, [103, 104], "Should remove 2-token overlap")
        XCTAssertEqual(removed1, 2, "Should report 2 removed tokens")

        // Test case: 3-token overlap
        let (deduped2, removed2) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101, 102, 103],
            current: [101, 102, 103, 104, 105]
        )
        XCTAssertEqual(deduped2, [104, 105], "Should remove 3-token overlap")
        XCTAssertEqual(removed2, 3, "Should report 3 removed tokens")

        // Test case: Maximum overlap (12 tokens - maxOverlap default)
        let previous = Array(100..<115)  // 15 tokens
        let current = Array(103..<120)  // Overlap of 12 tokens (103-114)
        let (deduped3, removed3) = asrManager.removeDuplicateTokenSequence(
            previous: previous,
            current: current
        )
        XCTAssertEqual(deduped3, Array(115..<120), "Should remove up to maxOverlap tokens")
        XCTAssertEqual(removed3, 12, "Should respect maxOverlap limit")
    }

    /// Test bounded substring search (Stage 3)
    func testRemoveDuplicateTokenSequence_BoundedSubstringSearch() async throws {
        let asrManager = AsrManager()

        // Test case: Overlap not at exact boundary but within search window
        let (deduped1, removed1) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101, 102, 103, 104],
            current: [999, 102, 103, 104, 105]  // Match starts at offset 1 in current
        )
        // Should find [102, 103, 104] match at position 1 in current
        XCTAssertEqual(deduped1, [105], "Should find and remove offset overlap")
        XCTAssertTrue(removed1 == 4, "Should remove offset (1) + match length (3)")
    }

    /// Test edge cases
    func testRemoveDuplicateTokenSequence_EdgeCases() async throws {
        let asrManager = AsrManager()

        // Test case: No overlap
        let (deduped1, removed1) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101, 102],
            current: [200, 201, 202]
        )
        XCTAssertEqual(deduped1, [200, 201, 202], "Should return original if no overlap")
        XCTAssertEqual(removed1, 0, "Should report 0 removed tokens")

        // Test case: Empty current
        let (deduped2, removed2) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101, 102],
            current: []
        )
        XCTAssertEqual(deduped2, [] as [Int], "Should return empty for empty current")
        XCTAssertEqual(removed2, 0, "Should report 0 removed tokens")

        // Test case: Empty previous
        let (deduped3, removed3) = asrManager.removeDuplicateTokenSequence(
            previous: [],
            current: [100, 101, 102]
        )
        XCTAssertEqual(deduped3, [100, 101, 102], "Should return original if previous empty")
        XCTAssertEqual(removed3, 0, "Should report 0 removed tokens")

        // Test case: Single token overlap (too short, minimum is 2)
        let (deduped4, removed4) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101],
            current: [101, 102]
        )
        // Single token overlaps are only handled for punctuation in Stage 1
        // For non-punctuation, minimum match is 2 tokens
        XCTAssertEqual(deduped4, [101, 102], "Should not remove single non-punctuation overlap")
        XCTAssertEqual(removed4, 0, "Should report 0 removed tokens")
    }

    /// Test combined scenarios (punctuation + overlap)
    func testRemoveDuplicateTokenSequence_CombinedScenarios() async throws {
        let asrManager = AsrManager()

        // Test case: Punctuation removal followed by suffix-prefix match
        // Previous ends with period, current starts with period + has overlap
        let (deduped1, removed1) = asrManager.removeDuplicateTokenSequence(
            previous: [100, 101, 7883],
            current: [7883, 101, 102, 103]
        )
        // Stage 1: Remove punctuation (7883), working = [101, 102, 103]
        // Stage 2: No suffix-prefix match because previous doesn't end with 101
        XCTAssertEqual(deduped1, [101, 102, 103], "Should only remove punctuation")
        XCTAssertEqual(removed1, 1, "Should report 1 removed (punctuation)")
    }

    // MARK: - SequenceMatcher Utility Tests

    /// Test SequenceMatcher.findSuffixPrefixMatch
    func testSequenceMatcher_SuffixPrefixMatch() {
        let previous = [100, 101, 102, 103]
        let current = [102, 103, 104, 105]

        let exactMatcher: (Int, Int) -> Bool = { $0 == $1 }

        let match = SequenceMatcher.findSuffixPrefixMatch(
            previous: previous,
            current: current,
            maxOverlap: 12,
            matcher: exactMatcher
        )

        XCTAssertNotNil(match, "Should find suffix-prefix match")
        XCTAssertEqual(match?.leftStartIndex, 2, "Should start at index 2 in previous")
        XCTAssertEqual(match?.rightStartIndex, 0, "Should start at index 0 in current")
        XCTAssertEqual(match?.length, 2, "Should match 2 tokens")
    }

    /// Test SequenceMatcher.findBoundedSubstringMatch
    func testSequenceMatcher_BoundedSubstringMatch() {
        let previous = [100, 101, 102, 103, 104]
        let current = [999, 102, 103, 104, 105]

        let exactMatcher: (Int, Int) -> Bool = { $0 == $1 }

        let match = SequenceMatcher.findBoundedSubstringMatch(
            previous: previous,
            current: current,
            maxSearchLength: 15,
            boundarySearchFrames: 10,
            matcher: exactMatcher
        )

        XCTAssertNotNil(match, "Should find bounded substring match")
        XCTAssertEqual(match?.leftStartIndex, 2, "Should start at index 2 in previous")
        XCTAssertEqual(match?.rightStartIndex, 1, "Should start at index 1 in current")
        XCTAssertEqual(match?.length, 3, "Should match 3 tokens")
    }

    /// Test SequenceMatcher.findLongestCommonSubsequence
    func testSequenceMatcher_LongestCommonSubsequence() {
        let left = [1, 2, 3, 4, 5]
        let right = [2, 3, 5, 6]

        let exactMatcher: (Int, Int) -> Bool = { $0 == $1 }

        let matches = SequenceMatcher.findLongestCommonSubsequence(
            left: left,
            right: right,
            matcher: exactMatcher
        )

        // LCS should find: 2, 3, 5 (indices: (1,0), (2,1), (4,2))
        XCTAssertEqual(matches.count, 3, "Should find 3 single-element matches")
        XCTAssertEqual(matches[0].leftStartIndex, 1, "First match at left[1]")
        XCTAssertEqual(matches[0].rightStartIndex, 0, "First match at right[0]")
        XCTAssertEqual(matches[1].leftStartIndex, 2, "Second match at left[2]")
        XCTAssertEqual(matches[1].rightStartIndex, 1, "Second match at right[1]")
        XCTAssertEqual(matches[2].leftStartIndex, 4, "Third match at left[4]")
        XCTAssertEqual(matches[2].rightStartIndex, 2, "Third match at right[2]")
    }

    /// Test SequenceMatcher.findContiguousMatches
    func testSequenceMatcher_ContiguousMatches() {
        let left = [1, 2, 3, 4, 5]
        let right = [2, 3, 4, 6, 7]

        let exactMatcher: (Int, Int) -> Bool = { $0 == $1 }

        let matches = SequenceMatcher.findContiguousMatches(
            left: left,
            right: right,
            matcher: exactMatcher
        )

        // Should find contiguous sequence: 2, 3, 4
        XCTAssertEqual(matches.count, 3, "Should find 3 contiguous matches")
        XCTAssertEqual(matches[0].leftStartIndex, 1, "First match at left[1]")
        XCTAssertEqual(matches[0].rightStartIndex, 0, "First match at right[0]")
        XCTAssertEqual(matches[1].leftStartIndex, 2, "Second match at left[2]")
        XCTAssertEqual(matches[1].rightStartIndex, 1, "Second match at right[1]")
        XCTAssertEqual(matches[2].leftStartIndex, 3, "Third match at left[3]")
        XCTAssertEqual(matches[2].rightStartIndex, 2, "Third match at right[2]")
    }

    /// Test SequenceMatcher.consolidateMatches
    func testSequenceMatcher_ConsolidateMatches() {
        // Create single-element matches that should be consolidated
        let matches = [
            SequenceMatch(leftStartIndex: 0, rightStartIndex: 0, length: 1),
            SequenceMatch(leftStartIndex: 1, rightStartIndex: 1, length: 1),
            SequenceMatch(leftStartIndex: 2, rightStartIndex: 2, length: 1),
            // Gap here
            SequenceMatch(leftStartIndex: 5, rightStartIndex: 5, length: 1),
            SequenceMatch(leftStartIndex: 6, rightStartIndex: 6, length: 1),
        ]

        let consolidated = SequenceMatcher<Int>.consolidateMatches(matches)

        XCTAssertEqual(consolidated.count, 2, "Should consolidate into 2 sequences")
        XCTAssertEqual(consolidated[0].leftStartIndex, 0, "First sequence starts at 0")
        XCTAssertEqual(consolidated[0].length, 3, "First sequence has length 3")
        XCTAssertEqual(consolidated[1].leftStartIndex, 5, "Second sequence starts at 5")
        XCTAssertEqual(consolidated[1].length, 2, "Second sequence has length 2")
    }

    // MARK: - Performance Tests

    /// Test performance of suffix-prefix matching
    func testPerformance_SuffixPrefixMatching() {
        let previous = Array(0..<1000)
        let current = Array(900..<2000)  // 100 token overlap

        let exactMatcher: (Int, Int) -> Bool = { $0 == $1 }

        measure {
            _ = SequenceMatcher.findSuffixPrefixMatch(
                previous: previous,
                current: current,
                maxOverlap: 12,
                matcher: exactMatcher
            )
        }
    }

    /// Test performance of LCS
    func testPerformance_LCS() {
        let left = Array(0..<100)
        let right = Array(50..<150)

        let exactMatcher: (Int, Int) -> Bool = { $0 == $1 }

        measure {
            _ = SequenceMatcher.findLongestCommonSubsequence(
                left: left,
                right: right,
                matcher: exactMatcher
            )
        }
    }
}
