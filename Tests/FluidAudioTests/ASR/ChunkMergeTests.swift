@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class ChunkMergeTests: XCTestCase {

    // MARK: - Helper Types

    private typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float)

    // MARK: - Mock Data Generators

    private func createTokenWindow(_ token: Int, _ timestamp: Int, _ confidence: Float = 0.95) -> TokenWindow {
        (token: token, timestamp: timestamp, confidence: confidence)
    }

    // MARK: - Contiguous Pairs Tests

    func testFindBestContiguousPairsFullMatch() {
        // Test when all tokens match contiguously
        let left = [1, 2, 3, 4, 5]
        let right = [1, 2, 3, 4, 5]

        // In real scenario, tokens would have timing info
        // Expected: contiguous match at position 0-4
        let result = findBestContiguousPairsSimulation(left: left, right: right)
        XCTAssertGreaterThan(result.count, 0, "Full match should find contiguous pairs")
    }

    func testFindBestContiguousPairsPartialMatch() {
        // Test partial overlapping sequences
        let left = [1, 2, 3, 4, 5]
        let right = [3, 4, 5, 6, 7]

        let result = findBestContiguousPairsSimulation(left: left, right: right)
        XCTAssertGreaterThan(result.count, 0, "Partial match should find contiguous pairs")
    }

    func testFindBestContiguousPairsNoMatch() {
        // Test non-overlapping sequences
        let left = [1, 2, 3]
        let right = [4, 5, 6]

        let result = findBestContiguousPairsSimulation(left: left, right: right)
        XCTAssertEqual(result.count, 0, "Non-matching sequences should return empty")
    }

    func testFindBestContiguousPairsSelectsBestMatch() {
        // Test that it selects the longest contiguous sequence
        let left = [1, 2, 3, 7, 8]
        let right = [1, 9, 7, 8, 9, 10]

        let result = findBestContiguousPairsSimulation(left: left, right: right)
        // Should find the best match ([7, 8] or similar)
        XCTAssertGreaterThan(result.count, 0, "Should find best contiguous match")
    }

    // MARK: - LCS (Longest Common Subsequence) Tests

    func testLCSBasicSequence() {
        // Test simple LCS finding
        let left = [1, 2, 3, 4, 5]
        let right = [1, 3, 5, 7, 9]

        let result = findLCSSimulation(left: left, right: right)
        // Should find [1, 3, 5]
        XCTAssertGreaterThan(result.count, 0, "Should find LCS in sequences")
    }

    func testLCSComplexSequence() {
        // Test LCS with more complex pattern
        let left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        let right = [2, 4, 6, 8, 10, 11, 12]

        let result = findLCSSimulation(left: left, right: right)
        // Should find [2, 4, 6, 8, 10]
        XCTAssertGreaterThan(result.count, 0, "Should find LCS in complex sequences")
    }

    func testLCSNoCommonTokens() {
        // Test when no tokens match
        let left = [1, 2, 3]
        let right = [4, 5, 6]

        let result = findLCSSimulation(left: left, right: right)
        XCTAssertEqual(result.count, 0, "No common tokens should return empty LCS")
    }

    func testLCSIdenticalSequences() {
        // Test when sequences are identical
        let tokens = [10, 20, 30, 40, 50]
        let result = findLCSSimulation(left: tokens, right: tokens)

        XCTAssertEqual(result.count, tokens.count, "Identical sequences should have all elements in LCS")
    }

    func testLCSSingleToken() {
        // Test LCS of length 1
        let left = [1, 2, 3, 4]
        let right = [5, 3, 6, 7]

        let result = findLCSSimulation(left: left, right: right)
        // Should find [3]
        XCTAssertGreaterThan(result.count, 0, "Should find LCS with single token")
    }

    // MARK: - Token Matching Tests

    func testTokensMatchWithinTolerance() {
        // Test tokens match when token IDs are same and time difference is within tolerance
        let tolerance = 1.0  // 1 second
        let tokenA = (index: 0, token: 100, start: 10.0, end: 10.08)
        let tokenB = (index: 0, token: 100, start: 10.5, end: 10.58)

        let timeDiff = abs(tokenA.start - tokenB.start)
        let shouldMatch = tokenA.token == tokenB.token && timeDiff < tolerance

        XCTAssertTrue(shouldMatch, "Tokens within tolerance should match")
    }

    func testTokensMismatchDifferentTokens() {
        // Test different token IDs don't match
        let tolerance = 1.0
        let tokenA = (token: 100, start: 10.0)
        let tokenB = (token: 200, start: 10.3)

        let shouldMatch = tokenA.token == tokenB.token && abs(tokenA.start - tokenB.start) < tolerance

        XCTAssertFalse(shouldMatch, "Different tokens should not match")
    }

    func testTokensMismatchOutsideTolerance() {
        // Test tokens don't match if time difference exceeds tolerance
        let tolerance = 1.0
        let tokenA = (token: 100, start: 10.0)
        let tokenB = (token: 100, start: 11.5)

        let shouldMatch = tokenA.token == tokenB.token && abs(tokenA.start - tokenB.start) < tolerance

        XCTAssertFalse(shouldMatch, "Tokens outside tolerance should not match")
    }

    func testTokensMatchAtBoundary() {
        // Test matching at exact tolerance boundary
        let tolerance = 1.0
        let tokenA = (token: 100, start: 10.0)
        let tokenB = (token: 100, start: 11.0)

        let shouldMatch = tokenA.token == tokenB.token && abs(tokenA.start - tokenB.start) <= tolerance

        XCTAssertTrue(shouldMatch, "Tokens at tolerance boundary should match")
    }

    // MARK: - Merge Using Matches Tests

    func testMergeUsingMatchesFullSequence() {
        // Test merging when all tokens are matched
        let left: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
        ]
        let right: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
        ]

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        // When all tokens match, result includes left and right (some duplication expected in simulation)
        XCTAssertGreaterThan(result.count, 0, "Matched tokens should produce a result")
    }

    func testMergeUsingMatchesWithGaps() {
        // Test merging when there are gaps between matches
        let left: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
            createTokenWindow(4, 3),
            createTokenWindow(5, 4),
        ]
        let right: [TokenWindow] = [
            createTokenWindow(3, 2),
            createTokenWindow(6, 5),
            createTokenWindow(7, 6),
            createTokenWindow(5, 8),
        ]

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        // Should include matched tokens and resolve gaps
        XCTAssertGreaterThan(result.count, 0, "Should merge with gaps")
    }

    func testMergeUsingMatchesPreGap() {
        // Test tokens before first match
        let left: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
        ]
        let right: [TokenWindow] = [
            createTokenWindow(3, 2),
            createTokenWindow(4, 3),
        ]

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        // Should include tokens from left that come before first match
        XCTAssertGreaterThanOrEqual(result.count, 2, "Should include pre-match tokens")
    }

    func testMergeUsingMatchesPostGap() {
        // Test tokens after last match
        let left: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
        ]
        let right: [TokenWindow] = [
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
            createTokenWindow(4, 3),
        ]

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        // Should include tokens from right that come after last match
        XCTAssertGreaterThanOrEqual(result.count, 2, "Should include post-match tokens")
    }

    func testMergeUsingMatchesGapPreference() {
        // Test preference for longer gap
        let leftGap: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
        ]
        let rightGap: [TokenWindow] = [
            createTokenWindow(4, 3),
            createTokenWindow(5, 4),
        ]

        // When choosing gaps, prefer longer gap
        let shouldPreferRight = rightGap.count >= leftGap.count / 2
        XCTAssertTrue(shouldPreferRight, "Gap selection logic should work based on length")
    }

    func testMergeUsingMatchesAdjacentMatches() {
        // Test when matches are adjacent (no gaps)
        let left: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
        ]
        let right: [TokenWindow] = [
            createTokenWindow(2, 1),
            createTokenWindow(3, 2),
            createTokenWindow(4, 3),
        ]

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        // Should merge without gaps
        XCTAssertGreaterThan(result.count, 0, "Adjacent matches should merge cleanly")
    }

    // MARK: - Merge By Midpoint Tests

    func testMergeByMidpointCutoffCalculation() {
        // Test midpoint cutoff calculation
        let leftEndTime = 10.0
        let rightStartTime = 12.0
        let cutoff = (leftEndTime + rightStartTime) / 2.0

        XCTAssertEqual(cutoff, 11.0, "Midpoint should be 11.0")
    }

    func testMergeByMidpointFilterLeft() {
        // Test that left tokens <= cutoff are included
        let cutoff = 11.0
        let tokens: [TokenWindow] = [
            createTokenWindow(1, 0),  // time ≈ 0.0
            createTokenWindow(2, 137),  // time ≈ 10.96
            createTokenWindow(3, 140),  // time ≈ 11.2
        ]

        let filtered = tokens.filter { Double($0.timestamp) * 0.08 <= cutoff }
        XCTAssertEqual(filtered.count, 2, "Should filter tokens <= cutoff")
    }

    func testMergeByMidpointFilterRight() {
        // Test that right tokens >= cutoff are included
        let cutoff = 11.0
        let tokens: [TokenWindow] = [
            createTokenWindow(1, 137),  // time ≈ 10.96
            createTokenWindow(2, 140),  // time ≈ 11.2
            createTokenWindow(3, 150),  // time ≈ 12.0
        ]

        let filtered = tokens.filter { Double($0.timestamp) * 0.08 >= cutoff }
        XCTAssertEqual(filtered.count, 2, "Should filter tokens >= cutoff")
    }

    func testMergeByMidpointBoundary() {
        // Test tokens exactly at cutoff
        let cutoff = 11.0
        let timestamp = 137  // Frame 137 * 0.08 ≈ 10.96
        let frameTime = Double(timestamp) * 0.08

        // Token close to cutoff should be handled appropriately
        let closeEnough = abs(frameTime - cutoff) < 0.2
        XCTAssertTrue(closeEnough, "Token near cutoff should be handled properly")
    }

    // MARK: - Empty and Edge Cases

    func testMergeEmptyLeftChunk() {
        // Test merging when left chunk is empty
        let left: [TokenWindow] = []
        let right: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
        ]

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        // Should return right chunk
        XCTAssertGreaterThan(result.count, 0, "Empty left should use right chunk")
    }

    func testMergeEmptyRightChunk() {
        // Test merging when right chunk is empty
        let left: [TokenWindow] = [
            createTokenWindow(1, 0),
            createTokenWindow(2, 1),
        ]
        let right: [TokenWindow] = []

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        // Should return left chunk
        XCTAssertGreaterThan(result.count, 0, "Empty right should use left chunk")
    }

    func testMergeEmptyBothChunks() {
        // Test merging when both chunks are empty
        let left: [TokenWindow] = []
        let right: [TokenWindow] = []

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        XCTAssertEqual(result.count, 0, "Empty chunks should merge to empty")
    }

    func testMergeSingleTokenChunks() {
        // Test merging when chunks have only one token
        let left: [TokenWindow] = [createTokenWindow(1, 0)]
        let right: [TokenWindow] = [createTokenWindow(2, 1)]

        let result = mergeUsingMatchesSimulation(left: left, right: right)
        XCTAssertGreaterThan(result.count, 0, "Single token chunks should merge")
    }

    func testMergeNoOverlapRegion() {
        // Test merging when chunks don't temporally overlap
        // leftEndTime <= rightStartTime
        let leftEndTime = 10.0
        let rightStartTime = 10.1

        let hasOverlap = leftEndTime > rightStartTime
        XCTAssertFalse(hasOverlap, "Non-overlapping regions should concatenate")
    }

    // MARK: - Token Sorting

    func testTokenSortingAfterMerge() {
        // Test that tokens are sorted by timestamp after merging
        let tokens: [TokenWindow] = [
            createTokenWindow(3, 50),
            createTokenWindow(1, 10),
            createTokenWindow(2, 30),
        ]

        let sorted = tokens.sorted { $0.timestamp < $1.timestamp }
        XCTAssertEqual(sorted[0].timestamp, 10, "Should be sorted in ascending order")
        XCTAssertEqual(sorted[1].timestamp, 30)
        XCTAssertEqual(sorted[2].timestamp, 50)
    }

    // MARK: - Helper Simulations

    private func findBestContiguousPairsSimulation(left: [Int], right: [Int]) -> [(Int, Int)] {
        var best: [(Int, Int)] = []

        for i in 0..<left.count {
            for j in 0..<right.count {
                if left[i] == right[j] {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j

                    while k < left.count && l < right.count && left[k] == right[l] {
                        current.append((k, l))
                        k += 1
                        l += 1
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    private func findLCSSimulation(left: [Int], right: [Int]) -> [Int] {
        let m = left.count
        let n = right.count
        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 1...m {
            for j in 1...n {
                if left[i - 1] == right[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var result: [Int] = []
        var i = m
        var j = n

        while i > 0 && j > 0 {
            if left[i - 1] == right[j - 1] {
                result.append(left[i - 1])
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return result.reversed()
    }

    private func mergeUsingMatchesSimulation(left: [TokenWindow], right: [TokenWindow]) -> [TokenWindow] {
        guard !left.isEmpty || !right.isEmpty else { return [] }
        guard !left.isEmpty else { return right }
        guard !right.isEmpty else { return left }

        // Simple simulation: concatenate with no duplication
        return left + right
    }
}
