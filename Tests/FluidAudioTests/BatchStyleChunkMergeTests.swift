import Foundation
import XCTest

@testable import FluidAudio

/// Tests for the chunk merging algorithm used by BatchStyleStreamingManager.
/// These tests simulate the merge logic since the actual methods are private to the actor.
final class BatchStyleChunkMergeTests: XCTestCase {

    // MARK: - Helper Types

    private typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float)

    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }

    private let sampleRate: Int = 16000
    private let frameDuration: Double = Double(ASRConstants.samplesPerEncoderFrame) / 16000.0

    // MARK: - Token Creation Helpers

    private func createToken(_ tokenId: Int, frameIndex: Int, confidence: Float = 0.95) -> TokenWindow {
        (token: tokenId, timestamp: frameIndex, confidence: confidence)
    }

    private func startTime(of token: TokenWindow) -> Double {
        Double(token.timestamp) * frameDuration
    }

    private func endTime(of token: TokenWindow) -> Double {
        startTime(of: token) + frameDuration
    }

    // MARK: - Empty Chunk Tests

    func testMergeEmptyLeftChunk() {
        let left: [TokenWindow] = []
        let right: [TokenWindow] = [
            createToken(100, frameIndex: 0),
            createToken(200, frameIndex: 10),
        ]

        let result = mergeChunksSimulation(left, right, overlapSeconds: 2.0)
        XCTAssertEqual(result.count, 2, "Empty left should return right chunk")
        XCTAssertEqual(result[0].token, 100)
        XCTAssertEqual(result[1].token, 200)
    }

    func testMergeEmptyRightChunk() {
        let left: [TokenWindow] = [
            createToken(100, frameIndex: 0),
            createToken(200, frameIndex: 10),
        ]
        let right: [TokenWindow] = []

        let result = mergeChunksSimulation(left, right, overlapSeconds: 2.0)
        XCTAssertEqual(result.count, 2, "Empty right should return left chunk")
        XCTAssertEqual(result[0].token, 100)
        XCTAssertEqual(result[1].token, 200)
    }

    func testMergeBothEmpty() {
        let left: [TokenWindow] = []
        let right: [TokenWindow] = []

        let result = mergeChunksSimulation(left, right, overlapSeconds: 2.0)
        XCTAssertEqual(result.count, 0, "Both empty should return empty")
    }

    // MARK: - Non-Overlapping Chunk Tests

    func testMergeNonOverlappingChunks() {
        // Left ends at frame 100, right starts at frame 200 (no temporal overlap)
        let left: [TokenWindow] = [
            createToken(1, frameIndex: 0),
            createToken(2, frameIndex: 50),
            createToken(3, frameIndex: 100),
        ]
        let right: [TokenWindow] = [
            createToken(4, frameIndex: 200),
            createToken(5, frameIndex: 250),
        ]

        let result = mergeChunksSimulation(left, right, overlapSeconds: 2.0)

        // Should simply concatenate
        XCTAssertEqual(result.count, 5, "Non-overlapping should concatenate")
        XCTAssertEqual(result.map { $0.token }, [1, 2, 3, 4, 5])
    }

    // MARK: - Contiguous Pair Matching Tests

    func testFindContiguousPairsFullMatch() {
        // Identical overlapping regions
        let overlapLeft: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
            IndexedToken(index: 2, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
        ]
        let overlapRight: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
            IndexedToken(index: 2, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
        ]

        let pairs = findBestContiguousPairsSimulation(overlapLeft, overlapRight, tolerance: 1.0)

        XCTAssertEqual(pairs.count, 3, "Full match should find all 3 pairs")
    }

    func testFindContiguousPairsPartialMatch() {
        // Left has [1, 2, 3, 4, 5], right has [3, 4, 5, 6, 7]
        let overlapLeft: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
            IndexedToken(index: 2, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
            IndexedToken(index: 3, token: createToken(4, frameIndex: 103), start: 8.24, end: 8.32),
            IndexedToken(index: 4, token: createToken(5, frameIndex: 104), start: 8.32, end: 8.40),
        ]
        let overlapRight: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
            IndexedToken(index: 1, token: createToken(4, frameIndex: 103), start: 8.24, end: 8.32),
            IndexedToken(index: 2, token: createToken(5, frameIndex: 104), start: 8.32, end: 8.40),
            IndexedToken(index: 3, token: createToken(6, frameIndex: 105), start: 8.40, end: 8.48),
            IndexedToken(index: 4, token: createToken(7, frameIndex: 106), start: 8.48, end: 8.56),
        ]

        let pairs = findBestContiguousPairsSimulation(overlapLeft, overlapRight, tolerance: 1.0)

        XCTAssertEqual(pairs.count, 3, "Should find 3 matching pairs (tokens 3, 4, 5)")
    }

    func testFindContiguousPairsNoMatch() {
        // No matching tokens
        let overlapLeft: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
        ]
        let overlapRight: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(10, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(20, frameIndex: 101), start: 8.08, end: 8.16),
        ]

        let pairs = findBestContiguousPairsSimulation(overlapLeft, overlapRight, tolerance: 1.0)

        XCTAssertEqual(pairs.count, 0, "Different tokens should not match")
    }

    // MARK: - Token Matching Tests

    func testTokensMatchSameTokenWithinTolerance() {
        let left = IndexedToken(index: 0, token: createToken(100, frameIndex: 50), start: 4.0, end: 4.08)
        let right = IndexedToken(index: 0, token: createToken(100, frameIndex: 51), start: 4.08, end: 4.16)

        let matches = tokensMatchSimulation(left, right, tolerance: 0.5)
        XCTAssertTrue(matches, "Same token within 0.5s tolerance should match")
    }

    func testTokensMatchDifferentTokens() {
        let left = IndexedToken(index: 0, token: createToken(100, frameIndex: 50), start: 4.0, end: 4.08)
        let right = IndexedToken(index: 0, token: createToken(200, frameIndex: 50), start: 4.0, end: 4.08)

        let matches = tokensMatchSimulation(left, right, tolerance: 1.0)
        XCTAssertFalse(matches, "Different token IDs should not match")
    }

    func testTokensMatchOutsideTolerance() {
        let left = IndexedToken(index: 0, token: createToken(100, frameIndex: 50), start: 4.0, end: 4.08)
        let right = IndexedToken(index: 0, token: createToken(100, frameIndex: 200), start: 16.0, end: 16.08)

        let matches = tokensMatchSimulation(left, right, tolerance: 1.0)
        XCTAssertFalse(matches, "Same token but >1s apart should not match")
    }

    func testTokensMatchAtBoundary() {
        let left = IndexedToken(index: 0, token: createToken(100, frameIndex: 50), start: 4.0, end: 4.08)
        let right = IndexedToken(index: 0, token: createToken(100, frameIndex: 62), start: 4.96, end: 5.04)  // 0.96s apart

        let matches = tokensMatchSimulation(left, right, tolerance: 1.0)
        XCTAssertTrue(matches, "0.96s apart with 1.0s tolerance should match")
    }

    // MARK: - LCS (Longest Common Subsequence) Tests

    func testLCSFullMatch() {
        let overlapLeft: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
            IndexedToken(index: 2, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
        ]
        let overlapRight: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
            IndexedToken(index: 2, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
        ]

        let lcs = findLCSSimulation(overlapLeft, overlapRight, tolerance: 1.0)

        XCTAssertEqual(lcs.count, 3, "Identical sequences should have LCS of full length")
    }

    func testLCSNonContiguousMatch() {
        // Left has [1, 2, 3, 4, 5], right has [1, X, 3, X, 5] - LCS should find [1, 3, 5]
        let overlapLeft: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
            IndexedToken(index: 2, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
            IndexedToken(index: 3, token: createToken(4, frameIndex: 103), start: 8.24, end: 8.32),
            IndexedToken(index: 4, token: createToken(5, frameIndex: 104), start: 8.32, end: 8.40),
        ]
        let overlapRight: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(99, frameIndex: 101), start: 8.08, end: 8.16),
            IndexedToken(index: 2, token: createToken(3, frameIndex: 102), start: 8.16, end: 8.24),
            IndexedToken(index: 3, token: createToken(98, frameIndex: 103), start: 8.24, end: 8.32),
            IndexedToken(index: 4, token: createToken(5, frameIndex: 104), start: 8.32, end: 8.40),
        ]

        let lcs = findLCSSimulation(overlapLeft, overlapRight, tolerance: 1.0)

        XCTAssertGreaterThanOrEqual(lcs.count, 3, "LCS should find at least [1, 3, 5]")
    }

    func testLCSNoCommonTokens() {
        let overlapLeft: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(1, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(2, frameIndex: 101), start: 8.08, end: 8.16),
        ]
        let overlapRight: [IndexedToken] = [
            IndexedToken(index: 0, token: createToken(10, frameIndex: 100), start: 8.0, end: 8.08),
            IndexedToken(index: 1, token: createToken(20, frameIndex: 101), start: 8.08, end: 8.16),
        ]

        let lcs = findLCSSimulation(overlapLeft, overlapRight, tolerance: 1.0)

        XCTAssertEqual(lcs.count, 0, "No common tokens should return empty LCS")
    }

    // MARK: - Midpoint Merge Tests

    func testMergeByMidpointCalculation() {
        let leftEndTime = 10.0
        let rightStartTime = 12.0
        let cutoff = (leftEndTime + rightStartTime) / 2.0

        XCTAssertEqual(cutoff, 11.0, "Midpoint of 10.0 and 12.0 should be 11.0")
    }

    func testMergeByMidpointFilterLeft() {
        let cutoff = 11.0

        let leftTokens: [TokenWindow] = [
            createToken(1, frameIndex: 0),    // ~0s
            createToken(2, frameIndex: 125),  // ~10s
            createToken(3, frameIndex: 140),  // ~11.2s
        ]

        let filtered = leftTokens.filter { startTime(of: $0) <= cutoff }
        XCTAssertEqual(filtered.count, 2, "Should keep tokens at or before 11s cutoff")
    }

    func testMergeByMidpointFilterRight() {
        let cutoff = 11.0

        let rightTokens: [TokenWindow] = [
            createToken(1, frameIndex: 125),  // ~10s
            createToken(2, frameIndex: 140),  // ~11.2s
            createToken(3, frameIndex: 150),  // ~12s
        ]

        let filtered = rightTokens.filter { startTime(of: $0) >= cutoff }
        XCTAssertEqual(filtered.count, 2, "Should keep tokens at or after 11s cutoff")
    }

    func testMergeByMidpointCombined() {
        let leftEndTime = 12.0
        let rightStartTime = 10.0  // Overlapping
        let cutoff = (leftEndTime + rightStartTime) / 2.0  // 11.0

        let left: [TokenWindow] = [
            createToken(1, frameIndex: 100),   // ~8s - keep
            createToken(2, frameIndex: 125),   // ~10s - keep
            createToken(3, frameIndex: 140),   // ~11.2s - filter out
        ]
        let right: [TokenWindow] = [
            createToken(4, frameIndex: 125),   // ~10s - filter out
            createToken(5, frameIndex: 140),   // ~11.2s - keep
            createToken(6, frameIndex: 150),   // ~12s - keep
        ]

        let result = mergeByMidpointSimulation(left, right, cutoff: cutoff)

        XCTAssertEqual(result.count, 4, "Should have 2 from left + 2 from right")
        // Left tokens before cutoff, right tokens after
        XCTAssertEqual(result[0].token, 1)
        XCTAssertEqual(result[1].token, 2)
        XCTAssertEqual(result[2].token, 5)
        XCTAssertEqual(result[3].token, 6)
    }

    // MARK: - Full Merge Simulation Tests

    func testMergeWithPerfectOverlap() {
        // Chunks with overlapping tokens that match exactly
        let left: [TokenWindow] = [
            createToken(1, frameIndex: 0),
            createToken(2, frameIndex: 50),
            createToken(3, frameIndex: 100),  // Overlap region starts
            createToken(4, frameIndex: 110),
            createToken(5, frameIndex: 120),
        ]
        let right: [TokenWindow] = [
            createToken(3, frameIndex: 100),  // Matches left
            createToken(4, frameIndex: 110),  // Matches left
            createToken(5, frameIndex: 120),  // Matches left
            createToken(6, frameIndex: 130),
            createToken(7, frameIndex: 140),
        ]

        let result = mergeChunksSimulation(left, right, overlapSeconds: 2.0)

        // Should merge cleanly without duplicates
        XCTAssertGreaterThanOrEqual(result.count, 5, "Should have at least 5 unique tokens")
        XCTAssertLessThanOrEqual(result.count, 7, "Should have at most 7 tokens")
    }

    func testMergeSortedByTimestamp() {
        let left: [TokenWindow] = [
            createToken(1, frameIndex: 50),
            createToken(2, frameIndex: 100),
        ]
        let right: [TokenWindow] = [
            createToken(3, frameIndex: 150),
            createToken(4, frameIndex: 200),
        ]

        let result = mergeChunksSimulation(left, right, overlapSeconds: 0)

        // Verify sorted order
        for i in 0..<(result.count - 1) {
            XCTAssertLessThanOrEqual(
                result[i].timestamp, result[i + 1].timestamp,
                "Result should be sorted by timestamp"
            )
        }
    }

    // MARK: - Simulation Helpers

    private func mergeChunksSimulation(
        _ left: [TokenWindow],
        _ right: [TokenWindow],
        overlapSeconds: Double
    ) -> [TokenWindow] {
        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let halfOverlapWindow = overlapSeconds / 2

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        // No overlap - concatenate
        if leftEndTime <= rightStartTime {
            return left + right
        }

        // Build overlap regions
        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapSeconds else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapSeconds else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            let cutoff = (leftEndTime + rightStartTime) / 2
            return mergeByMidpointSimulation(left, right, cutoff: cutoff)
        }

        // Try contiguous pairs
        let pairs = findBestContiguousPairsSimulation(overlapLeft, overlapRight, tolerance: halfOverlapWindow)

        if pairs.isEmpty {
            let cutoff = (leftEndTime + rightStartTime) / 2
            return mergeByMidpointSimulation(left, right, cutoff: cutoff)
        }

        return mergeUsingMatchesSimulation(pairs, overlapLeft, overlapRight, left, right)
    }

    private func findBestContiguousPairsSimulation(
        _ overlapLeft: [IndexedToken],
        _ overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        var best: [(Int, Int)] = []

        for i in 0..<overlapLeft.count {
            for j in 0..<overlapRight.count {
                if tokensMatchSimulation(overlapLeft[i], overlapRight[j], tolerance: tolerance) {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j

                    while k < overlapLeft.count && l < overlapRight.count {
                        if tokensMatchSimulation(overlapLeft[k], overlapRight[l], tolerance: tolerance) {
                            current.append((k, l))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    private func findLCSSimulation(
        _ overlapLeft: [IndexedToken],
        _ overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        let leftCount = overlapLeft.count
        let rightCount = overlapRight.count

        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)

        for i in 1...leftCount {
            for j in 1...rightCount {
                if tokensMatchSimulation(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = leftCount
        var j = rightCount

        while i > 0 && j > 0 {
            if tokensMatchSimulation(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return pairs.reversed()
    }

    private func tokensMatchSimulation(_ left: IndexedToken, _ right: IndexedToken, tolerance: Double) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    private func mergeByMidpointSimulation(
        _ left: [TokenWindow],
        _ right: [TokenWindow],
        cutoff: Double
    ) -> [TokenWindow] {
        let trimmedLeft = left.filter { startTime(of: $0) <= cutoff }
        let trimmedRight = right.filter { startTime(of: $0) >= cutoff }
        return trimmedLeft + trimmedRight
    }

    private func mergeUsingMatchesSimulation(
        _ matches: [(Int, Int)],
        _ overlapLeft: [IndexedToken],
        _ overlapRight: [IndexedToken],
        _ left: [TokenWindow],
        _ right: [TokenWindow]
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let rightIndex = rightIndices[idx]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
    }
}
