import XCTest

@testable import FluidAudio

final class CtcDPAlgorithmTests: XCTestCase {

    // MARK: - Synthetic Data Helpers

    /// Build a log-prob matrix [T x V] where at each specified frame, one token is
    /// "hot" (highScore) and all others are "cold" (coldScore).
    private func makeLogProbs(
        frames: Int,
        vocabSize: Int,
        hotTokens: [(frame: Int, tokenId: Int)],
        highScore: Float = -0.1,
        coldScore: Float = -10.0
    ) -> [[Float]] {
        var matrix = Array(
            repeating: Array(repeating: coldScore, count: vocabSize),
            count: frames
        )
        for (frame, token) in hotTokens where frame < frames && token < vocabSize {
            matrix[frame][token] = highScore
        }
        return matrix
    }

    // MARK: - nonWildcardCount

    func testNonWildcardCountAllRegular() {
        XCTAssertEqual(CtcDPAlgorithm.nonWildcardCount([0, 1, 2]), 3)
    }

    func testNonWildcardCountMixed() {
        let wildcard = CtcDPAlgorithm.wildcardTokenId
        XCTAssertEqual(CtcDPAlgorithm.nonWildcardCount([0, wildcard, 1]), 2)
    }

    func testNonWildcardCountAllWildcards() {
        let wildcard = CtcDPAlgorithm.wildcardTokenId
        XCTAssertEqual(CtcDPAlgorithm.nonWildcardCount([wildcard, wildcard, wildcard]), 0)
    }

    func testNonWildcardCountEmpty() {
        XCTAssertEqual(CtcDPAlgorithm.nonWildcardCount([]), 0)
    }

    // MARK: - ctcWordSpot

    func testPerfectAlignmentSingleToken() {
        // 3 frames, vocab size 5
        // Token 2 is hot at all 3 frames -> keyword [2] aligns perfectly
        let logProbs = makeLogProbs(
            frames: 3, vocabSize: 5,
            hotTokens: [(0, 2), (1, 2), (2, 2)]
        )
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [2])
        // Score should be close to highScore (-0.1) since token 2 is strongly present
        XCTAssertGreaterThan(result.score, -1.0)
    }

    func testPerfectAlignmentMultipleTokens() {
        // 6 frames, vocab size 5
        // Frames 0-1: token 0 hot, Frames 2-3: token 1 hot, Frames 4-5: token 2 hot
        let logProbs = makeLogProbs(
            frames: 6, vocabSize: 5,
            hotTokens: [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2), (5, 2)]
        )
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [0, 1, 2])
        // All tokens have strong alignment
        XCTAssertGreaterThan(result.score, -1.0)
    }

    func testNoMatchReturnsLowScore() {
        // 3 frames, token 0 is always hot, but keyword asks for token 3
        let logProbs = makeLogProbs(
            frames: 3, vocabSize: 5,
            hotTokens: [(0, 0), (1, 0), (2, 0)]
        )
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [3])
        // Token 3 always cold -> score should be very negative
        XCTAssertLessThan(result.score, -5.0)
    }

    func testEmptyKeyword() {
        let logProbs = makeLogProbs(frames: 3, vocabSize: 5, hotTokens: [])
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [])
        XCTAssertEqual(result.score, -Float.infinity)
    }

    func testEmptyLogProbs() {
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: [], keywordTokens: [0, 1])
        XCTAssertEqual(result.score, -Float.infinity)
    }

    func testSingleFrameSingleToken() {
        let logProbs = makeLogProbs(
            frames: 1, vocabSize: 3,
            hotTokens: [(0, 1)],
            highScore: -0.1
        )
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [1])
        XCTAssertEqual(result.score, -0.1, accuracy: 0.01)
    }

    func testKeywordLongerThanFrames() {
        // 2 frames but keyword needs 3 tokens -> T < N
        let logProbs = makeLogProbs(frames: 2, vocabSize: 5, hotTokens: [(0, 0), (1, 1)])
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [0, 1, 2])
        // Cannot fit keyword in frames -> should get very bad score
        XCTAssertLessThan(result.score, -5.0)
    }

    func testStartAndEndFrame() {
        // 10 frames, keyword [0, 1, 2] hot at frames 3-5
        let logProbs = makeLogProbs(
            frames: 10, vocabSize: 5,
            hotTokens: [(3, 0), (4, 1), (5, 2)]
        )
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [0, 1, 2])
        // Frames use 1-based indexing in DP (frame 3 in 0-based = t=4 in DP)
        // End frame should be near 4-6 (1-based offset of 0-based frames 3-5)
        XCTAssertGreaterThanOrEqual(result.endFrame, 3)
        XCTAssertLessThanOrEqual(result.endFrame, 7)
    }

    func testWildcardTokenMatchesAnything() {
        let wildcard = CtcDPAlgorithm.wildcardTokenId
        // keyword: [wildcard, 1, wildcard] — only token 1 needs to match
        let logProbs = makeLogProbs(
            frames: 5, vocabSize: 5,
            hotTokens: [(2, 1)],
            highScore: -0.1
        )
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [wildcard, 1, wildcard])
        // nonWildcardCount = 1, so score normalized by 1
        // Token 1 is hot at frame 2 -> score should be close to -0.1
        XCTAssertGreaterThan(result.score, -1.0)
    }

    // MARK: - ctcWordSpotConstrained

    func testConstrainedWindowBasic() {
        // 20 frames, keyword [0, 1] hot at frames 5-6
        let logProbs = makeLogProbs(
            frames: 20, vocabSize: 5,
            hotTokens: [(5, 0), (6, 1)]
        )
        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0, 1],
            searchStartFrame: 3,
            searchEndFrame: 12
        )
        XCTAssertGreaterThan(result.score, -1.0)
        // Result should be in global coordinates
        XCTAssertGreaterThanOrEqual(result.startFrame, 3)
        XCTAssertLessThanOrEqual(result.endFrame, 12)
    }

    func testConstrainedWindowMissesKeyword() {
        // Keyword hot at frames 15-16, but window only covers 0-10
        let logProbs = makeLogProbs(
            frames: 20, vocabSize: 5,
            hotTokens: [(15, 0), (16, 1)]
        )
        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0, 1],
            searchStartFrame: 0,
            searchEndFrame: 10
        )
        // Keyword is outside window -> should get poor score
        XCTAssertLessThan(result.score, -5.0)
    }

    func testConstrainedWindowClamped() {
        // Out-of-bounds window should be clamped
        let logProbs = makeLogProbs(
            frames: 5, vocabSize: 3,
            hotTokens: [(2, 0)]
        )
        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0],
            searchStartFrame: -5,
            searchEndFrame: 100
        )
        // Should work fine with clamped bounds
        XCTAssertGreaterThan(result.score, -Float.infinity)
    }

    func testConstrainedWindowTooSmall() {
        // Window has 2 frames but keyword needs 3 tokens
        let logProbs = makeLogProbs(frames: 20, vocabSize: 5, hotTokens: [])
        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0, 1, 2],
            searchStartFrame: 5,
            searchEndFrame: 7
        )
        XCTAssertEqual(result.score, -Float.infinity)
    }

    func testConstrainedEmptyWindow() {
        let logProbs = makeLogProbs(frames: 10, vocabSize: 5, hotTokens: [])
        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0],
            searchStartFrame: 5,
            searchEndFrame: 5
        )
        XCTAssertEqual(result.score, -Float.infinity)
    }

    // MARK: - ctcWordSpotMultiple

    func testMultipleEmptyKeyword() {
        let logProbs = makeLogProbs(frames: 5, vocabSize: 3, hotTokens: [])
        let results = CtcDPAlgorithm.ctcWordSpotMultiple(logProbs: logProbs, keywordTokens: [])
        XCTAssertTrue(results.isEmpty)
    }

    func testMultipleEmptyLogProbs() {
        let results = CtcDPAlgorithm.ctcWordSpotMultiple(logProbs: [], keywordTokens: [0])
        XCTAssertTrue(results.isEmpty)
    }

    func testMultipleBelowMinScore() {
        // All tokens cold -> all scores below threshold
        let logProbs = makeLogProbs(frames: 5, vocabSize: 3, hotTokens: [])
        let results = CtcDPAlgorithm.ctcWordSpotMultiple(
            logProbs: logProbs,
            keywordTokens: [0],
            minScore: -5.0
        )
        XCTAssertTrue(results.isEmpty)
    }

    func testMultipleSingleOccurrence() {
        // Token 0 hot only at frame 2 -> one detection
        let logProbs = makeLogProbs(
            frames: 10, vocabSize: 5,
            hotTokens: [(2, 0)],
            highScore: -0.1
        )
        let results = CtcDPAlgorithm.ctcWordSpotMultiple(
            logProbs: logProbs,
            keywordTokens: [0],
            minScore: -1.0
        )
        XCTAssertGreaterThanOrEqual(results.count, 1)
        if let first = results.first {
            XCTAssertGreaterThan(first.score, -1.0)
        }
    }

    // MARK: - fillDPTable (indirectly tested through ctcWordSpot)

    func testDPTableScoreMonotonicity() {
        // With a perfect match, the score at the end should be non-negative
        // (sum of hot log-probs, each ≈ -0.1, normalized by count)
        let logProbs = makeLogProbs(
            frames: 3, vocabSize: 3,
            hotTokens: [(0, 0), (1, 1), (2, 2)],
            highScore: -0.05
        )
        let result = CtcDPAlgorithm.ctcWordSpot(logProbs: logProbs, keywordTokens: [0, 1, 2])
        // Normalized score = sum(-0.05 * 3) / 3 = -0.05
        XCTAssertEqual(result.score, -0.05, accuracy: 0.01)
    }
}
