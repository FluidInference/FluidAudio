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

    // MARK: - fillDPTable (indirectly tested through ctcWordSpotConstrained)

    func testDPTableScoreMonotonicity() {
        // With a perfect match, the score at the end should be non-negative
        // (sum of hot log-probs, each ≈ -0.1, normalized by count)
        let logProbs = makeLogProbs(
            frames: 3, vocabSize: 3,
            hotTokens: [(0, 0), (1, 1), (2, 2)],
            highScore: -0.05
        )
        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0, 1, 2],
            searchStartFrame: 0,
            searchEndFrame: logProbs.count
        )
        // Normalized score = sum(-0.05 * 3) / 3 = -0.05
        XCTAssertEqual(result.score, -0.05, accuracy: 0.01)
    }

    // MARK: - Blank-aware DP behavior (arXiv:2406.07096)

    /// Build a per-frame log-prob row by exact assignment (one token = highScore,
    /// blank = blankScore, everything else = coldScore). Lets tests reason about
    /// blank emission cost explicitly.
    private func makeFrame(
        vocabSize: Int,
        hotToken: Int?,
        highScore: Float,
        blankId: Int,
        blankScore: Float,
        coldScore: Float
    ) -> [Float] {
        var row = [Float](repeating: coldScore, count: vocabSize)
        if blankId < vocabSize { row[blankId] = blankScore }
        if let h = hotToken, h < vocabSize { row[h] = highScore }
        return row
    }

    /// When the keyword is two tokens and the audio shows token0 then a long
    /// blank stretch then token1, the per-token average score must reflect the
    /// blank emission costs along the stay path. Previously the DP added 0 for
    /// stay frames, which hid the blank cost.
    func testBlankEmissionCostIsAccumulated() {
        // 5 frames, vocab = 3 token IDs + blank at 3.
        // Frame 0: token 0 hot. Frames 1-3: blank hot. Frame 4: token 1 hot.
        let blankId = 3
        let vocabSize = 4
        let hi: Float = -0.1
        let blank: Float = -0.5
        let cold: Float = -10.0

        let logProbs: [[Float]] = [
            makeFrame(
                vocabSize: vocabSize, hotToken: 0, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold),
            makeFrame(
                vocabSize: vocabSize, hotToken: nil, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold
            ),
            makeFrame(
                vocabSize: vocabSize, hotToken: nil, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold
            ),
            makeFrame(
                vocabSize: vocabSize, hotToken: nil, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold
            ),
            makeFrame(
                vocabSize: vocabSize, hotToken: 1, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold),
        ]

        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0, 1],
            searchStartFrame: 0,
            searchEndFrame: logProbs.count,
            blankId: blankId
        )

        // The best path emits token0 at frame 0, three blanks at frames 1-3,
        // and token1 at frame 4. Raw = -0.1 -0.5 -0.5 -0.5 -0.1 = -1.7.
        // Per-token normalization (N = 2) → -0.85.
        XCTAssertEqual(result.score, -0.85, accuracy: 0.01)
    }

    /// CTC requires a blank between identical adjacent tokens. The previous DP
    /// allowed `t t → t` directly, collapsing repeats and inflating scores.
    /// With the blank-aware DP, a keyword with repeated tokens scores noticeably
    /// worse on audio that lacks the intervening blank than on audio that has it.
    func testRepeatedTokensRequireInterveningBlank() {
        let blankId = 2
        let vocabSize = 3
        let hi: Float = -0.1
        let blank: Float = -0.5
        let cold: Float = -10.0

        // Two-frame audio with token 0 hot at both frames (no blank between).
        let noBlank: [[Float]] = [
            makeFrame(
                vocabSize: vocabSize, hotToken: 0, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold),
            makeFrame(
                vocabSize: vocabSize, hotToken: 0, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold),
        ]

        // Three-frame audio with token0 / blank / token0 (proper CTC shape).
        let withBlank: [[Float]] = [
            makeFrame(
                vocabSize: vocabSize, hotToken: 0, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold),
            makeFrame(
                vocabSize: vocabSize, hotToken: nil, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold
            ),
            makeFrame(
                vocabSize: vocabSize, hotToken: 0, highScore: hi, blankId: blankId, blankScore: blank, coldScore: cold),
        ]

        let resNoBlank = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: noBlank,
            keywordTokens: [0, 0],
            searchStartFrame: 0,
            searchEndFrame: noBlank.count,
            blankId: blankId
        )
        let resWithBlank = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: withBlank,
            keywordTokens: [0, 0],
            searchStartFrame: 0,
            searchEndFrame: withBlank.count,
            blankId: blankId
        )

        // The two-frame "no blank" alignment is forbidden, so the DP has to
        // burn a cold token to bridge → score drops well below the
        // properly-spaced version.
        XCTAssertGreaterThan(resWithBlank.score, resNoBlank.score + 1.0)
    }

    /// Wildcards must remain free-cost matches even with blank-aware DP.
    func testWildcardStillFreeCost() {
        let wildcard = CtcDPAlgorithm.wildcardTokenId
        let blankId = 3
        let vocabSize = 4
        let hi: Float = -0.1
        let cold: Float = -10.0

        // 3 frames: token0 hot, blank hot, token2 hot. Wildcard sits between
        // tokens 0 and 2 in the keyword and should match the middle frame
        // for free.
        let logProbs: [[Float]] = [
            makeFrame(
                vocabSize: vocabSize, hotToken: 0, highScore: hi, blankId: blankId, blankScore: cold, coldScore: cold),
            makeFrame(
                vocabSize: vocabSize, hotToken: nil, highScore: hi, blankId: blankId, blankScore: hi, coldScore: cold),
            makeFrame(
                vocabSize: vocabSize, hotToken: 2, highScore: hi, blankId: blankId, blankScore: cold, coldScore: cold),
        ]

        let result = CtcDPAlgorithm.ctcWordSpotConstrained(
            logProbs: logProbs,
            keywordTokens: [0, wildcard, 2],
            searchStartFrame: 0,
            searchEndFrame: logProbs.count,
            blankId: blankId
        )

        // Raw = -0.1 + 0 (wildcard) + -0.1. Normalization counts only the
        // two non-wildcard tokens → -0.1 per token.
        XCTAssertEqual(result.score, -0.1, accuracy: 0.05)
    }
}
