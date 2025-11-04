import XCTest

@testable import FluidAudio

final class TemporalAgreementTests: XCTestCase {
    var processor: LocalAgreementStreamingProcessor!
    var config: ASRConfig!

    override func setUp() {
        super.setUp()
        config = ASRConfig()
        processor = LocalAgreementStreamingProcessor(
            decoder: TdtDecoderV3(config: config),
            config: config
        )
    }

    override func tearDown() {
        processor = nil
        config = nil
        super.tearDown()
    }

    // MARK: - Temporal Comparison Tests

    /// Test that first chunk produces no confirmed tokens (nothing to compare against).
    func testFirstChunkAllProvisional() {
        let tokens = [101, 102, 103, 104, 105]
        let timestamps = [0, 1, 2, 3, 4]
        let confidences: [Float] = [0.9, 0.95, 0.85, 0.92, 0.88]

        // Simulate first chunk processing
        let (confirmed, provisional) = testProcessChunkSequence(
            chunks: [(tokens, timestamps, confidences)],
            threshold: 0.7
        )

        XCTAssertEqual(confirmed, [], "First chunk should produce no confirmed tokens")
        XCTAssertEqual(provisional, tokens, "First chunk should hold all tokens provisional")
    }

    /// Test temporal agreement: matching prefix between consecutive chunks.
    func testTemporalAgreementWithMatchingPrefix() {
        let chunk1Tokens = [101, 102, 103, 104, 105]
        let chunk1Confidences: [Float] = [0.9, 0.95, 0.85, 0.92, 0.88]

        let chunk2Tokens = [101, 102, 103, 200, 205]  // First 3 match, then diverge
        let chunk2Confidences: [Float] = [0.9, 0.95, 0.85, 0.75, 0.80]

        let (confirmed, provisional) = testProcessChunkSequence(
            chunks: [
                (chunk1Tokens, Array(0..<chunk1Tokens.count), chunk1Confidences),
                (chunk2Tokens, Array(0..<chunk2Tokens.count), chunk2Confidences),
            ],
            threshold: 0.7
        )

        // Second chunk confirms first 3 tokens from chunk 1
        XCTAssertEqual(confirmed.count, 3, "Should confirm first 3 matching tokens")
        XCTAssertEqual(confirmed, [101, 102, 103], "Confirmed tokens should match")
        // Chunk 2's tokens after the match become provisional
        XCTAssertEqual(provisional, [200, 205], "Remainder should be provisional from chunk 2")
    }

    /// Test temporal agreement with complete match between consecutive chunks.
    func testTemporalAgreementCompleteMatch() {
        let chunk1Tokens = [101, 102, 103]
        let chunk1Confidences: [Float] = [0.9, 0.95, 0.85]

        let chunk2Tokens = [101, 102, 103, 104, 105]  // Extends previous
        let chunk2Confidences: [Float] = [0.9, 0.95, 0.85, 0.92, 0.88]

        let (confirmed, provisional) = testProcessChunkSequence(
            chunks: [
                (chunk1Tokens, Array(0..<chunk1Tokens.count), chunk1Confidences),
                (chunk2Tokens, Array(0..<chunk2Tokens.count), chunk2Confidences),
            ],
            threshold: 0.7
        )

        // All of chunk 1 is confirmed in chunk 2
        XCTAssertEqual(confirmed.count, 3, "All chunk 1 tokens should be confirmed")
        XCTAssertEqual(confirmed, [101, 102, 103], "Confirmed tokens should match chunk 1")
        // Chunk 2's new tokens are provisional
        XCTAssertEqual(provisional, [104, 105], "New tokens should be provisional")
    }

    /// Test temporal agreement with early divergence (no matching prefix).
    func testTemporalAgreementEarlyDivergence() {
        let chunk1Tokens = [101, 102, 103]
        let chunk1Confidences: [Float] = [0.9, 0.95, 0.85]

        let chunk2Tokens = [200, 202, 203]  // Completely different
        let chunk2Confidences: [Float] = [0.9, 0.95, 0.85]

        let (confirmed, provisional) = testProcessChunkSequence(
            chunks: [
                (chunk1Tokens, Array(0..<chunk1Tokens.count), chunk1Confidences),
                (chunk2Tokens, Array(0..<chunk2Tokens.count), chunk2Confidences),
            ],
            threshold: 0.7
        )

        // No agreement - nothing confirmed
        XCTAssertTrue(confirmed.isEmpty, "No tokens should be confirmed on divergence")
        XCTAssertEqual(provisional, chunk2Tokens, "All chunk 2 tokens held provisional")
    }

    /// Test confidence threshold enforcement.
    func testTemporalAgreementWithLowConfidence() {
        let chunk1Tokens = [101, 102, 103, 104]
        let chunk1Confidences: [Float] = [0.9, 0.65, 0.95, 0.88]  // Second below 0.7

        let chunk2Tokens = [101, 102, 103, 104]  // Same tokens
        let chunk2Confidences: [Float] = [0.9, 0.65, 0.95, 0.88]

        let (confirmed, provisional) = testProcessChunkSequence(
            chunks: [
                (chunk1Tokens, Array(0..<chunk1Tokens.count), chunk1Confidences),
                (chunk2Tokens, Array(0..<chunk2Tokens.count), chunk2Confidences),
            ],
            threshold: 0.7
        )

        // Should stop at low-confidence token even with matching output
        XCTAssertEqual(confirmed.count, 1, "Should stop at low-confidence token")
        XCTAssertEqual(confirmed, [101], "Only first high-confidence token confirmed")
    }

    /// Test three-chunk progression to verify token promotion.
    func testThreeChunkProgression() {
        let chunk1Tokens = [101, 102, 103]
        let chunk1Confidences: [Float] = [0.9, 0.95, 0.85]

        let chunk2Tokens = [101, 102, 103, 104, 105]
        let chunk2Confidences: [Float] = [0.9, 0.95, 0.85, 0.92, 0.88]

        let chunk3Tokens = [101, 102, 103, 104, 105, 106, 107]
        let chunk3Confidences: [Float] = [0.9, 0.95, 0.85, 0.92, 0.88, 0.90, 0.87]

        var allConfirmed: [Int] = []

        // Process chunk 1 - all provisional
        let (confirmed1, provisional1) = testProcessChunkSequence(
            chunks: [(chunk1Tokens, Array(0..<chunk1Tokens.count), chunk1Confidences)],
            threshold: 0.7
        )
        allConfirmed.append(contentsOf: confirmed1)

        // Process chunk 2 - confirms chunk 1, chunk 2 tokens provisional
        let (confirmed2, provisional2) = testProcessChunkSequence(
            chunks: [
                (chunk1Tokens, Array(0..<chunk1Tokens.count), chunk1Confidences),
                (chunk2Tokens, Array(0..<chunk2Tokens.count), chunk2Confidences),
            ],
            threshold: 0.7
        )
        XCTAssertEqual(confirmed2.count, 3, "Chunk 2 should confirm all of chunk 1")

        // Process chunk 3 - confirms chunk 2's matched tokens, chunk 3 tokens provisional
        let (confirmed3, provisional3) = testProcessChunkSequence(
            chunks: [
                (chunk1Tokens, Array(0..<chunk1Tokens.count), chunk1Confidences),
                (chunk2Tokens, Array(0..<chunk2Tokens.count), chunk2Confidences),
                (chunk3Tokens, Array(0..<chunk3Tokens.count), chunk3Confidences),
            ],
            threshold: 0.7
        )
        XCTAssertEqual(confirmed3.count, 5, "Chunk 3 should confirm chunk 2 prefix")
    }

    // MARK: - Helper Methods

    /// Simulate processing multiple chunks sequentially to test temporal agreement.
    /// Returns the final confirmed and provisional tokens.
    private func testProcessChunkSequence(
        chunks: [(tokens: [Int], timestamps: [Int], confidences: [Float])],
        threshold: Float
    ) -> (confirmed: [Int], provisional: [Int]) {
        var previousTokens: [Int]?
        var previousConfidences: [Float]?
        var lastConfirmed: [Int] = []
        var lastProvisional: [Int] = []

        for (tokens, _, confidences) in chunks {
            if let previous = previousTokens {
                // Find common prefix (temporal agreement)
                let confirmed = findCommonPrefix(
                    previousTokens: previous,
                    currentTokens: tokens,
                    previousConfidences: previousConfidences ?? [],
                    currentConfidences: confidences,
                    threshold: threshold
                )

                let confirmedCount = confirmed.count
                let provisional = Array(tokens.dropFirst(confirmedCount))

                lastConfirmed = confirmed
                lastProvisional = provisional
            } else {
                // First chunk - all provisional
                lastConfirmed = []
                lastProvisional = tokens
            }

            previousTokens = tokens
            previousConfidences = confidences
        }

        return (lastConfirmed, lastProvisional)
    }

    /// Find longest common prefix with confidence checking.
    private func findCommonPrefix(
        previousTokens: [Int],
        currentTokens: [Int],
        previousConfidences: [Float],
        currentConfidences: [Float],
        threshold: Float
    ) -> [Int] {
        guard !previousTokens.isEmpty && !currentTokens.isEmpty else {
            return []
        }

        var matchLength = 0
        let minLength = min(previousTokens.count, currentTokens.count)

        for i in 0..<minLength {
            let prevToken = previousTokens[i]
            let currToken = currentTokens[i]
            let prevConf = previousConfidences[i]
            let currConf = currentConfidences[i]

            let tokensMatch = prevToken == currToken
            let bothAboveThreshold = prevConf >= threshold && currConf >= threshold

            if tokensMatch && bothAboveThreshold {
                matchLength += 1
            } else {
                break
            }
        }

        return Array(previousTokens.prefix(matchLength))
    }
}
