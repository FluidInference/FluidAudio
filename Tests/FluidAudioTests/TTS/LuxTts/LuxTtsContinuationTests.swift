import XCTest

@testable import FluidAudio

final class LuxTtsContinuationTests: XCTestCase {

    func testChunksStayBalancedAndBreakAtBoundaries() {
        let space = 0
        let tokens = Array(repeating: [1, 2, 3, 4, space], count: 22).flatMap { $0 }
        let chunks = LuxTtsContinuation.chunks(
            tokenIds: tokens, maxTokens: 102, boundaryTokenIds: [space])

        XCTAssertEqual(chunks.count, 2)
        XCTAssertEqual(chunks.flatMap { $0 }, tokens)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 102 })
        XCTAssertTrue(chunks.dropLast().allSatisfy { $0.last == space })
        XCTAssertLessThanOrEqual(abs(chunks[0].count - chunks[1].count), 5)
    }

    func testChunksKeepStableIssue937BoundarySinglePass() {
        XCTAssertEqual(
            LuxTtsContinuation.chunks(
                tokenIds: Array(0..<102), maxTokens: 102, boundaryTokenIds: []
            ).count,
            1)
        XCTAssertEqual(
            LuxTtsContinuation.chunks(
                tokenIds: Array(0..<106), maxTokens: 102, boundaryTokenIds: []
            ).map(\.count),
            [53, 53])
    }

    func testChunksNeverLeaveAnOversizedRemainder() {
        let tokens = Array(0..<205)
        let chunks = LuxTtsContinuation.chunks(
            tokenIds: tokens, maxTokens: 102, boundaryTokenIds: [0, 1])

        XCTAssertEqual(chunks.flatMap { $0 }, tokens)
        XCTAssertTrue(chunks.allSatisfy { !$0.isEmpty && $0.count <= 102 })
        XCTAssertEqual(chunks.map(\.count), [68, 69, 68])
    }

    func testMaxSpanTokensIsBoundedBySpanSizeTokenBucketAndFrameBudget() {
        // Fast prompt: the 102-token span size wins.
        XCTAssertEqual(
            LuxTtsContinuation.maxSpanTokens(promptFrames: 300, promptTokenCount: 100, speed: 1.0),
            102)
        // Long transcript: the 256-token encoder bucket (minus pad slot) wins.
        XCTAssertEqual(
            LuxTtsContinuation.maxSpanTokens(promptFrames: 300, promptTokenCount: 200, speed: 1.0),
            55)
        // Issue #937 prompt (100 tokens / 5 s): 468 frames ÷ 4.69 frames/token.
        XCTAssertEqual(
            LuxTtsContinuation.maxSpanTokens(promptFrames: 469, promptTokenCount: 100, speed: 1.0),
            99)
        // Slower speech needs proportionally fewer tokens per span.
        XCTAssertEqual(
            LuxTtsContinuation.maxSpanTokens(promptFrames: 469, promptTokenCount: 100, speed: 0.5),
            49)
        // Degenerate prompts fall back to the cap (the synthesizer rejects them).
        XCTAssertEqual(
            LuxTtsContinuation.maxSpanTokens(promptFrames: 0, promptTokenCount: 0, speed: 1.0),
            102)
        // Pathological ratios floor at 1; the manager rejects anything under
        // `minimumSpanTokens` instead of rendering dozens of passes.
        XCTAssertEqual(
            LuxTtsContinuation.maxSpanTokens(promptFrames: 469, promptTokenCount: 4, speed: 1.0),
            3)
    }

    func testSpanFrameBudgetFitsInsidePromptCap() {
        // A span at the budget yields (budget - 1) * hop48k samples at 48 kHz;
        // at 24 kHz that must fit under the synthesizer's prompt cap so the
        // next span's prompt is not truncated away from its transcript.
        let budget = LuxTtsConstants.continuationSpanFrameBudget
        let samples24k = (budget - 1) * LuxTtsConstants.hop48k / 2
        XCTAssertLessThanOrEqual(
            samples24k,
            Int(LuxTtsConstants.maxPromptSeconds * Double(LuxTtsConstants.melSampleRate)))
        XCTAssertLessThanOrEqual(budget, LuxTtsConstants.vocoderBuckets.max() ?? 0)
    }

    func testFitsSinglePassUsesSpanCapAndGraphLimits() {
        // Reporter's prompt: 100 tokens over 5 s (469 frames), speed 1.0.
        XCTAssertTrue(
            LuxTtsContinuation.fitsSinglePass(
                textTokenCount: 102, promptFrames: 469, promptTokenCount: 100, speed: 1.0))
        // 106 tokens would fit the graph (498 frames) but exceed the span cap.
        XCTAssertFalse(
            LuxTtsContinuation.fitsSinglePass(
                textTokenCount: 106, promptFrames: 469, promptTokenCount: 100, speed: 1.0))
        // 102 tokens at a slow prompt → 613 generated frames, past the 555 bucket.
        XCTAssertFalse(
            LuxTtsContinuation.fitsSinglePass(
                textTokenCount: 102, promptFrames: 600, promptTokenCount: 100, speed: 1.0))
        // Half speed doubles the frame estimate past the 555 bucket.
        XCTAssertFalse(
            LuxTtsContinuation.fitsSinglePass(
                textTokenCount: 102, promptFrames: 469, promptTokenCount: 100, speed: 0.5))
        // Prompt transcript + text must leave the pad slot in the 256 bucket.
        XCTAssertFalse(
            LuxTtsContinuation.fitsSinglePass(
                textTokenCount: 60, promptFrames: 300, promptTokenCount: 196, speed: 1.0))
        XCTAssertTrue(
            LuxTtsContinuation.fitsSinglePass(
                textTokenCount: 59, promptFrames: 300, promptTokenCount: 196, speed: 1.0))
    }

    func testTextPauseAllowanceCountsSilentBreaksOnly() {
        XCTAssertEqual(LuxTtsContinuation.textPauseAllowance("He paused... then said \"no\"."), 3)
        XCTAssertEqual(LuxTtsContinuation.textPauseAllowance("Wait… (really)"), 3)
        XCTAssertEqual(LuxTtsContinuation.textPauseAllowance("don't, won't; can't."), 0)
    }

    func testExpectedPauseCountIgnoresTrailingBoundaryTokens() {
        let space = 0
        let comma = 1
        let period = 2
        let pauses: Set<Int> = [comma, period]
        let boundaries: Set<Int> = [space, comma, period]
        XCTAssertEqual(
            LuxTtsContinuation.expectedPauseCount(
                in: [5, 6, comma, space, 7, 8, period, space],
                pauseTokenIds: pauses, boundaryTokenIds: boundaries),
            1)
        XCTAssertEqual(
            LuxTtsContinuation.expectedPauseCount(
                in: [5, 6, 7], pauseTokenIds: pauses, boundaryTokenIds: boundaries),
            0)
    }

    func testInnerPauseCountFindsOnlyMidSpeechSilences() {
        let sampleRate = 1_000
        let speech = Array(repeating: Float(0.5), count: 300)
        let lead = Array(repeating: Float.zero, count: 200)
        let shortClosure = Array(repeating: Float.zero, count: 50)
        let pause = Array(repeating: Float.zero, count: 150)
        // The vocoder's onset click: loud but far shorter than sustained speech.
        let click = Array(repeating: Float(1), count: 20)

        XCTAssertEqual(
            LuxTtsContinuation.innerPauseCount(lead + speech + lead, sampleRate: sampleRate), 0)
        XCTAssertEqual(
            LuxTtsContinuation.innerPauseCount(click + lead + speech + lead, sampleRate: sampleRate),
            0)
        XCTAssertEqual(
            LuxTtsContinuation.innerPauseCount(
                lead + speech + shortClosure + speech + lead, sampleRate: sampleRate),
            0)
        XCTAssertEqual(
            LuxTtsContinuation.innerPauseCount(
                lead + speech + pause + speech + shortClosure + speech + pause + speech,
                sampleRate: sampleRate),
            2)
        XCTAssertEqual(LuxTtsContinuation.innerPauseCount([], sampleRate: sampleRate), 0)
    }

    func testCrossfadePreservesEdgesAndExpectedLength() {
        var output: [Float] = [1, 1, 1, 1]
        LuxTtsContinuation.appendWithCrossfade(
            [0, 0, 0, 0], to: &output, crossfadeSamples: 3)

        XCTAssertEqual(output.count, 5)
        XCTAssertEqual(output.first, 1)
        XCTAssertEqual(output.last, 0)
        XCTAssertEqual(output[1], 1)
        XCTAssertEqual(output[2], 0.5)
        XCTAssertEqual(output[3], 0)
    }

    func testLeadingPaddingTrimKeepsOnsetPreroll() {
        let samples =
            Array(repeating: Float.zero, count: 50)
            + Array(repeating: Float(1), count: 50)

        let trimmed = LuxTtsContinuation.trimmingLeadingPadding(samples, sampleRate: 1_000)

        XCTAssertEqual(trimmed.count, 80)
        XCTAssertEqual(Array(trimmed.prefix(30)), Array(repeating: Float.zero, count: 30))
        XCTAssertEqual(trimmed[30], 1)
    }

    func testLeadingPaddingTrimSkipsOnsetClick() {
        let click = Array(repeating: Float(1), count: 20)
        let padding = Array(repeating: Float.zero, count: 200)
        let speech = Array(repeating: Float(0.5), count: 100)

        let trimmed = LuxTtsContinuation.trimmingLeadingPadding(
            click + padding + speech, sampleRate: 1_000)

        // 30 ms preroll before the sustained onset at sample 220.
        XCTAssertEqual(trimmed.count, 130)
        XCTAssertEqual(trimmed[30], 0.5)
    }

    func testTrailingPaddingTrimKeepsPostroll() {
        let speech = Array(repeating: Float(1), count: 50)
        let padding = Array(repeating: Float.zero, count: 200)

        let trimmed = LuxTtsContinuation.trimmingTrailingPadding(speech + padding, sampleRate: 1_000)

        XCTAssertEqual(trimmed.count, 80)
        XCTAssertEqual(trimmed[49], 1)
        XCTAssertEqual(trimmed[50], 0)
    }

    func testEndsWithPausePunctuationIgnoresTrailingSpaces() {
        let space = 0
        let comma = 1
        XCTAssertTrue(
            LuxTtsContinuation.endsWithPausePunctuation(
                [5, 6, comma, space], pauseTokenIds: [comma], boundaryTokenIds: [space, comma]))
        XCTAssertFalse(
            LuxTtsContinuation.endsWithPausePunctuation(
                [5, comma, 6, space], pauseTokenIds: [comma], boundaryTokenIds: [space, comma]))
    }
}
