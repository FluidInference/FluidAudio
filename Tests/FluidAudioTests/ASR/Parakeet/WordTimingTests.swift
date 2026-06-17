import Testing

@testable import FluidAudio

/// Tests for `buildWordTimings`, the pure aggregation that turns per-token
/// `TokenTiming`s (with SentencePiece `▁` boundary markers) into word-level
/// spans. Exposed for streaming word-level timestamps (issue #704), composing
/// with `StreamingUnifiedAsrManager.consumeTokenTimings()`.
struct WordTimingTests {

    private func token(
        _ piece: String, id: Int = 0, start: Double, end: Double, confidence: Float = 1.0
    ) -> TokenTiming {
        TokenTiming(token: piece, tokenId: id, startTime: start, endTime: end, confidence: confidence)
    }

    @Test
    func groupsSubwordTokensIntoWords() {
        // "▁Hello" "▁wor" "ld" -> ["Hello", "world"]
        let timings = [
            token("\u{2581}Hello", start: 0.0, end: 0.08),
            token("\u{2581}wor", start: 0.16, end: 0.24),
            token("ld", start: 0.24, end: 0.32),
        ]
        let words = buildWordTimings(from: timings)
        #expect(words.map(\.word) == ["Hello", "world"])
        #expect(words[0].startTime == 0.0)
        #expect(words[0].endTime == 0.08)
        // Second word spans from its first sub-word start to its last sub-word end.
        #expect(words[1].startTime == 0.16)
        #expect(words[1].endTime == 0.32)
    }

    @Test
    func firstWordWithoutBoundaryMarkerStillStarts() {
        // A leading token without a boundary marker still opens the first word.
        let timings = [
            token("the", start: 0.0, end: 0.08),
            token("\u{2581}cat", start: 0.08, end: 0.16),
        ]
        let words = buildWordTimings(from: timings)
        #expect(words.map(\.word) == ["the", "cat"])
    }

    @Test
    func leadingSpaceTreatedAsBoundary() {
        let timings = [
            token(" Hello", start: 0.0, end: 0.08),
            token(" world", start: 0.16, end: 0.24),
        ]
        let words = buildWordTimings(from: timings)
        #expect(words.map(\.word) == ["Hello", "world"])
    }

    @Test
    func skipsSpecialTokens() {
        let timings = [
            token("\u{2581}hi", start: 0.0, end: 0.08),
            token("<blank>", start: 0.08, end: 0.16),
            token("\u{2581}there", start: 0.16, end: 0.24),
        ]
        let words = buildWordTimings(from: timings)
        #expect(words.map(\.word) == ["hi", "there"])
    }

    @Test
    func emptyInputProducesNoWords() {
        #expect(buildWordTimings(from: []).isEmpty)
    }
}
