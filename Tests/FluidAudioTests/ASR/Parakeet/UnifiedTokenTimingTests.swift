import Testing

@testable import FluidAudio

/// Tests for the offline batch path's emission-frame → seconds conversion
/// (`UnifiedAsrManager.transcribeWithTimings`). The conversion is pure, so
/// none of this needs the 600M parameter model loaded.
struct UnifiedTokenTimingTests {

    private let secondsPerFrame = 1280.0 / 16000.0  // 80 ms

    /// id → SentencePiece piece, with `▁` marking a word boundary.
    private let vocabulary: [Int: String] = [
        10: "\u{2581}play", 11: "back", 12: "\u{2581}the", 13: "\u{2581}note",
    ]

    private func emission(
        _ token: Int, frame: Int, confidence: Float = 0.9
    ) -> ChunkProcessor.TokenWindow {
        (token: token, timestamp: frame, confidence: confidence, duration: 0)
    }

    @Test
    func framesConvertAtEightyMilliseconds() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary)
        #expect(timings.count == 1)
        #expect(abs(timings[0].startTime - 2.96) < 0.0001)
        #expect(timings[0].tokenId == 10)
    }

    /// RNNT emits a token AT a frame with no duration, so a token's end is
    /// provisional and only clamped back when it would overrun its successor.
    /// A real pause is therefore preserved as a gap between one token's end
    /// and the next one's start, which is what pause-based segmentation reads.
    @Test
    func silenceSurvivesAsAGapBetweenTokens() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37), emission(11, frame: 38), emission(12, frame: 44)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary)
        #expect(timings.count == 3)
        // Consecutive frames: no gap, and no overlap either.
        #expect(abs(timings[0].endTime - timings[1].startTime) < 0.0001)
        // Six frames of silence between "back" and "the" stay visible as 0.4 s.
        #expect(abs((timings[2].startTime - timings[1].endTime) - 0.4) < 0.0001)
    }

    /// No token may extend past the one that follows it, or spans overlap and
    /// a pause can read as negative.
    @Test
    func noTokenOverrunsItsSuccessor() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37), emission(11, frame: 38), emission(12, frame: 44)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary)
        for (earlier, later) in zip(timings, timings.dropFirst()) {
            #expect(earlier.endTime <= later.startTime + 0.0001)
            #expect(earlier.endTime >= earlier.startTime)
        }
    }

    /// The frontier token has no successor, so it gets one frame rather than
    /// a zero-length span.
    @Test
    func theLastTokenGetsAProvisionalOneFrameEnd() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary)
        #expect(abs((timings[0].endTime - timings[0].startTime) - secondsPerFrame) < 0.0001)
    }

    /// That provisional frame must not run past the end of the audio, or a
    /// caller seeking to the last token's end lands past EOF. Offline knows the
    /// sample count, so it can bound what streaming has to leave open.
    @Test
    func theLastTokenIsClampedToTheEndOfTheClip() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary,
            clipDuration: 3.0)
        #expect(abs(timings[0].endTime - 3.0) < 0.0001)
        #expect(timings[0].endTime >= timings[0].startTime)
    }

    /// A clip longer than the last emission leaves the one-frame end alone.
    @Test
    func aLongerClipLeavesTheProvisionalEndAlone() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary,
            clipDuration: 60.0)
        #expect(abs((timings[0].endTime - timings[0].startTime) - secondsPerFrame) < 0.0001)
    }

    /// Two tokens emitted on the same frame must not produce a negative span.
    @Test
    func tokensSharingAFrameKeepNonNegativeSpans() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37), emission(11, frame: 37)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary)
        #expect(timings[0].endTime >= timings[0].startTime)
        #expect(abs(timings[0].endTime - timings[0].startTime) < 0.0001)
    }

    /// The whole point of the conversion: the result feeds `buildWordTimings`,
    /// which regroups sub-word pieces into words on the `▁` marker.
    @Test
    func outputGroupsIntoWordsWithTheSharedHelper() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [
                emission(10, frame: 37), emission(11, frame: 38),
                emission(12, frame: 44), emission(13, frame: 47),
            ],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary)
        let words = buildWordTimings(from: timings)
        #expect(words.map(\.word) == ["playback", "the", "note"])
        #expect(abs(words[0].startTime - 2.96) < 0.0001)
        #expect(abs(words[1].startTime - 3.52) < 0.0001)
    }

    /// An id the vocabulary doesn't cover is skipped rather than emitted as an
    /// empty token that would swallow the next word's boundary marker.
    @Test
    func unknownTokenIdsAreSkipped() {
        let timings = UnifiedAsrManager.tokenTimings(
            from: [emission(10, frame: 37), emission(999, frame: 40), emission(12, frame: 44)],
            secondsPerFrame: secondsPerFrame, vocabulary: vocabulary)
        #expect(timings.map(\.tokenId) == [10, 12])
    }
}
