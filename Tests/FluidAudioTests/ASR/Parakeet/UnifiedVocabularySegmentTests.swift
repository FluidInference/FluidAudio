import Testing

@testable import FluidAudio

/// Tests for the streaming unified vocabulary-boosting segmentation rules
/// (`StreamingUnifiedAsrManager.vocabSegmentCut` / `segmentLocalTimings`).
/// Both are pure, so none of this needs models loaded.
struct UnifiedVocabularySegmentTests {

    private func timing(
        _ token: String, start: Double, end: Double? = nil, id: Int = 0
    ) -> TokenTiming {
        TokenTiming(
            token: token, tokenId: id, startTime: start,
            endTime: end ?? start + 0.08, confidence: 0.9
        )
    }

    // MARK: - vocabSegmentCut

    @Test
    func emptyTimingsYieldNoCut() {
        #expect(StreamingUnifiedAsrManager.vocabSegmentCut(timings: [], force: false) == nil)
        #expect(StreamingUnifiedAsrManager.vocabSegmentCut(timings: [], force: true) == nil)
    }

    @Test
    func forceCutsEverything() {
        let timings = [timing(" hel", start: 0.0), timing("lo", start: 0.1)]
        #expect(StreamingUnifiedAsrManager.vocabSegmentCut(timings: timings, force: true) == 2)
    }

    @Test
    func nonFinalCutHoldsFrontierWordBack() {
        // " play" "back" | " the" — frontier word " the" must stay pending.
        let timings = [
            timing(" play", start: 0.0),
            timing("back", start: 0.1),
            timing(" the", start: 0.3),
        ]
        #expect(StreamingUnifiedAsrManager.vocabSegmentCut(timings: timings, force: false) == 2)
    }

    @Test
    func cutNeverSplitsAWord() {
        // Frontier word spans two sub-word tokens; the cut lands before its
        // first token, not between them.
        let timings = [
            timing(" romvim", start: 0.0),
            timing(" za", start: 0.4, id: 1),  // separate word
            timing("wa", start: 0.5, id: 2),  // continuation of " za"
        ]
        let cut = StreamingUnifiedAsrManager.vocabSegmentCut(timings: timings, force: false)
        #expect(cut == 1)
    }

    @Test
    func singlePendingWordYieldsNoCut() {
        // All tokens belong to one word — nothing can be released yet.
        let timings = [timing(" hel", start: 0.0), timing("lo", start: 0.1)]
        #expect(StreamingUnifiedAsrManager.vocabSegmentCut(timings: timings, force: false) == nil)
    }

    @Test
    func noWordBoundaryYieldsNoCut() {
        // Degenerate stream of continuation tokens only.
        let timings = [timing("lo", start: 0.0), timing("re", start: 0.1)]
        #expect(StreamingUnifiedAsrManager.vocabSegmentCut(timings: timings, force: false) == nil)
    }

    // MARK: - segmentLocalTimings

    @Test
    func rebasesOntoSegmentClock() {
        let rebased = StreamingUnifiedAsrManager.segmentLocalTimings(
            [timing(" the", start: 20.0, end: 20.08)], segmentStartSeconds: 18.5
        )
        #expect(rebased.count == 1)
        #expect(abs(rebased[0].startTime - 1.5) < 0.0001)
        #expect(abs(rebased[0].endTime - 1.58) < 0.0001)
        #expect(rebased[0].token == " the")
    }

    @Test
    func dropsTokensBeforeRetainedAudio() {
        // Tokens decoded before boosting was configured have no retained
        // audio behind them and must not reach the rescorer.
        let rebased = StreamingUnifiedAsrManager.segmentLocalTimings(
            [
                timing(" old", start: 3.0),
                timing(" new", start: 10.0),
            ],
            segmentStartSeconds: 5.0
        )
        #expect(rebased.count == 1)
        #expect(rebased[0].token == " new")
        #expect(abs(rebased[0].startTime - 5.0) < 0.0001)
    }

    // MARK: - itnDefaultConfig

    @Test
    func itnDefaultConfigSetsRescueSimilarityFloors() {
        // ITN engines need the #702 floors; without them the spotter-anchored
        // rescue replaces digit spans at garbage similarity (see session docs).
        let config = VocabularyBoostingSession.itnDefaultConfig
        #expect(config.spotterRescueMinSimilarity == 0.30)
        #expect(config.spotterRescueMultiWordMinSimilarity == 0.50)
        #expect(config.spotterRescueEnabled)
    }

    @Test
    func preservesTokenIdentityAndConfidence() {
        let original = TokenTiming(
            token: " word", tokenId: 42, startTime: 7.0, endTime: 7.2, confidence: 0.75
        )
        let rebased = StreamingUnifiedAsrManager.segmentLocalTimings(
            [original], segmentStartSeconds: 7.0
        )
        #expect(rebased[0].tokenId == 42)
        #expect(rebased[0].confidence == 0.75)
        #expect(abs(rebased[0].startTime) < 0.0001)
    }
}
