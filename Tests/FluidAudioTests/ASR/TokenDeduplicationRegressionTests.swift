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

    // MARK: - Issue #787: Temporal gating of prefix-collision dedup

    /// Token IDs from issue #787: two different Russian words that share the
    /// leading subword prefix " тра" + "ран" (= " тран").
    /// 6841 = " тра" (leading space / word-start marker), 2394 = "ран".
    private static let sharedPrefixA = 6841  // " тра"
    private static let sharedPrefixB = 2394  // "ран"

    /// The earlier word (трансформацию) sits in the tail of the accumulated
    /// history; the later, different word (транскрибируется) opens the new chunk.
    private var previousWithEarlierWord: [Int] {
        [10, 11, 12, Self.sharedPrefixA, Self.sharedPrefixB, 9001, 13, 14, 15]
    }
    private var currentWithLaterWord: [Int] {
        [Self.sharedPrefixA, Self.sharedPrefixB, 9002, 20, 21]
    }

    /// Without timestamps the legacy id-only matcher still strips the shared
    /// prefix — this documents the #787 bug and pins the fallback behavior.
    func testDedup_787_NoTimestamps_LegacyStripsSharedPrefix() {
        let asrManager = AsrManager()
        let (deduped, removed) = asrManager.removeDuplicateTokenSequence(
            previous: previousWithEarlierWord,
            current: currentWithLaterWord
        )
        XCTAssertEqual(deduped, [9002, 20, 21], "Legacy id-only path removes the shared prefix")
        XCTAssertEqual(removed, 2)
    }

    /// With global timestamps that put the two occurrences ~10s apart, the shared
    /// prefix must NOT be treated as a duplicate — the later word stays intact.
    func testDedup_787_FarApartTimestamps_KeepsSharedPrefix() {
        let asrManager = AsrManager()
        // 6841@frame 3, 2394@frame 4 in the earlier word.
        let previousTs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        // Same token IDs but ~10.4s (130 frames) later — a different utterance.
        let currentTs = [130, 131, 132, 133, 134]

        let (deduped, removed) = asrManager.removeDuplicateTokenSequence(
            previous: previousWithEarlierWord,
            current: currentWithLaterWord,
            previousTimestamps: previousTs,
            currentTimestamps: currentTs
        )
        XCTAssertEqual(
            deduped, currentWithLaterWord,
            "Temporally distant prefix collision must not be deduped (#787)")
        XCTAssertEqual(removed, 0)
    }

    /// A genuine chunk-boundary duplicate (same acoustic moment, near-identical
    /// global timestamps) must still be deduped when timestamps are supplied.
    func testDedup_787_CloseTimestamps_StillDedupes() {
        let asrManager = AsrManager()
        let previousTs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        // Same region re-decoded in the overlapping window: a few frames of jitter.
        let currentTs = [3, 4, 5, 6, 7]

        let (deduped, removed) = asrManager.removeDuplicateTokenSequence(
            previous: previousWithEarlierWord,
            current: currentWithLaterWord,
            previousTimestamps: previousTs,
            currentTimestamps: currentTs
        )
        XCTAssertEqual(deduped, [9002, 20, 21], "Real boundary overlap should still be deduped")
        XCTAssertEqual(removed, 2)
    }

    /// A gap just outside `frameTolerance` is rejected; just inside is accepted.
    func testDedup_787_ToleranceBoundary() {
        let asrManager = AsrManager()
        let previousTs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        let tolerance = ASRConstants.duplicateFrameTolerance

        // prefix at prev frame 3; current just beyond tolerance -> keep
        let justOutside = [3 + tolerance + 1, 3 + tolerance + 2, 100, 101, 102]
        let (dedupedOut, removedOut) = asrManager.removeDuplicateTokenSequence(
            previous: previousWithEarlierWord, current: currentWithLaterWord,
            previousTimestamps: previousTs, currentTimestamps: justOutside)
        XCTAssertEqual(removedOut, 0, "Gap beyond tolerance is not a duplicate")
        XCTAssertEqual(dedupedOut, currentWithLaterWord)

        // current just within tolerance -> dedup
        let justInside = [3 + tolerance, 4 + tolerance, 100, 101, 102]
        let (dedupedIn, removedIn) = asrManager.removeDuplicateTokenSequence(
            previous: previousWithEarlierWord, current: currentWithLaterWord,
            previousTimestamps: previousTs, currentTimestamps: justInside)
        XCTAssertEqual(removedIn, 2, "Gap within tolerance is a duplicate")
        XCTAssertEqual(dedupedIn, [9002, 20, 21])
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

    // MARK: - Issue #855: last-window overlap re-decode must be stripped

    /// Captured from the issue #855 repro (repetitive speech, final flush window
    /// re-decoded from frame 0): the previous tail ends "...would go for where we
    /// would." and the re-decode repeats that run 1-3 frames later before the new
    /// words (" remove filler words."). The temporally-gated dedup must strip the
    /// re-decoded overlap (including the partial leading token) and keep the new
    /// trailing words.
    func testDedup_855_LastWindowRedecodeStripped() {
        let asrManager = AsrManager()
        let previous = [
            4223, 6882, 317, 910, 3463, 6314, 1316, 7950, 283, 7877, 1974, 4223, 1455, 509, 6843, 750, 4223, 7883,
        ]
        let previousTs = [100, 103, 105, 106, 108, 111, 113, 115, 117, 120, 122, 124, 126, 128, 130, 131, 134, 137]
        // 5831 is a partial word at the window edge; 4223...4223 re-decodes the
        // previous tail; 4942... are the genuinely new trailing words.
        let current = [5831, 4223, 1455, 509, 6843, 750, 4223, 4942, 1337, 309, 6312, 4128, 7870, 7883]
        let currentTs = [122, 125, 127, 130, 133, 135, 137, 139, 141, 144, 146, 150, 152, 154]

        let (deduped, removed) = asrManager.removeDuplicateTokenSequence(
            previous: previous,
            current: current,
            previousTimestamps: previousTs,
            currentTimestamps: currentTs
        )
        XCTAssertEqual(
            deduped, [4942, 1337, 309, 6312, 4128, 7870, 7883],
            "Re-decoded overlap must be stripped; trailing words must survive")
        XCTAssertEqual(removed, 7)
    }

    /// Same trace with the timestamps 10s apart: the identical token runs are then
    /// genuine repetition, not a seam duplicate, and must be kept (gate holds).
    func testDedup_855_FarApartRepetitionKept() {
        let asrManager = AsrManager()
        let previous = [1974, 4223, 1455, 509, 6843, 750, 4223]
        let previousTs = [10, 12, 14, 16, 18, 20, 22]
        let current = [1974, 4223, 1455, 509, 6843, 750, 4223, 4942]
        let currentTs = [145, 147, 149, 151, 153, 155, 157, 159]

        let (deduped, removed) = asrManager.removeDuplicateTokenSequence(
            previous: previous,
            current: current,
            previousTimestamps: previousTs,
            currentTimestamps: currentTs
        )
        XCTAssertEqual(deduped, current, "Repetition ~10s later is not a seam duplicate")
        XCTAssertEqual(removed, 0)
    }

    // MARK: - Issue #855: final-window re-decode plan (decoder-entry behavior)

    /// The plan drives the production decoder entry: frame-0 re-decode plus an
    /// emission cutoff. Removing the production change removes this helper, so
    /// these tests are coupled to the fix itself, not just to dedup.
    func testRedecodePlan_LastStreamingChunk() {
        let plan = AsrManager.lastChunkRedecodePlan(
            isLastChunk: true,
            previousTokens: [1, 2, 3],
            previousTokenTimestamps: [130, 134, 137],
            globalFrameOffset: 112
        )
        XCTAssertEqual(plan.initialTimeIndexOverride, 0, "Final window must re-decode from frame 0")
        XCTAssertEqual(
            plan.emitTokensAfterFrame,
            137 - 112 - AsrManager.redecodeEmissionJitterFrames,
            "Cutoff = last emitted frame in window-local space minus the jitter margin")
    }

    func testRedecodePlan_CutoffClampedToZero() {
        let plan = AsrManager.lastChunkRedecodePlan(
            isLastChunk: true,
            previousTokens: [1],
            previousTokenTimestamps: [3],
            globalFrameOffset: 112
        )
        XCTAssertEqual(plan.initialTimeIndexOverride, 0)
        XCTAssertEqual(plan.emitTokensAfterFrame, 0, "Cutoff before the window start suppresses nothing")
    }

    func testRedecodePlan_InactiveOutsideFinalStreamingWindow() {
        let interior = AsrManager.lastChunkRedecodePlan(
            isLastChunk: false,
            previousTokens: [1, 2],
            previousTokenTimestamps: [10, 20],
            globalFrameOffset: 0
        )
        XCTAssertNil(interior.initialTimeIndexOverride)
        XCTAssertNil(interior.emitTokensAfterFrame)

        let noTimestamps = AsrManager.lastChunkRedecodePlan(
            isLastChunk: true,
            previousTokens: [1, 2],
            previousTokenTimestamps: nil,
            globalFrameOffset: 0
        )
        XCTAssertNil(noTimestamps.initialTimeIndexOverride, "Batch callers (no timestamps) keep legacy navigation")

        let firstWindow = AsrManager.lastChunkRedecodePlan(
            isLastChunk: true,
            previousTokens: [],
            previousTokenTimestamps: [],
            globalFrameOffset: 0
        )
        XCTAssertNil(firstWindow.initialTimeIndexOverride, "Single-window streams have nothing to re-decode")
    }

    // MARK: - Issue #897: final-window seam reconciliation

    /// Clip 03 of the #855 fixtures: window 1 ends `▁and`@153 `▁an`@156 (the
    /// fragment of "analyzing" cut by the window edge); the re-decoded final
    /// window emits `,`@152 `▁and`@155 `▁anal`@159 `y`@162 … The cutoff must
    /// anchor at the last word start, the fragment must go, and the seam
    /// comma plus the re-emitted `and` must not survive as duplicates.
    func testRedecodePlan_CutoffAnchorsAtLastWordStart() {
        let plan = AsrManager.lastChunkRedecodePlan(
            isLastChunk: true,
            previousTokens: [10, 11, 12, 13],
            previousTokenTimestamps: [147, 149, 153, 156],
            globalFrameOffset: 112,
            lastWordStartFrame: 156
        )
        XCTAssertEqual(plan.emitTokensAfterFrame, 156 - 112 - AsrManager.redecodeEmissionJitterFrames)
        // A multi-token last word anchors at its first token, not its last.
        let multi = AsrManager.lastChunkRedecodePlan(
            isLastChunk: true,
            previousTokens: [10, 11, 12],
            previousTokenTimestamps: [140, 150, 160],
            globalFrameOffset: 112,
            lastWordStartFrame: 150
        )
        XCTAssertEqual(multi.emitTokensAfterFrame, 150 - 112 - AsrManager.redecodeEmissionJitterFrames)
    }

    func testTrailingWordStartIndex() {
        XCTAssertEqual(AsrManager.trailingWordStartIndex(pieces: ["▁c", "ode", "▁and", "▁an"]), 3)
        // The loaded vocabulary normalizes the boundary to a leading space.
        XCTAssertEqual(AsrManager.trailingWordStartIndex(pieces: [" c", "ode", " and", " an"]), 3)
        XCTAssertEqual(AsrManager.trailingWordStartIndex(pieces: ["▁c", "ode", "▁anal", "y", "z"]), 2)
        XCTAssertNil(AsrManager.trailingWordStartIndex(pieces: ["▁one", "word"]), "index 0 would drop everything")
        XCTAssertNil(AsrManager.trailingWordStartIndex(pieces: ["no", "starts"]))
        XCTAssertNil(AsrManager.trailingWordStartIndex(pieces: []))
    }

    /// Clip 03 of the #855 fixtures: previous `▁c ode ▁and ▁an`, re-decode
    /// `, ▁and ▁anal y z ing`. `an` is a strict prefix of `analyzing`: retire it;
    /// the seam comma and the re-emitted `and` (dup of the kept `and`@153) go.
    func testReconcileFinalWindowSeam_FragmentPrefixRetiresAndStripsSeam() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4],
            previousTimestamps: [147, 149, 153, 156],
            trailingWordStart: 3,
            currentTokens: [99, 3, 5, 6, 7, 8],
            currentTimestamps: [152, 155, 159, 162, 164, 166],
            currentPieces: [",", " and", " anal", "y", "z", "ing"],
            previousPieces: [" c", "ode", " and", " an"]
        )
        XCTAssertEqual(seam.droppedPrevious, 1)
        XCTAssertEqual(seam.droppedCurrent, 2)
    }

    /// #897 corpus `99C654B5`: previous `… ▁in ▁the ▁box`, re-decode `x , ▁but …`.
    /// The leading `x` is a continuation piece — the tail of `box`, whose start
    /// fell under the cutoff — so `box` is kept and `x` dropped; the comma is
    /// real. The first version of this rule produced `in thex`.
    func testReconcileFinalWindowSeam_ContinuationHeadKeepsPreviousWord() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],
            previousTimestamps: [300, 303, 306],
            trailingWordStart: 2,
            currentTokens: [30, 99, 31, 32],
            currentTimestamps: [308, 309, 311, 314],
            currentPieces: ["x", ",", " but", " she"],
            previousPieces: [" in", " the", " box"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0, "`but` is not `box` nor an extension of it")
        XCTAssertEqual(seam.droppedCurrent, 1, "only the continuation tail `x` goes; the comma stays")
    }

    /// #897 corpus `0A575FDA`: previous `… ▁capture ▁every`, re-decode
    /// `turing ▁everything ▁saying`. The continuation `turing` is dropped, and
    /// `every` is a strict prefix of `everything`: retire the fragment.
    func testReconcileFinalWindowSeam_ContinuationThenPrefixRetires() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2],
            previousTimestamps: [400, 404],
            trailingWordStart: 1,
            currentTokens: [40, 41, 42],
            currentTimestamps: [407, 409, 414],
            currentPieces: ["turing", " everything", " saying"],
            previousPieces: [" capture", " every"]
        )
        XCTAssertEqual(seam.droppedPrevious, 1)
        XCTAssertEqual(seam.droppedCurrent, 1)
    }

    /// Same word re-emitted: keep the previous copy (it carries the period the
    /// re-decode omits at the audio end) and drop the re-emission as a duplicate.
    func testReconcileFinalWindowSeam_SameWordKeepsPreviousAndDropsReemission() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4],  // … ▁them ▁out .
            previousTimestamps: [250, 254, 258, 261],
            trailingWordStart: 2,
            currentTokens: [3, 9],
            currentTimestamps: [259, 270],
            currentPieces: [" out", " uh"],
            previousPieces: [" help", " them", " out", "."]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 1)
    }

    /// #897 corpus doubled-word shape: previous `… ▁two .`, re-decode `. ▁two ▁in`.
    /// The re-emitted period duplicates the kept one and the word behind it
    /// duplicates `two`: both go, nothing is retired.
    func testReconcileFinalWindowSeam_DuplicatePunctuationThenWordBothDropped() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 7883],  // ▁or ▁two .
            previousTimestamps: [200, 203, 205],
            trailingWordStart: 1,
            currentTokens: [7883, 2, 5],
            currentTimestamps: [204, 206, 212],
            currentPieces: [".", " two", " in"],
            previousPieces: [" or", " two", "."]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 2)
    }

    /// A different word overlapping the previous word's span (`properly.` vs
    /// `correctly.`): the re-decode disagrees with more context — retire.
    func testReconcileFinalWindowSeam_DifferentOverlappingWordRetires() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 7883],
            previousTimestamps: [100, 104, 108],
            trailingWordStart: 1,
            currentTokens: [9, 7883, 10],
            currentTimestamps: [105, 109, 120],
            currentPieces: [" correctly", ".", " and"],
            previousPieces: [" it", " properly", "."]
        )
        XCTAssertEqual(seam.droppedPrevious, 2)
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    /// A different word that starts *after* the previous word's span: the
    /// re-decode skipped the previous word — keep it, drop nothing.
    func testReconcileFinalWindowSeam_LaterWordKeepsPrevious() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],
            previousTimestamps: [100, 104, 108],
            trailingWordStart: 2,
            currentTokens: [9, 10],
            currentTimestamps: [130, 134],
            currentPieces: [" later", " words"],
            previousPieces: [" a", " b", " kept"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 0)
    }

    func testIsPunctuationPiece() {
        XCTAssertTrue(AsrManager.isPunctuationPiece(","))
        XCTAssertTrue(AsrManager.isPunctuationPiece("▁."))
        XCTAssertTrue(AsrManager.isPunctuationPiece(" ?"))
        XCTAssertFalse(AsrManager.isPunctuationPiece(" and"))
        XCTAssertFalse(AsrManager.isPunctuationPiece("▁"))
        XCTAssertFalse(AsrManager.isPunctuationPiece(""))
    }

    /// Same word, different segmentation: the previous `▁out` is one piece, the
    /// re-decode spells it `▁o` `ut` with a comma behind. Id-level matching
    /// cannot see the duplicate, so the same-word branch must consume the whole
    /// re-emitted word range itself.
    func testReconcileFinalWindowSeam_SameWordDifferentSegmentationIsConsumed() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],  // ▁them ▁out .
            previousTimestamps: [254, 258, 261],
            trailingWordStart: 1,
            currentTokens: [40, 41, 99, 50],  // ▁o ut , ▁and
            currentTimestamps: [259, 260, 262, 270],
            currentPieces: [" o", "ut", ",", " and"],
            previousPieces: [" them", " out", "."]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(
            seam.droppedCurrent, 3,
            "both pieces of the re-emitted `out` go, and so does the comma behind it: the kept `out.` already carries its punctuation"
        )
    }

    /// Punctuation policy, the other way round: the kept previous word has no
    /// trailing punctuation, so the re-decoded period is the only one and stays.
    func testReconcileFinalWindowSeam_SameWordKeepsRedecodedPunctuationWhenPreviousHasNone() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2],  // ▁them ▁out
            previousTimestamps: [254, 258],
            trailingWordStart: 1,
            currentTokens: [2, 7883, 50],  // ▁out . ▁and
            currentTimestamps: [259, 261, 270],
            currentPieces: [" out", ".", " and"],
            previousPieces: [" them", " out"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 1, "only the re-emitted word; its period is the only one")
    }

    /// `well` vs `we'll`: the comparison form keeps interior apostrophes, so the
    /// re-decoded contraction is a *different* overlapping word and retires the
    /// earlier `well` instead of being consumed as a re-emission of it. Both
    /// tokenizations of the contraction are covered.
    func testReconcileFinalWindowSeam_ContractionIsNotTheSameWord() {
        XCTAssertNotEqual(AsrManager.wordCore([" well"]), AsrManager.wordCore([" we'll"]))
        XCTAssertNotEqual(AsrManager.wordCore([" cant"]), AsrManager.wordCore([" can\u{2019}t"]))
        XCTAssertEqual(AsrManager.wordCore([" we", "'", "ll"]), "we'll")
        XCTAssertEqual(AsrManager.wordCore([" out", "."]), "out")
        XCTAssertEqual(AsrManager.firstWordPieces([" we", "'", "ll", " go"]), [" we", "'", "ll"])
        // Only a bare apostrophe/hyphen joins; the boundary-marked variants the v3
        // vocabulary also carries (" '" = 7306, " -" = 5071) start a new word.
        for joiner in ["'", "\u{2019}", "-"] {
            XCTAssertTrue(AsrManager.isJoiningPunctuationPiece(joiner), joiner)
            XCTAssertFalse(AsrManager.isJoiningPunctuationPiece(" " + joiner), "space-marked " + joiner)
            XCTAssertFalse(AsrManager.isJoiningPunctuationPiece("▁" + joiner), "boundary-marked " + joiner)
        }
        XCTAssertFalse(AsrManager.isJoiningPunctuationPiece(","))
        XCTAssertEqual(
            AsrManager.firstWordPieces([" rock", " -", "and", " roll"]), [" rock"],
            "a boundary-marked hyphen ends the word; it is not absorbed")
        XCTAssertEqual(AsrManager.firstWordPieces([" rock", "-", "and", " roll"]), [" rock", "-", "and"])

        let singlePiece = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2],  // ▁and ▁well
            previousTimestamps: [96, 100],
            trailingWordStart: 1,
            currentTokens: [30, 31],  // ▁we'll ▁go
            currentTimestamps: [101, 108],
            currentPieces: [" we'll", " go"],
            previousPieces: [" and", " well"]
        )
        XCTAssertEqual(singlePiece.droppedPrevious, 1, "the re-decoded contraction replaces `well`")
        XCTAssertEqual(singlePiece.droppedCurrent, 0)

        let splitPieces = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2],
            previousTimestamps: [96, 100],
            trailingWordStart: 1,
            currentTokens: [30, 99, 32, 33],  // ▁we ' ll ▁go
            currentTimestamps: [101, 102, 103, 108],
            currentPieces: [" we", "'", "ll", " go"],
            previousPieces: [" and", " well"]
        )
        XCTAssertEqual(splitPieces.droppedPrevious, 1)
        XCTAssertEqual(splitPieces.droppedCurrent, 0)
    }

    /// A word that begins with a boundary-marked apostrophe (`▁'` `cause`) is one
    /// word, not leading punctuation plus a stray continuation. Same-word
    /// consumption must cover exactly that word.
    func testReconcileFinalWindowSeam_LeadingApostropheWordIsOneWord() {
        XCTAssertEqual(AsrManager.firstWordPieces([" '", "cause", " every"]), [" '", "cause"])
        XCTAssertEqual(AsrManager.wordCore([" '", "cause"]), "cause")
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3],  // ▁just ▁' cause
            previousTimestamps: [96, 100, 101],
            trailingWordStart: 1,
            currentTokens: [2, 3, 9],  // ▁' cause ▁every
            currentTimestamps: [100, 102, 110],
            currentPieces: [" '", "cause", " every"],
            previousPieces: [" just", " '", "cause"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 2, "exactly the re-emitted `'cause`; `every` stays")
    }

    /// Token ids are not punctuation evidence: id 7948 is `ó` in the v3
    /// vocabulary although it sits in `ASRConstants.punctuationTokens`. A word
    /// starting with it must not be dropped as a seam artifact.
    func testReconcileFinalWindowSeam_ClassifiesPunctuationByPieceNotId() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2],
            previousTimestamps: [96, 100],
            trailingWordStart: 1,
            currentTokens: [7948, 9],
            currentTimestamps: [107, 112],  // starts after the previous word's span: a new word
            currentPieces: [" ó", " si"],
            previousPieces: [" digo", " que"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 0, "`ó` is a word, not punctuation, regardless of its id")
    }

    /// A genuine later repetition of the same word (`go` … `go again`) must not
    /// be consumed: the same-word rule applies only when the re-emission starts
    /// within the previous word's span.
    func testReconcileFinalWindowSeam_LaterRepetitionOfSameWordIsKept() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2],  // ▁let's ▁go
            previousTimestamps: [96, 100],
            trailingWordStart: 1,
            currentTokens: [30, 2, 31],  // o ▁go ▁again
            currentTimestamps: [103, 115, 119],
            currentPieces: ["o", " go", " again"],
            previousPieces: [" let's", " go"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0)
        XCTAssertEqual(seam.droppedCurrent, 1, "only the continuation head; the later `go` is a new word")
    }

    /// Adversarial for the prefix rule: a continuation head followed by a new
    /// word that merely extends the previous text (`an` … `another`). The new
    /// word starts after the previous word's span, so it is kept as a new word
    /// and the previous `an` survives.
    func testReconcileFinalWindowSeam_LaterExtendingWordDoesNotRetire() {
        let seam = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2],
            previousTimestamps: [96, 100],
            trailingWordStart: 1,
            currentTokens: [30, 31, 32],
            currentTimestamps: [103, 112, 116],
            currentPieces: ["x", " another", " thing"],
            previousPieces: [" is", " an"]
        )
        XCTAssertEqual(seam.droppedPrevious, 0, "`another` at 112 starts after `an`'s span; not a replacement")
        XCTAssertEqual(seam.droppedCurrent, 1, "the continuation head still goes")
    }

    func testReconcileFinalWindowSeam_EmptyEarlyMisalignedAndDegenerateInputsKeepPrevious() {
        let empty = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4], previousTimestamps: [250, 254, 258, 261], trailingWordStart: 2,
            currentTokens: [], currentTimestamps: [], currentPieces: [], previousPieces: [" a", " b", " c", " d"])
        XCTAssertEqual(empty.droppedPrevious, 0)
        XCTAssertEqual(empty.droppedCurrent, 0)
        // Only a punctuation before the previous word's onset: nothing re-emitted.
        let earlyOnly = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3, 4], previousTimestamps: [250, 254, 258, 261], trailingWordStart: 2,
            currentTokens: [7883], currentTimestamps: [240], currentPieces: ["."],
            previousPieces: [" a", " b", " out", "."])
        XCTAssertEqual(earlyOnly.droppedPrevious, 0)
        XCTAssertEqual(earlyOnly.droppedCurrent, 1, "punctuation before the previous word's onset is a seam artifact")
        // Missing or misaligned piece arrays: a no-op, never a mass drop.
        let noPieces = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3], previousTimestamps: [100, 104, 108], trailingWordStart: 2,
            currentTokens: [9, 10], currentTimestamps: [110, 114])
        XCTAssertEqual(noPieces.droppedPrevious, 0)
        XCTAssertEqual(noPieces.droppedCurrent, 0)
        let misaligned = AsrManager.reconcileFinalWindowSeam(
            previousTokens: [1, 2, 3], previousTimestamps: [100, 104, 108], trailingWordStart: 2,
            currentTokens: [9, 10], currentTimestamps: [110, 114], currentPieces: [" only"],
            previousPieces: [" a", " b", " c"])
        XCTAssertEqual(misaligned.droppedCurrent, 0)
        XCTAssertEqual(
            AsrManager.reconcileFinalWindowSeam(
                previousTokens: [1], previousTimestamps: [5], trailingWordStart: 0,
                currentTokens: [2], currentTimestamps: [6], currentPieces: [" x"], previousPieces: [" y"]
            ).droppedPrevious, 0)
    }
}
