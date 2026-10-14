import Foundation
import XCTest

@testable import FluidAudio

@available(macOS 14, iOS 17, *)
final class CohereLongFormTests: XCTestCase {

    // MARK: - Config

    func testChunkOverlapMatchesUpstream() {
        // Matches `overlap_chunk_second: 5` in upstream cohere-pytorch config.
        XCTAssertEqual(CohereAsrConfig.chunkOverlapSeconds, 5.0)
    }

    func testChunkHopIsMaxMinusOverlap() {
        XCTAssertEqual(
            CohereAsrConfig.chunkHopSeconds,
            CohereAsrConfig.maxAudioSeconds - CohereAsrConfig.chunkOverlapSeconds)
        XCTAssertEqual(CohereAsrConfig.chunkHopSeconds, 30.0)
    }

    // MARK: - mergeTokenStreams

    func testMergePrefixEmpty() {
        let merged = CoherePipeline.mergeTokenStreams(prefix: [], suffix: [1, 2, 3])
        XCTAssertEqual(merged, [1, 2, 3])
    }

    func testMergeSuffixEmpty() {
        let merged = CoherePipeline.mergeTokenStreams(prefix: [1, 2, 3], suffix: [])
        XCTAssertEqual(merged, [1, 2, 3])
    }

    func testMergeNoCommonRunFallsBackToConcat() {
        let merged = CoherePipeline.mergeTokenStreams(
            prefix: [10, 11, 12, 13],
            suffix: [20, 21, 22, 23])
        XCTAssertEqual(merged, [10, 11, 12, 13, 20, 21, 22, 23])
    }

    func testMergeShortMatchBelowThresholdFallsBackToConcat() {
        // A common run of 3 tokens is below the default minMatch=4 threshold.
        let merged = CoherePipeline.mergeTokenStreams(
            prefix: [1, 2, 3, 7, 8, 9],
            suffix: [7, 8, 9, 100, 200])
        XCTAssertEqual(merged, [1, 2, 3, 7, 8, 9, 7, 8, 9, 100, 200])
    }

    func testMergeOverlapAtBoundary() {
        // Prefix tail and suffix head share a 5-token run. Merge keeps one
        // copy of the run and continues with the suffix after it.
        let prefix = [1, 2, 3, 4, 50, 51, 52, 53, 54]
        let suffix = [50, 51, 52, 53, 54, 60, 61, 62]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 4, 50, 51, 52, 53, 54, 60, 61, 62])
    }

    func testMergeOverlapOffsetWithinWindow() {
        // The overlap doesn't have to align with the absolute boundary — the
        // merge uses an LCS over a bounded window, so a shifted match still
        // works.
        let prefix = [1, 2, 3, 90, 91, 92, 93, 94, 95]
        let suffix = [91, 92, 93, 94, 95, 200, 201]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 90, 91, 92, 93, 94, 95, 200, 201])
    }

    func testMergePrefersLongestRun() {
        // Two candidate matches: a 4-token run earlier and a 5-token run later.
        // The longer run wins.
        let prefix = [1, 2, 3, 4, 7, 8, 9, 10, 11]
        let suffix = [1, 2, 3, 4, 7, 8, 9, 10, 11, 99]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        // The 9-token tail of prefix matches the 9-token head of suffix → the
        // suffix tail '99' is the only thing kept after the prefix.
        XCTAssertEqual(merged, [1, 2, 3, 4, 7, 8, 9, 10, 11, 99])
    }

    func testMergeWindowBoundsLcsCost() {
        // 4-token threshold; the run lives entirely inside the merge window
        // (default 32 tokens). Verifies behavior when prefix is much longer
        // than the window.
        let leadIn = Array(0..<200)
        let prefix = leadIn + [500, 501, 502, 503, 504]
        let suffix = [500, 501, 502, 503, 504, 700, 701]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, leadIn + [500, 501, 502, 503, 504, 700, 701])
    }

    // MARK: - Seam cut after the matched run (#1013)

    func testMergeDropsPrefixTailAfterMatchedRun() {
        // Both windows transcribe the overlap audio. The prefix's tokens after
        // the matched run (70, 71) and the suffix's copy of that same audio
        // (70, 71) must not both survive — the seam is cut at the run's end
        // and the suffix carries on from there.
        let prefix = [1, 2, 3, 50, 51, 52, 53, 70, 71]
        let suffix = [50, 51, 52, 53, 70, 71, 80, 81]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 50, 51, 52, 53, 70, 71, 80, 81])
    }

    func testMergeDropsPrefixTailWhenSuffixDecodesItDifferently() {
        // The prefix ends with a different reading of the overlap tail (90,
        // 91) than the suffix (70, 71). Keeping the prefix verbatim would emit
        // the tail twice; the cut after the run keeps only the suffix's copy.
        let prefix = [1, 2, 3, 50, 51, 52, 53, 90, 91]
        let suffix = [50, 51, 52, 53, 70, 71, 80, 81]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 50, 51, 52, 53, 70, 71, 80, 81])
    }

    func testMergeRepeatedPhraseCutsAtLaterPrefixOccurrence() {
        // The same 4-token phrase occurs twice inside the prefix window. The
        // seam must be cut at the later occurrence so the text between the
        // two occurrences (10, 11, 12) is not lost.
        let prefix = [1, 2, 50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53]
        let suffix = [50, 51, 52, 53, 80, 81]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53, 80, 81])
    }

    func testMergeRepeatedPhraseKeepsEarlierSuffixOccurrence() {
        // The phrase repeats inside the suffix window. Cutting at the later
        // suffix occurrence would drop (10, 11, 12); the earlier one wins.
        let prefix = [1, 2, 3, 50, 51, 52, 53]
        let suffix = [50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53, 80]
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, [1, 2, 3, 50, 51, 52, 53, 10, 11, 12, 50, 51, 52, 53, 80])
    }

    func testMergeMatchInsideLongDenseWindows() {
        // Dense 35 s windows now decode ~180–250 tokens; the ~5 s overlap is
        // ~30 tokens, so the run can sit well inside a 48-token window.
        let body = Array(1000..<1180)
        let overlap = Array(2000..<2030)
        let prefix = body + overlap
        let suffix = overlap + Array(3000..<3150)
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, body + overlap + Array(3000..<3150))
    }

    func testMergeIgnoresChancePhraseFarFromTheSeam() {
        // A common 4-token phrase sits at the very start of the prefix window
        // and at the very end of the suffix window. Cutting there would drop
        // ~90 tokens; the cut distance exceeds the window, so the match is
        // treated as a chance repeat and the streams are concatenated.
        let phrase = [7, 8, 9, 10]
        let prefix = Array(0..<100) + phrase + Array(200..<244)
        let suffix = Array(300..<344) + phrase + Array(400..<460)
        let merged = CoherePipeline.mergeTokenStreams(prefix: prefix, suffix: suffix)
        XCTAssertEqual(merged, prefix + suffix)
    }
}
