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
        // Prefix tail and suffix head share a 5-token run. Merge drops the
        // overlap from prefix's tail and the overlap from suffix's head.
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
}
