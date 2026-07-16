import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class ConstrainedClusterAssignmentTests: XCTestCase {

    func testCoChunkSpeakersGetDistinctClusters() {
        // Both embeddings score highest against cluster 0. Plain argmax would
        // merge them; the per-chunk constraint forces the weaker match onto
        // cluster 1 (the #801 failure mode).
        let scores: [[Double]] = [
            [0.9, 0.3],
            [0.8, 0.6],
        ]
        let assignments = ConstrainedClusterAssignment.assign(
            scores: scores,
            chunkIndices: [0, 0]
        )
        XCTAssertEqual(assignments, [0, 1])
    }

    func testSeparateChunksAreUnconstrained() {
        let scores: [[Double]] = [
            [0.9, 0.3],
            [0.8, 0.6],
        ]
        let assignments = ConstrainedClusterAssignment.assign(
            scores: scores,
            chunkIndices: [0, 1]
        )
        XCTAssertEqual(assignments, [0, 0], "Embeddings in different chunks may share a cluster")
    }

    func testMoreSpeakersThanClustersDropsWeakestSlot() {
        let scores: [[Double]] = [
            [0.9],
            [0.2],
        ]
        let assignments = ConstrainedClusterAssignment.assign(
            scores: scores,
            chunkIndices: [0, 0]
        )
        XCTAssertEqual(assignments, [0, -2], "Surplus co-chunk speaker should be left unassigned")
    }

    func testSingleEmbeddingPerChunkIsPlainArgmax() {
        let scores: [[Double]] = [
            [0.1, 0.7, 0.4],
            [0.5, 0.2, 0.9],
        ]
        let assignments = ConstrainedClusterAssignment.assign(
            scores: scores,
            chunkIndices: [3, 7]
        )
        XCTAssertEqual(assignments, [1, 2])
    }

    func testEmptyInput() {
        XCTAssertTrue(ConstrainedClusterAssignment.assign(scores: [], chunkIndices: []).isEmpty)
    }

    func testMaximizesTotalScoreAcrossChunk() {
        // Row 0 slightly prefers cluster 1, but giving cluster 1 to row 1
        // yields a higher total; the matching should be globally optimal,
        // not first-come-first-served.
        let scores: [[Double]] = [
            [0.50, 0.55],
            [0.10, 0.90],
        ]
        let assignments = ConstrainedClusterAssignment.assign(
            scores: scores,
            chunkIndices: [0, 0]
        )
        XCTAssertEqual(assignments, [0, 1])
    }
}
