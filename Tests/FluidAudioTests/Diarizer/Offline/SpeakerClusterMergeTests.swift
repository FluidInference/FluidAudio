import XCTest

@testable import FluidAudio

/// Unit tests for the agglomerative over-split merge. Pure vector math — no models.
final class SpeakerClusterMergeTests: XCTestCase {

    /// Well-separated (orthogonal) speakers are never folded together.
    func testKeepsSeparateSpeakers() {
        let emb: [[Double]] = [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]
        let out = SpeakerClusterMerge.mergedLabels(
            embeddings: emb, labels: [0, 0, 1, 1], threshold: 0.75)
        XCTAssertEqual(out, [0, 0, 1, 1])
    }

    /// A drift chain of near-parallel sub-clusters collapses to one speaker.
    func testMergesOverSplitChain() {
        let emb: [[Double]] = [[1, 0, 0], [0.98, 0.2, 0], [0.95, 0.31, 0]]
        let out = SpeakerClusterMerge.mergedLabels(
            embeddings: emb, labels: [0, 1, 2], threshold: 0.75)
        XCTAssertEqual(Set(out).count, 1)
    }

    /// The threshold gates the merge: a ~0.6-cosine pair splits at 0.75, folds at 0.5.
    func testThresholdGates() {
        let emb: [[Double]] = [[1, 0, 0], [0.6, 0.8, 0]]  // centroid cosine = 0.6
        XCTAssertEqual(
            Set(SpeakerClusterMerge.mergedLabels(embeddings: emb, labels: [0, 1], threshold: 0.75))
                .count, 2)
        XCTAssertEqual(
            Set(SpeakerClusterMerge.mergedLabels(embeddings: emb, labels: [0, 1], threshold: 0.5))
                .count, 1)
    }

    /// Non-contiguous input labels are renumbered to 0..<k (no merge when orthogonal).
    func testRenumbersNonContiguous() {
        let emb: [[Double]] = [[1, 0, 0], [0, 1, 0]]
        let out = SpeakerClusterMerge.mergedLabels(
            embeddings: emb, labels: [5, 3], threshold: 0.99)
        XCTAssertEqual(out, [0, 1])
    }

    /// Zero-energy embeddings (garbage masks) are ignored without crashing.
    func testIgnoresZeroVectors() {
        let emb: [[Double]] = [[0, 0, 0], [1, 0, 0], [1, 0, 0]]
        let out = SpeakerClusterMerge.mergedLabels(
            embeddings: emb, labels: [0, 1, 1], threshold: 0.75)
        XCTAssertEqual(out.count, 3)
    }
}
