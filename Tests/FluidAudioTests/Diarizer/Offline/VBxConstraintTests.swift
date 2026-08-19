import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class VBxConstraintTests: XCTestCase {

    func testVBxOutputReportsAdjustedFlag() {
        let output = VBxOutput(
            gamma: [],
            pi: [],
            hardClusters: [],
            centroids: [],
            numClusters: 3,
            elbos: [],
            wasAdjusted: true,
            originalClusterCount: 5
        )
        XCTAssertTrue(output.wasAdjusted)
        XCTAssertEqual(output.originalClusterCount, 5)
    }

    func testVBxOutputDefaultsToNotAdjusted() {
        let output = VBxOutput(
            gamma: [],
            pi: [],
            hardClusters: [],
            centroids: [],
            numClusters: 3,
            elbos: []
        )
        XCTAssertFalse(output.wasAdjusted)
        XCTAssertNil(output.originalClusterCount)
    }

    func testVBxOutputTracksOriginalClusterCount() {
        let output = VBxOutput(
            gamma: [],
            pi: [],
            hardClusters: [],
            centroids: [],
            numClusters: 2,
            elbos: [],
            wasAdjusted: true,
            originalClusterCount: 8
        )
        XCTAssertEqual(output.numClusters, 2)
        XCTAssertEqual(output.originalClusterCount, 8)
    }

    // MARK: - Active cluster count (pyannote auto_num_clusters parity)

    func testActiveClusterCountIgnoresCollapsedClusters() {
        // VBx warm-started with 5 AHC clusters but collapsed 3 of them
        // (mixture weight ~0). The detected speaker count is 2, not 5.
        let output = VBxOutput(
            gamma: [],
            pi: [0.63, 0.0, 1e-12, 0.37, 0.0],
            hardClusters: [],
            centroids: [],
            numClusters: 5,
            elbos: []
        )
        XCTAssertEqual(output.activeClusterCount, 2)
    }

    func testActiveClusterCountWithAllClustersActive() {
        let output = VBxOutput(
            gamma: [],
            pi: [0.5, 0.3, 0.2],
            hardClusters: [],
            centroids: [],
            numClusters: 3,
            elbos: []
        )
        XCTAssertEqual(output.activeClusterCount, 3)
    }

    func testActiveClusterCountFallsBackToNumClustersWithoutPi() {
        let output = VBxOutput(
            gamma: [],
            pi: [],
            hardClusters: [],
            centroids: [],
            numClusters: 4,
            elbos: []
        )
        XCTAssertEqual(output.activeClusterCount, 4)
    }

    // MARK: - Assigned cluster count (the count callers observe, #802 review)

    func testAssignedClusterCountIgnoresClustersWithNoArgmaxWins() {
        // Clusters 3 and 4 keep trace mixture weight (pi > epsilon) but never win
        // an embedding's argmax responsibility: they receive no hard assignments
        // and vanish from the output. The pi census says 5; callers see 3.
        let output = VBxOutput(
            gamma: [
                [0.7, 0.1, 0.1, 0.05, 0.05],
                [0.1, 0.7, 0.1, 0.05, 0.05],
                [0.1, 0.1, 0.7, 0.05, 0.05],
                [0.6, 0.2, 0.1, 0.05, 0.05],
            ],
            pi: [0.4, 0.3, 0.28, 0.01, 0.01],
            hardClusters: [],
            centroids: [],
            numClusters: 5,
            elbos: []
        )
        XCTAssertEqual(output.activeClusterCount, 5, "pi census counts the ghost clusters")
        XCTAssertEqual(output.assignedClusterCount, 3, "assignment count must not")
    }

    func testAssignedClusterCountMatchesActiveWhenAllClustersWin() {
        let output = VBxOutput(
            gamma: [
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ],
            pi: [0.4, 0.3, 0.3],
            hardClusters: [],
            centroids: [],
            numClusters: 3,
            elbos: []
        )
        XCTAssertEqual(output.assignedClusterCount, 3)
        XCTAssertEqual(output.assignedClusterCount, output.activeClusterCount)
    }

    func testAssignedClusterCountFallsBackWithoutGamma() {
        let output = VBxOutput(
            gamma: [],
            pi: [0.6, 0.4, 1e-12],
            hardClusters: [],
            centroids: [],
            numClusters: 3,
            elbos: []
        )
        XCTAssertEqual(output.assignedClusterCount, 2, "no gamma -> fall back to the pi census")
    }

    /// The #802-review probe scenario: numSpeakers equals the pi census but two
    /// clusters win no assignments. The constraint must fire (needsAdjustment),
    /// not be silently satisfied.
    func testConstraintFiresWhenRequestMatchesPiCensusButNotAssignments() {
        let output = VBxOutput(
            gamma: [
                [0.7, 0.1, 0.1, 0.05, 0.05],
                [0.1, 0.7, 0.1, 0.05, 0.05],
                [0.1, 0.1, 0.7, 0.05, 0.05],
                [0.6, 0.2, 0.1, 0.05, 0.05],
                [0.2, 0.6, 0.1, 0.05, 0.05],
                [0.1, 0.2, 0.6, 0.05, 0.05],
            ],
            pi: [0.4, 0.3, 0.28, 0.01, 0.01],
            hardClusters: [],
            centroids: [],
            numClusters: 5,
            elbos: []
        )
        // numEmbeddings must exceed the request: resolve() clamps to it.
        let constraints = SpeakerCountConstraints.resolve(
            numEmbeddings: 6, numSpeakers: 5, minSpeakers: nil, maxSpeakers: nil)
        XCTAssertFalse(
            constraints.needsAdjustment(detectedCount: output.activeClusterCount),
            "gating on the pi census silently ignores the request (the old bug)")
        XCTAssertTrue(
            constraints.needsAdjustment(detectedCount: output.assignedClusterCount),
            "gating on the assignment count re-clusters to the requested 5")
    }
}
