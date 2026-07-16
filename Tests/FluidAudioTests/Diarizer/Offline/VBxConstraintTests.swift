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
}
