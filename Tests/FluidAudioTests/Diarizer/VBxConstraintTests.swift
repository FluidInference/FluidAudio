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
}
