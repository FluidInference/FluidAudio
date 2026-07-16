import Foundation
import XCTest

@testable import FluidAudio

final class AHCClusteringTests: XCTestCase {

    private let clustering = AHCClustering()

    // MARK: - Empty & Single

    func testEmptyEmbeddingsReturnsEmptyAssignments() {
        let result = clustering.cluster(embeddingFeatures: [], threshold: 0.7)
        XCTAssertTrue(result.isEmpty)
    }

    func testSingleEmbeddingReturnsSingleCluster() {
        let result = clustering.cluster(embeddingFeatures: [[1.0, 0.0, 0.0]], threshold: 0.7)
        XCTAssertEqual(result.count, 1)
        XCTAssertEqual(result[0], 0)
    }

    // MARK: - Identical Embeddings

    func testIdenticalEmbeddingsClusterTogether() {
        let embedding = [1.0, 2.0, 3.0]
        let embeddings = Array(repeating: embedding, count: 5)

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.7)
        XCTAssertEqual(result.count, 5)

        // All should be in the same cluster
        let uniqueClusters = Set(result)
        XCTAssertEqual(uniqueClusters.count, 1, "Identical embeddings should all be in one cluster")
    }

    // MARK: - Orthogonal Embeddings

    func testOrthogonalGroupsSeparateWhenThresholdBelowGroupDistance() {
        // Two groups: pointing in very different directions. Unit-normalized,
        // the inter-group Euclidean distance is ~sqrt(2) ≈ 1.414, so a cut at
        // 0.8 merges within groups but not across them.
        let group1: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.95, 0.05, 0.0],
        ]
        let group2: [[Double]] = [
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.95, 0.05],
        ]

        let embeddings = group1 + group2
        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.8)
        XCTAssertEqual(result.count, 6)

        // Group 1 should share a cluster, group 2 should share a different cluster
        let group1Clusters = Set(result[0..<3])
        let group2Clusters = Set(result[3..<6])

        XCTAssertEqual(group1Clusters.count, 1, "Group 1 should be in one cluster")
        XCTAssertEqual(group2Clusters.count, 1, "Group 2 should be in one cluster")
        XCTAssertNotEqual(
            group1Clusters.first, group2Clusters.first,
            "Groups should be in different clusters"
        )
    }

    func testLargerThresholdMergesMore() {
        // pyannote semantics: the threshold is the dendrogram cut distance,
        // so raising it can only reduce the number of clusters.
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]

        let tight = Set(clustering.cluster(embeddingFeatures: embeddings, threshold: 0.5)).count
        let loose = Set(clustering.cluster(embeddingFeatures: embeddings, threshold: 1.5)).count

        XCTAssertEqual(tight, 2, "Cut below the inter-group distance should keep two clusters")
        XCTAssertEqual(loose, 1, "Cut above the inter-group distance should merge everything")
        XCTAssertLessThanOrEqual(loose, tight)
    }

    // MARK: - Cluster ID Properties

    func testClusterIdsAreContiguousFromZero() {
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.5)
        XCTAssertEqual(result.count, 3)

        let uniqueIds = Set(result).sorted()
        // IDs should be contiguous starting from 0
        for (i, id) in uniqueIds.enumerated() {
            XCTAssertEqual(id, i, "Cluster IDs should be contiguous from 0")
        }
    }

    // MARK: - Threshold Extremes

    func testMaxThresholdMergesAll() {
        // 2.0 is the largest possible distance between unit vectors, so
        // everything should merge into one cluster.
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 2.0)
        let uniqueClusters = Set(result)
        XCTAssertEqual(uniqueClusters.count, 1, "Maximum threshold should merge all into one cluster")
    }

    func testZeroThresholdKeepsDistinctEmbeddingsApart() {
        let embeddings: [[Double]] = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.0)
        let uniqueClusters = Set(result)
        XCTAssertEqual(uniqueClusters.count, 3, "Zero threshold should not merge distinct embeddings")
    }

    // MARK: - Zero Vector

    func testZeroDimensionEmbeddings() {
        // Zero-dimension vectors should all get cluster 0
        let embeddings: [[Double]] = [[], [], []]
        let result = clustering.cluster(embeddingFeatures: embeddings, threshold: 0.7)
        XCTAssertEqual(result.count, 3)
        for id in result {
            XCTAssertEqual(id, 0)
        }
    }
}
