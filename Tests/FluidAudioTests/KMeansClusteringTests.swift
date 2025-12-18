import XCTest

@testable import FluidAudio

@available(macOS 14.0, iOS 17.0, *)
final class KMeansClusteringTests: XCTestCase {

    // MARK: - Basic Clustering Tests

    func testKMeansProducesRequestedClusterCount() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.0, 1.0],
            [0.1, 1.1],
            [-1.0, 0.0],
            [-0.9, 0.1],
        ]

        // Use seed for reproducibility (Expert Panel: Crispin)
        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 6)
        let uniqueClusters = Set(clusters)
        XCTAssertEqual(uniqueClusters.count, 3)
    }

    func testKMeansHandlesSingleCluster() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [1.1, 0.1],
            [0.9, 0.2],
        ]

        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 1,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 3)
        XCTAssertTrue(clusters.allSatisfy { $0 == 0 })
    }

    func testKMeansReturnsSequentialAssignmentsForMoreClustersThanEmbeddings() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [0.0, 1.0],
        ]

        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 5,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 2)
        XCTAssertEqual(clusters[0], 0)
        XCTAssertEqual(clusters[1], 1)
    }

    func testKMeansComputesCentroids() {
        let embeddings: [[Double]] = [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]

        let (clusters, centroids) = KMeansClustering.clusterWithCentroids(
            embeddings: embeddings,
            numClusters: 2,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(centroids.count, 2)
        XCTAssertEqual(clusters.count, 4)
    }

    // MARK: - Reproducibility Tests (Expert Panel: Crispin)

    func testKMeansIsDeterministicWithSameSeed() {
        let embeddings: [[Double]] = [
            [1.0, 0.0], [1.1, 0.1], [0.0, 1.0],
            [0.1, 1.1], [-1.0, 0.0], [-0.9, 0.1],
        ]

        let clusters1 = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            seed: 12345
        )

        let clusters2 = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            seed: 12345
        )

        XCTAssertEqual(clusters1, clusters2)
    }

    // MARK: - Realistic Dimension Tests (Expert Panel: Adzic)

    func testKMeansWithRealisticEmbeddingDimension() {
        let dimension = 192  // Real speaker embedding dimension
        var rng = KMeansClustering.SeededRNG(seed: 42)

        let embeddings = (0..<20).map { _ in
            (0..<dimension).map { _ in Double.random(in: -1...1, using: &rng) }
        }

        let clusters = KMeansClustering.cluster(
            embeddings: embeddings,
            numClusters: 3,
            maxIterations: 100,
            seed: 42
        )

        XCTAssertEqual(clusters.count, 20)
        XCTAssertEqual(Set(clusters).count, 3)
    }
}
