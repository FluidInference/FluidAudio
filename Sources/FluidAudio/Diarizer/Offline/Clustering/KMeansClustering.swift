import Accelerate
import Foundation
import OSLog
import os.signpost

/// K-Means clustering for forcing a specific number of speakers.
@available(macOS 14.0, iOS 17.0, *)
struct KMeansClustering {

    private static let logger = AppLogger(category: "KMeansClustering")
    private static let signposter = OSSignposter(
        subsystem: "com.fluidaudio.diarization",
        category: .pointsOfInterest
    )

    /// Clusters embeddings into exactly `numClusters` groups.
    ///
    /// - Parameters:
    ///   - embeddings: Array of embedding vectors (each is a [Double] of same dimension).
    ///   - numClusters: Target number of clusters.
    ///   - maxIterations: Maximum iterations before stopping.
    ///   - seed: Optional seed for reproducible centroid initialization.
    /// - Returns: Cluster assignment for each embedding (0-indexed).
    static func cluster(
        embeddings: [[Double]],
        numClusters: Int,
        maxIterations: Int = 300,
        seed: UInt64? = nil
    ) -> [Int] {
        clusterWithCentroids(
            embeddings: embeddings,
            numClusters: numClusters,
            maxIterations: maxIterations,
            seed: seed
        ).clusters
    }

    /// Clusters embeddings and returns both assignments and centroids.
    static func clusterWithCentroids(
        embeddings: [[Double]],
        numClusters: Int,
        maxIterations: Int = 300,
        seed: UInt64? = nil
    ) -> (clusters: [Int], centroids: [[Double]]) {
        let kmeansState = signposter.beginInterval("KMeans Clustering")
        defer { signposter.endInterval("KMeans Clustering", kmeansState) }

        let count = embeddings.count
        guard count > 0 else {
            return ([], [])
        }
        guard let dimension = embeddings.first?.count, dimension > 0 else {
            return (Array(repeating: 0, count: count), [])
        }

        let k = min(numClusters, count)
        if k <= 0 {
            return (Array(repeating: 0, count: count), [])
        }
        if count <= k {
            return (Array(0..<count), embeddings)
        }

        var rng = SeededRNG(seed: seed ?? UInt64.random(in: 0...UInt64.max))
        let normalized = normalizeEmbeddings(embeddings)
        var centroids = initializeCentroids(from: normalized, k: k, rng: &rng)
        var assignments = [Int](repeating: 0, count: count)

        for iteration in 0..<maxIterations {
            let newAssignments = assignToCentroids(normalized, centroids: centroids)
            if newAssignments == assignments {
                assignments = newAssignments
                break
            }
            assignments = newAssignments
            centroids = updateCentroids(
                embeddings: normalized,
                assignments: assignments,
                k: k,
                dimension: dimension,
                rng: &rng
            )

            if iteration > 80 {
                logger.warning("K-Means slow convergence at iteration \(iteration + 1)")
            }
        }

        return (assignments, centroids)
    }

    private static func normalizeEmbeddings(_ embeddings: [[Double]]) -> [[Double]] {
        embeddings.map { embedding in
            var norm: Double = 0
            vDSP_svesqD(embedding, 1, &norm, vDSP_Length(embedding.count))
            norm = sqrt(norm)
            guard norm > 1e-10 else { return embedding }
            var invNorm = 1.0 / norm
            var result = [Double](repeating: 0, count: embedding.count)
            vDSP_vsmulD(embedding, 1, &invNorm, &result, 1, vDSP_Length(embedding.count))
            return result
        }
    }

    private static func initializeCentroids(
        from embeddings: [[Double]],
        k: Int,
        rng: inout SeededRNG
    ) -> [[Double]] {
        var indices = Array(embeddings.indices)
        indices.shuffle(using: &rng)
        return Array(indices.prefix(k).map { embeddings[$0] })
    }

    private static func assignToCentroids(_ embeddings: [[Double]], centroids: [[Double]]) -> [Int] {
        embeddings.map { embedding in
            var bestCluster = 0
            var bestDistance = Double.greatestFiniteMagnitude
            for (idx, centroid) in centroids.enumerated() {
                let dist = euclideanDistanceSquared(embedding, centroid)
                if dist < bestDistance {
                    bestDistance = dist
                    bestCluster = idx
                }
            }
            return bestCluster
        }
    }

    private static func euclideanDistanceSquared(_ a: [Double], _ b: [Double]) -> Double {
        guard a.count == b.count else { return Double.greatestFiniteMagnitude }
        var diff = [Double](repeating: 0, count: a.count)
        vDSP_vsubD(b, 1, a, 1, &diff, 1, vDSP_Length(a.count))
        var result: Double = 0
        vDSP_svesqD(diff, 1, &result, vDSP_Length(a.count))
        return result
    }

    private static func updateCentroids(
        embeddings: [[Double]],
        assignments: [Int],
        k: Int,
        dimension: Int,
        rng: inout SeededRNG
    ) -> [[Double]] {
        var sums = [[Double]](repeating: [Double](repeating: 0, count: dimension), count: k)
        var counts = [Int](repeating: 0, count: k)

        for (idx, cluster) in assignments.enumerated() {
            let embedding = embeddings[idx]
            counts[cluster] += 1
            for d in 0..<dimension {
                sums[cluster][d] += embedding[d]
            }
        }

        return (0..<k).map { cluster in
            let count = counts[cluster]
            guard count > 0 else {
                // Empty cluster: reinitialize from random data point (sklearn approach)
                return embeddings.randomElement(using: &rng) ?? sums[cluster]
            }
            var invCount = 1.0 / Double(count)
            var result = [Double](repeating: 0, count: dimension)
            vDSP_vsmulD(sums[cluster], 1, &invCount, &result, 1, vDSP_Length(dimension))
            return result
        }
    }

    /// Seeded random number generator for reproducible clustering.
    /// Uses LCG (Linear Congruential Generator) algorithm.
    struct SeededRNG: RandomNumberGenerator {
        var state: UInt64

        init(seed: UInt64) {
            self.state = seed
        }

        mutating func next() -> UInt64 {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            return state
        }
    }
}
