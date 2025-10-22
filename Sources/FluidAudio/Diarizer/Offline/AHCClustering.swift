import Accelerate
import Foundation
import OSLog
import os.signpost

#if canImport(FastClusterWrapper)
import FastClusterWrapper
#elseif canImport(FluidAudio_FastClusterWrapper)
import FluidAudio_FastClusterWrapper
#endif

struct AHCClustering {
    private let logger = AppLogger(category: "OfflineAHC")
    private let signposter = OSSignposter(
        subsystem: "com.fluidaudio.diarization",
        category: .pointsOfInterest
    )

    // MARK: - Agglomerative Hierarchical Clustering
    func cluster(
        embeddingFeatures: [[Double]],
        threshold: Double
    ) -> [Int] {
        let count = embeddingFeatures.count
        guard count > 0 else { return [] }
        guard let dimension = embeddingFeatures.first?.count, dimension > 0 else {
            return Array(repeating: 0, count: count)
        }
        if count == 1 {
            return [0]
        }

        let ahcState = signposter.beginInterval("Agglomerative Hierarchical Clustering")

        let normalized = normalizeFeatures(embeddingFeatures, dimension: dimension)
        let dendrogramLength = (count - 1) * 4
        var dendrogram = [Double](repeating: 0, count: dendrogramLength)

        // MARK: - Fastcluster FFI Boundary
        let status = normalized.withUnsafeBufferPointer { normalizedPointer in
            dendrogram.withUnsafeMutableBufferPointer { dendrogramPointer in
                fastcluster_compute_centroid_linkage(
                    normalizedPointer.baseAddress,
                    count,
                    dimension,
                    dendrogramPointer.baseAddress,
                    dendrogramLength
                )
            }
        }

        guard status == FASTCLUSTER_WRAPPER_SUCCESS else {
            logger.error("fastcluster failed with status \(status.rawValue)")
            return Array(0..<count)
        }

        let distanceThreshold = convertThresholdToDistance(threshold)
        let assignments = assignmentsFromDendrogram(
            dendrogram,
            count: count,
            distanceThreshold: distanceThreshold
        )

        let result = remapClusterIds(assignments)
        signposter.endInterval("Agglomerative Hierarchical Clustering", ahcState)
        return result
    }

    // MARK: - L2 Feature Normalization
    private func normalizeFeatures(_ features: [[Double]], dimension: Int) -> [Double] {
        var normalized = [Double](repeating: 0, count: features.count * dimension)

        for (rowIndex, vector) in features.enumerated() {
            precondition(vector.count == dimension, "All feature vectors must share the same dimension")

            var norm: Double = 0
            vector.withUnsafeBufferPointer { pointer in
                vDSP_dotprD(
                    pointer.baseAddress!,
                    1,
                    pointer.baseAddress!,
                    1,
                    &norm,
                    vDSP_Length(dimension)
                )
            }

            let scale = norm > 0 ? 1.0 / sqrt(norm) : 0
            var mutableScale = scale
            vector.withUnsafeBufferPointer { source in
                normalized.withUnsafeMutableBufferPointer { destination in
                    vDSP_vsmulD(
                        source.baseAddress!,
                        1,
                        &mutableScale,
                        destination.baseAddress!.advanced(by: rowIndex * dimension),
                        1,
                        vDSP_Length(dimension)
                    )
                }
            }
        }

        return normalized
    }

    // MARK: - Similarity-to-Distance Conversion
    private func convertThresholdToDistance(_ similarity: Double) -> Double {
        guard !similarity.isNaN else { return Double.infinity }
        if similarity < -1.0 || similarity > 1.0 {
            logger.debug("Clustering threshold \(similarity) outside cosine range; clamping to [-1, 1]")
        }
        let clamped = max(-1.0, min(1.0, similarity))
        return sqrt(max(0, 2.0 - 2.0 * clamped))
    }

    // MARK: - Dendrogram Parsing & Threshold-Based Cluster Assignment
    private func assignmentsFromDendrogram(
        _ dendrogram: [Double],
        count: Int,
        distanceThreshold: Double
    ) -> [Int] {
        guard count > 0 else { return [] }
        if count == 1 {
            return [0]
        }

        let totalNodes = count * 2 - 1
        var leftChild = [Int](repeating: -1, count: totalNodes)
        var rightChild = [Int](repeating: -1, count: totalNodes)
        var nodeDistance = [Double](repeating: 0, count: totalNodes)

        for mergeIndex in 0..<(count - 1) {
            let base = mergeIndex * 4
            let left = Int(dendrogram[base])
            let right = Int(dendrogram[base + 1])
            let dist = dendrogram[base + 2]
            let newNode = count + mergeIndex
            leftChild[newNode] = left
            rightChild[newNode] = right
            nodeDistance[newNode] = dist
        }

        let root = totalNodes - 1
        var assignments = [Int](repeating: -1, count: count)
        var stack = [root]
        var nextLabel = 0

        while let node = stack.popLast() {
            if node < 0 {
                continue
            }

            if node < count {
                if assignments[node] == -1 {
                    assignments[node] = nextLabel
                    nextLabel += 1
                }
                continue
            }

            let distance = nodeDistance[node]
            if distance <= distanceThreshold {
                let label = nextLabel
                nextLabel += 1
                var queue = [node]
                while let current = queue.popLast() {
                    if current < count {
                        assignments[current] = label
                    } else {
                        let left = leftChild[current]
                        let right = rightChild[current]
                        if left >= 0 { queue.append(left) }
                        if right >= 0 { queue.append(right) }
                    }
                }
            } else {
                let left = leftChild[node]
                let right = rightChild[node]
                if left >= 0 { stack.append(left) }
                if right >= 0 { stack.append(right) }
            }
        }

        for index in 0..<assignments.count where assignments[index] == -1 {
            assignments[index] = nextLabel
            nextLabel += 1
        }

        return assignments
    }

    // MARK: - Cluster ID Remapping
    private func remapClusterIds(_ assignments: [Int]) -> [Int] {
        var mapping: [Int: Int] = [:]
        var nextId = 0
        return assignments.map { original in
            if mapping[original] == nil {
                mapping[original] = nextId
                nextId += 1
            }
            return mapping[original]!
        }
    }
}
