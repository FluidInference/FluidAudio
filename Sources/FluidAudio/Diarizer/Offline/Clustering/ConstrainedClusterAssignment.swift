import Foundation

/// pyannote-parity constrained assignment (`constrained_argmax`): within a
/// segmentation chunk, distinct local speakers must map to distinct clusters.
/// Plain per-embedding argmax lets two speakers who share a chunk both snap to
/// the same centroid, silently absorbing one speaker's turns into the other's.
@available(macOS 14.0, iOS 17.0, *)
enum ConstrainedClusterAssignment {

    /// Assigns each embedding to a cluster, maximizing total similarity per
    /// chunk under the one-cluster-per-local-speaker constraint.
    ///
    /// - Parameters:
    ///   - scores: Per-embedding similarity against each cluster centroid
    ///     (higher is better; non-finite values rank below all finite ones).
    ///   - chunkIndices: Chunk index of each embedding (parallel to `scores`).
    /// - Returns: Cluster index per embedding, or `-2` when a chunk holds more
    ///   local speakers than there are clusters and the slot is dropped
    ///   (matching pyannote, which leaves such slots unassigned).
    static func assign(scores: [[Double]], chunkIndices: [Int]) -> [Int] {
        precondition(
            scores.count == chunkIndices.count,
            "scores and chunkIndices must be parallel arrays"
        )
        var assignments = [Int](repeating: -2, count: scores.count)

        var rowsByChunk: [Int: [Int]] = [:]
        for (row, chunk) in chunkIndices.enumerated() {
            rowsByChunk[chunk, default: []].append(row)
        }

        for rows in rowsByChunk.values {
            let chunkScores = rows.map { scores[$0] }
            let assigned = HungarianAssignment.maxScoreAssignment(scores: chunkScores)
            for (localIndex, row) in rows.enumerated() {
                let cluster = assigned[localIndex]
                assignments[row] = cluster >= 0 ? cluster : -2
            }
        }

        return assignments
    }
}
