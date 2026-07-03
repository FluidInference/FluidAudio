import Foundation

/// Pure decision logic for the zero-vote re-embed post-pass.
///
/// During reconstruction, aggregated timeline frames collect per-cluster "votes"
/// (activation sums) from every covering window. When the active local speaker slot
/// received no embedding in any covering window (assignment −2 everywhere), a
/// speech-active frame ends up with zero votes across all clusters and the per-frame
/// cluster ranking tie-breaks it arbitrarily to cluster 0 — silently absorbing whole
/// speaker turns into the surrounding speaker's segment.
///
/// This type detects maximal contiguous zero-vote runs and, given a fresh embedding
/// extracted over the run's exact audio span, picks the closest speaker centroid.
/// Because zero votes means there is no incumbent evidence at all, assignment ignores
/// any margin — the best centroid simply wins.
///
/// All CoreML-dependent extraction lives in `OfflineEmbeddingExtractor.embedSpan`;
/// keeping the decision pure makes it unit-testable without models.
enum ZeroVoteReembedder {

    /// Outcome of assigning a re-embedded zero-vote run to a speaker centroid.
    struct Assignment: Equatable {
        /// Zero-based cluster index of the winning centroid.
        let cluster: Int
        /// Cosine similarity between the span embedding and each centroid, by index.
        let cosines: [Double]
    }

    /// Detect maximal contiguous zero-vote runs over the aggregated frame timeline.
    ///
    /// A frame belongs to a run when it is speech-active (`speakerCountPerFrame > 0`)
    /// and its vote sums are zero for every cluster. Runs are naturally bounded by
    /// non-speech gaps and voted frames. Only runs spanning at least
    /// `minDurationSeconds` of audio are returned.
    ///
    /// - Parameters:
    ///   - speakerCountPerFrame: Estimated speaker count for each aggregated frame.
    ///   - activationSums: Per-frame, per-cluster vote sums (`[frame][cluster]`).
    ///   - frameDuration: Duration of one aggregated frame in seconds.
    ///   - minDurationSeconds: Minimum run duration; shorter runs are dropped.
    /// - Returns: Frame-index ranges of qualifying runs, in ascending order.
    static func detectRuns(
        speakerCountPerFrame: [Int],
        activationSums: [[Double]],
        frameDuration: Double,
        minDurationSeconds: Double
    ) -> [Range<Int>] {
        guard frameDuration > 0 else { return [] }
        let frameCount = min(speakerCountPerFrame.count, activationSums.count)
        guard frameCount > 0 else { return [] }

        var runs: [Range<Int>] = []
        var runStart: Int? = nil

        for frame in 0..<frameCount {
            // Single-speaker frames only: an overlap frame (2+ active speakers) with zero
            // votes must keep the existing behavior — re-embedding would collapse the
            // overlap to one speaker and silently change overlap semantics.
            let isZeroVote =
                speakerCountPerFrame[frame] == 1
                && activationSums[frame].allSatisfy { $0 == 0 }

            if isZeroVote {
                if runStart == nil {
                    runStart = frame
                }
            } else if let start = runStart {
                runs.append(start..<frame)
                runStart = nil
            }
        }
        if let start = runStart {
            runs.append(start..<frameCount)
        }

        return runs.filter { run in
            Double(run.count) * frameDuration >= minDurationSeconds
        }
    }

    /// Assign a re-extracted span embedding to the closest speaker centroid.
    ///
    /// Unlike `ShortSegmentRelabeler.decision`, there is no incumbent label to defend:
    /// the best centroid wins regardless of margin. Ties resolve to the lowest cluster
    /// index for determinism.
    ///
    /// - Returns: The winning cluster plus all per-centroid cosines, or `nil` when the
    ///   embedding is empty/non-finite or the centroids are empty/dimension-mismatched
    ///   (callers should fall back to the existing tie-break behavior).
    static func assignment(
        embedding: [Double],
        centroids: [[Double]]
    ) -> Assignment? {
        guard !embedding.isEmpty, !centroids.isEmpty else { return nil }
        guard embedding.allSatisfy({ $0.isFinite }) else { return nil }

        var cosines: [Double] = []
        cosines.reserveCapacity(centroids.count)
        var bestIndex = -1
        var bestCosine = -Double.infinity

        for (index, centroid) in centroids.enumerated() {
            guard centroid.count == embedding.count else { return nil }
            let cosine = Self.cosineSimilarity(embedding, centroid)
            guard cosine.isFinite else { return nil }
            cosines.append(cosine)
            if cosine > bestCosine {
                bestCosine = cosine
                bestIndex = index
            }
        }

        guard bestIndex >= 0 else { return nil }
        return Assignment(cluster: bestIndex, cosines: cosines)
    }
    /// Cosine similarity between an embedding and a centroid. Scale-invariant; returns
    /// a non-finite value when either vector has zero magnitude.
    static func cosineSimilarity(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for index in a.indices {
            let x = a[index]
            let y = b[index]
            dot += x * y
            normA += x * x
            normB += y * y
        }
        return dot / ((normA * normB).squareRoot())
    }
}
