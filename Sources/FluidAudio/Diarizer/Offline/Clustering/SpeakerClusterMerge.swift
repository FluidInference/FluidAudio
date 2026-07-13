import Foundation

/// Agglomerative post-merge of over-split NME-SC clusters.
///
/// NME-SC's eigengap `k`-selection over-segments: it tends to slice a single
/// speaker into several temporal sub-clusters because the voice drifts slowly
/// over a recording. On near-field 1–2 speaker capture the within-speaker
/// cross-cluster cosine sits at ~0.85–0.96 while true cross-speaker cosine is
/// ~0.15–0.55, so a single threshold in the ~0.55–0.80 gap recovers the real
/// speaker count. Well-separated speakers (large centroid gap) are never touched.
///
/// Greedy centroid-linkage, pure and deterministic: repeatedly fold the two
/// clusters whose L2-normalized centroids are most similar while that similarity
/// is ≥ `threshold`, recomputing the merged centroid from all member embeddings
/// each step. This is the library-level twin of the app's `SpeakerClusterMerge`
/// (which remaps segment speaker-ids); here it relabels the per-embedding cluster
/// assignment directly, before reconstruction.
enum SpeakerClusterMerge {

    /// Fold over-split clusters together.
    /// - Parameters:
    ///   - embeddings: per-window embedding (parallel to `labels`).
    ///   - labels: per-window cluster id from NME-SC (any integers).
    ///   - threshold: merge while the most-similar centroid pair has cosine ≥ this.
    /// - Returns: new per-window labels, renumbered to a contiguous `0..<k` in
    ///   order of first appearance. Fewer than two clusters → labels unchanged
    ///   (renumbered). Returns the input untouched if lengths mismatch.
    static func mergedLabels(
        embeddings: [[Double]],
        labels: [Int],
        threshold: Float
    ) -> [Int] {
        guard embeddings.count == labels.count, !labels.isEmpty else { return labels }

        // 1. Group L2-normalized embeddings by cluster (stable first-seen order).
        var members: [Int: [[Double]]] = [:]
        var order: [Int] = []
        for (emb, cid) in zip(embeddings, labels) {
            guard let n = normed(emb) else { continue }
            if members[cid] == nil { order.append(cid) }
            members[cid, default: []].append(n)
        }
        guard order.count > 1 else { return renumber(labels) }

        // 2. One group per cluster; each carries its member ids + centroid.
        var groups: [(ids: Set<Int>, centroid: [Double])] =
            order.map { (Set([$0]), centroidOf(members[$0]!)) }

        // 3. Greedy: fold the most-similar pair while ≥ threshold.
        let thr = Double(threshold)
        while groups.count > 1 {
            var bestI = -1
            var bestJ = -1
            var bestSim = -Double.greatestFiniteMagnitude
            for i in 0..<groups.count {
                for j in (i + 1)..<groups.count {
                    let s = dot(groups[i].centroid, groups[j].centroid)
                    if s > bestSim {
                        bestSim = s
                        bestI = i
                        bestJ = j
                    }
                }
            }
            guard bestSim >= thr, bestI >= 0 else { break }
            let mergedIds = groups[bestI].ids.union(groups[bestJ].ids)
            let all = mergedIds.flatMap { members[$0]! }  // recompute from all members
            groups[bestI] = (mergedIds, centroidOf(all))
            groups.remove(at: bestJ)
        }

        // 4. Build oldCluster → canonicalCluster, then renumber contiguously.
        var canonical: [Int: Int] = [:]
        for group in groups {
            let rep = group.ids.min()!
            for id in group.ids { canonical[id] = rep }
        }
        let remapped = labels.map { canonical[$0] ?? $0 }
        return renumber(remapped)
    }

    // MARK: - Helpers

    /// Renumber arbitrary labels to a contiguous `0..<k` in first-appearance order.
    private static func renumber(_ labels: [Int]) -> [Int] {
        var map: [Int: Int] = [:]
        var next = 0
        return labels.map { l in
            if let m = map[l] { return m }
            map[l] = next
            defer { next += 1 }
            return next
        }
    }

    /// L2-normalize; `nil` for empty or ~zero-energy vectors (garbage masks).
    private static func normed(_ v: [Double]) -> [Double]? {
        guard !v.isEmpty else { return nil }
        var sq = 0.0
        for x in v { sq += x * x }
        let n = sq.squareRoot()
        guard n > 1e-9 else { return nil }
        return v.map { $0 / n }
    }

    /// Mean of already-normalized vectors, renormalized to unit length.
    private static func centroidOf(_ vs: [[Double]]) -> [Double] {
        let d = vs[0].count
        var c = [Double](repeating: 0, count: d)
        for v in vs where v.count == d {
            for k in 0..<d { c[k] += v[k] }
        }
        return normed(c) ?? c
    }

    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        let d = min(a.count, b.count)
        var s = 0.0
        for k in 0..<d { s += a[k] * b[k] }
        return s
    }
}
