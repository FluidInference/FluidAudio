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
/// Greedy AVERAGE-LINKAGE, pure and deterministic: repeatedly fold the two
/// clusters whose linkage — the mean cosine over ALL cross-cluster member pairs —
/// is highest, while it is ≥ `threshold`. This matches the Python reference
/// `clustering.merge_oversplit` (`sim[ix_(ai,aj)].mean()`), which is deliberately
/// NOT centroid-linkage: averaging over members is more robust to an off-centroid
/// outlier than comparing two means. Note the threshold does NOT transfer from the
/// old centroid-linkage tuning — average-linkage sits below centroid cosine for the
/// same pair, so the operating point is the reference's ~0.80, pending the 14-clip
/// A/B. This relabels the per-embedding cluster assignment directly, before
/// reconstruction.
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

        // 2. One group per cluster (a set of original cluster ids).
        var groups: [Set<Int>] = order.map { Set([$0]) }

        /// Average-linkage: mean cosine over every (member of g1) × (member of g2)
        /// pair. Members are already unit-normalized, so dot == cosine.
        func linkage(_ g1: Set<Int>, _ g2: Set<Int>) -> Double {
            let m1 = g1.flatMap { members[$0]! }
            let m2 = g2.flatMap { members[$0]! }
            guard !m1.isEmpty, !m2.isEmpty else { return -Double.greatestFiniteMagnitude }
            var sum = 0.0
            for a in m1 { for b in m2 { sum += dot(a, b) } }
            return sum / Double(m1.count * m2.count)
        }

        // 3. Greedy: fold the highest-linkage pair while ≥ threshold.
        let thr = Double(threshold)
        while groups.count > 1 {
            var bestI = -1
            var bestJ = -1
            var bestLink = -Double.greatestFiniteMagnitude
            for i in 0..<groups.count {
                for j in (i + 1)..<groups.count {
                    let s = linkage(groups[i], groups[j])
                    if s > bestLink {
                        bestLink = s
                        bestI = i
                        bestJ = j
                    }
                }
            }
            guard bestLink >= thr, bestI >= 0 else { break }
            groups[bestI].formUnion(groups[bestJ])
            groups.remove(at: bestJ)
        }

        // 4. Build oldCluster → canonicalCluster, then renumber contiguously.
        var canonical: [Int: Int] = [:]
        for group in groups {
            let rep = group.min()!
            for id in group { canonical[id] = rep }
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

    private static func dot(_ a: [Double], _ b: [Double]) -> Double {
        let d = min(a.count, b.count)
        var s = 0.0
        for k in 0..<d { s += a[k] * b[k] }
        return s
    }
}
