import Foundation

/// Cross-window speaker-permutation alignment for offline Sortformer.
///
/// The offline model processes independent 30.72s windows with no speaker cache, so window
/// `N`'s speaker columns are in an arbitrary order relative to the accumulated global timeline.
/// Each window overlaps the previous one by a fixed number of output frames; over that overlap
/// the same speakers are active in both, so we recover the permutation that best matches the
/// two windows' per-frame speaker activity and remap the new window into global speaker IDs.
///
/// Speaker count is small and fixed (4), so we brute-force all `numSpeakers!` bijections — 24
/// for 4 speakers — and keep the one maximizing summed activity agreement. No external solver.
enum SortformerSpeakerStitcher {

    /// Best bijection between a new window's speaker columns and the global speaker columns over
    /// the overlap region.
    ///
    /// - Parameters:
    ///   - global: Overlap-region probabilities already committed to the global timeline,
    ///     flat `[frames * numSpeakers]` (frame-major), in global speaker order.
    ///   - window: The new window's probabilities for the same overlap frames, flat
    ///     `[frames * numSpeakers]`, in the window's own speaker order.
    ///   - frames: Number of overlap frames compared.
    ///   - numSpeakers: Speaker-column count (4 for current models).
    /// - Returns: `mapping` of length `numSpeakers` where `mapping[windowSpeaker] == globalSpeaker`.
    ///   Identity (`[0, 1, ...]`) when there is nothing to align on.
    static func alignment(
        global: [Float],
        window: [Float],
        frames: Int,
        numSpeakers: Int
    ) -> [Int] {
        let identity = Array(0..<numSpeakers)
        guard frames > 0, numSpeakers > 0,
            global.count >= frames * numSpeakers,
            window.count >= frames * numSpeakers
        else {
            return identity
        }

        // correlation[g][w] = summed per-frame activity agreement between global speaker g and
        // window speaker w over the overlap. Higher = more likely the same physical speaker.
        var correlation = [[Float]](
            repeating: [Float](repeating: 0, count: numSpeakers), count: numSpeakers)
        for f in 0..<frames {
            let base = f * numSpeakers
            for g in 0..<numSpeakers {
                let gv = global[base + g]
                guard gv != 0 else { continue }
                for w in 0..<numSpeakers {
                    correlation[g][w] += gv * window[base + w]
                }
            }
        }

        // perm[g] = w : assign each global speaker the window column that maximizes total agreement.
        var bestPerm = identity
        var bestScore = -Float.greatestFiniteMagnitude
        var perm = identity
        permute(&perm, 0) { candidate in
            var score: Float = 0
            for g in 0..<numSpeakers {
                score += correlation[g][candidate[g]]
            }
            if score > bestScore {
                bestScore = score
                bestPerm = candidate
            }
        }

        // Invert perm[g] = w into mapping[w] = g (window column -> global column).
        var mapping = identity
        for g in 0..<numSpeakers {
            mapping[bestPerm[g]] = g
        }
        return mapping
    }

    /// Heap-free recursive permutation enumeration; `body` is called once per permutation.
    private static func permute(_ array: inout [Int], _ k: Int, _ body: ([Int]) -> Void) {
        if k == array.count {
            body(array)
            return
        }
        for i in k..<array.count {
            array.swapAt(k, i)
            permute(&array, k + 1, body)
            array.swapAt(k, i)
        }
    }
}
