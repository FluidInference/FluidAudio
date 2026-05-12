import Foundation

/// Jieba's character-position HMM, ported as a standalone Viterbi
/// decoder over the four B/M/E/S states.
///
/// Used by `MandarinG2P.segment(_:)` as a *post-pass* over runs of
/// consecutive single-character lookups (i.e. characters the
/// forward-maximum-match phrase dictionary did not cover). For modern
/// proper nouns the FMM frequently misses — `特朗普`, `比亚迪`, `比特币`
/// — and falls back to per-char lookups, breaking word boundaries and
/// pushing polyphones onto their isolated-char readings instead of the
/// (correct) compound reading. The HMM recovers those boundaries by
/// scoring `argmax_path P(states | chars)` and reading off contiguous
/// `B…E` / `S` spans.
///
/// The decoder is deterministic and stateless — a single instance is
/// safe to share across threads/actors. Construction is cheap (no
/// model weights beyond the `MandarinJiebaHmmTables` reference).
public struct MandarinJiebaHmm: Sendable {

    private let tables: MandarinJiebaHmmTables

    /// Pre-computed list of valid predecessor states for each next
    /// state, mirroring `PrevStatus` in `jieba.finalseg`. Encoding the
    /// constraint that:
    ///   * `B` and `S` can only follow `E` or `S` (a word must end
    ///     before another word starts);
    ///   * `M` and `E` can only follow `B` or `M` (must be inside a
    ///     word that has already begun).
    /// Without these constraints the decoder happily emits `B B` /
    /// `E S` paths that produce nonsense splits.
    private static let allowedPredecessors: [JiebaHmmState: [JiebaHmmState]] = [
        .begin: [.end, .single],
        .middle: [.middle, .begin],
        .single: [.single, .end],
        .end: [.begin, .middle],
    ]

    public init(tables: MandarinJiebaHmmTables) {
        self.tables = tables
    }

    /// Run Viterbi on `text` and return the resulting word boundaries
    /// as substrings (preserving `Character` granularity).
    ///
    /// Behaviour notes:
    ///   * Empty input → empty output.
    ///   * Single-char input → one one-char word (HMM bypassed).
    ///   * The output concatenates back to the input verbatim — useful
    ///     invariant to assert in tests.
    public func segment(_ text: String) -> [String] {
        let chars = Array(text)
        switch chars.count {
        case 0: return []
        case 1: return [String(chars[0])]
        default: break
        }

        let states = JiebaHmmState.allCases
        let stateCount = states.count

        // Viterbi tables: viterbi[t][s] = best log-prob to reach state
        // s at position t; path[t][s] = the predecessor state on that
        // path.
        var viterbi = Array(
            repeating: Array(repeating: -Double.infinity, count: stateCount),
            count: chars.count)
        var path = Array(
            repeating: Array(repeating: 0, count: stateCount),
            count: chars.count)

        // t = 0: only `begin` and `single` can start a sentence
        // (`middle` / `end` require a predecessor inside a word). Match
        // jieba's `start` matrix semantics directly — start[middle] /
        // start[end] are -inf upstream anyway, but the explicit
        // short-circuit keeps the path tables honest.
        let firstEmit = emission(for: chars[0])
        for (idx, state) in states.enumerated() {
            switch state {
            case .begin, .single:
                viterbi[0][idx] = tables.start[idx] + firstEmit[idx]
            case .middle, .end:
                viterbi[0][idx] = -Double.infinity
            }
            path[0][idx] = idx
        }

        // t > 0: standard Viterbi, restricted to the allowed
        // predecessors per next state.
        for t in 1..<chars.count {
            let emit = emission(for: chars[t])
            for (toIdx, toState) in states.enumerated() {
                let predecessors = Self.allowedPredecessors[toState] ?? states
                var bestProb = -Double.infinity
                var bestFrom = predecessors.first?.rawValue ?? 0
                for from in predecessors {
                    let prevProb = viterbi[t - 1][from.rawValue]
                    let candidate =
                        prevProb + tables.trans[from.rawValue][toIdx] + emit[toIdx]
                    if candidate > bestProb {
                        bestProb = candidate
                        bestFrom = from.rawValue
                    }
                }
                viterbi[t][toIdx] = bestProb
                path[t][toIdx] = bestFrom
            }
        }

        // Backtrack from the more probable terminal state. Only `end`
        // and `single` are valid sentence-final states; picking the
        // better one mirrors `jieba.finalseg.viterbi`.
        let lastIdx = chars.count - 1
        let endProb = viterbi[lastIdx][JiebaHmmState.end.rawValue]
        let singleProb = viterbi[lastIdx][JiebaHmmState.single.rawValue]
        var current =
            endProb >= singleProb
            ? JiebaHmmState.end.rawValue : JiebaHmmState.single.rawValue

        var assignments = Array(repeating: 0, count: chars.count)
        assignments[lastIdx] = current
        var t = lastIdx
        while t > 0 {
            current = path[t][current]
            assignments[t - 1] = current
            t -= 1
        }

        // Read off contiguous begin…end and single spans into word
        // slices. Anything unexpected (e.g. middle without a preceding
        // begin due to a degenerate transition matrix in tests) is
        // conservatively closed.
        var words: [String] = []
        var wordStart = 0
        for i in 0..<chars.count {
            let state = JiebaHmmState(rawValue: assignments[i]) ?? .single
            switch state {
            case .single:
                words.append(String(chars[i]))
                wordStart = i + 1
            case .end:
                words.append(String(chars[wordStart...i]))
                wordStart = i + 1
            case .begin, .middle:
                continue
            }
        }
        // Tail flush: if the path ended mid-word (begin or middle
        // without a following end), emit the remainder as a single
        // word so the output still concatenates back to the input.
        if wordStart < chars.count {
            words.append(String(chars[wordStart..<chars.count]))
        }
        return words
    }

    /// Look up the emission row for `ch`, falling back to the uniform
    /// "unknown character" log-prob row when the char isn't in the
    /// trained vocabulary.
    @inline(__always)
    private func emission(for ch: Character) -> [Double] {
        if let row = tables.emit[ch] { return row }
        return Array(
            repeating: MandarinJiebaHmmTables.unknownCharLogProb,
            count: JiebaHmmState.allCases.count)
    }
}
