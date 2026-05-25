/// Pure dynamic programming algorithms for CTC keyword spotting.
///
/// Extracted from `CtcKeywordSpotter` so that the DP logic can be tested
/// independently of CoreML model loading. All methods are static and
/// take only primitive inputs (`[[Float]]` log-prob matrices and `[Int]`
/// token ID arrays).
///
/// Implements the CTC-WS dynamic program from NeMo's `ctc_word_spotter.py`
/// (arXiv:2406.07096). Unlike a naive token-only DP, this version operates
/// on the **blank-expanded symbol sequence** `[B, t1, B, t2, ..., tN, B]`
/// and accumulates blank emission log-probs along stay/within-token paths.
/// This is what makes the score probabilistically meaningful and what
/// correctly enforces a blank between repeated tokens.
enum CtcDPAlgorithm {

    /// Wildcard token ID: represents "*" that matches anything at zero cost.
    static let wildcardTokenId = ContextBiasingConstants.wildcardTokenId

    // MARK: - Expanded symbol helpers

    /// Symbols on the blank-expanded CTC alignment graph.
    private enum ExpandedSymbol {
        case blank
        case token(Int)
        case wildcard
    }

    /// Build the blank-expanded symbol sequence `[B, t1, B, t2, ..., tN, B]`
    /// of length `2N + 1` from the keyword token sequence.
    private static func buildExpandedSequence(_ keywordTokens: [Int]) -> [ExpandedSymbol] {
        var s: [ExpandedSymbol] = []
        s.reserveCapacity(2 * keywordTokens.count + 1)
        for id in keywordTokens {
            s.append(.blank)
            s.append(id == wildcardTokenId ? .wildcard : .token(id))
        }
        s.append(.blank)
        return s
    }

    /// Emission log-probability for a symbol at a given frame.
    /// - Wildcard symbols emit zero (free match).
    /// - Out-of-range token / blank IDs emit zero (caller's vocab does not
    ///   include the symbol — same convention as the previous DP, which
    ///   simply skipped invalid tokens).
    @inline(__always)
    private static func emissionLogProb(
        symbol: ExpandedSymbol,
        frame: [Float],
        blankId: Int
    ) -> Float {
        switch symbol {
        case .blank:
            return blankId >= 0 && blankId < frame.count ? frame[blankId] : 0
        case .token(let id):
            return id >= 0 && id < frame.count ? frame[id] : -Float.greatestFiniteMagnitude
        case .wildcard:
            return 0
        }
    }

    /// True iff the DP may transition from position `s-2` directly to `s`
    /// (skipping the intermediate blank). This is allowed only when `s` is
    /// a non-blank symbol distinct from `s-2`. Repeated tokens MUST go
    /// through the intervening blank — this is the CTC rule that the old
    /// DP violated.
    @inline(__always)
    private static func canSkipBlank(_ s: [ExpandedSymbol], at idx: Int) -> Bool {
        guard idx >= 2 else { return false }
        switch s[idx] {
        case .blank:
            return false
        case .token(let cur):
            if case .token(let prev) = s[idx - 2], prev == cur { return false }
            return true
        case .wildcard:
            if case .wildcard = s[idx - 2] { return false }
            return true
        }
    }

    // MARK: - Core DP

    /// Core DP table construction shared by all CTC word spotting variants.
    ///
    /// The internal table is built on the blank-expanded symbol sequence of
    /// length `2N + 1`. Three transitions are evaluated per state:
    ///   - **stay** at `s`: adds `log p_t[symbol_s]` (blank emission cost
    ///     for stays in blank states; token emission for stays in token
    ///     states).
    ///   - **advance** from `s-1`: standard CTC step.
    ///   - **skip blank** from `s-2`: only when `canSkipBlank` permits it.
    ///
    /// The returned `dp[t][n]` is projected back to the public
    /// "n tokens consumed" view via
    /// `dp[t][n] = max(dpI[t][2n - 1], dpI[t][2n])`, i.e. the best of
    /// "ended on token n" or "ended on the blank after token n". Free start
    /// is preserved with `dp[t][0] = 0` for all `t`, matching the previous
    /// API semantics.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword (may include `wildcardTokenId`)
    ///   - blankId: Vocabulary index of the CTC blank token
    /// - Returns: `(dp, backtrack, lastMatch)` with public `[T+1][N+1]` shape:
    ///   - `dp[t][n]` = best raw log-prob score for consuming the first `n`
    ///     tokens by frame `t` (sum of emission log-probs along the path,
    ///     **including** blank emissions).
    ///   - `backtrack[t][n]` = inferred keyword start frame (0-indexed) for
    ///     the best path ending at `dp[t][n]`.
    ///   - `lastMatch[t][n]` = frame at which the most recent non-blank
    ///     token was emitted along that path.
    ///
    /// > Note: Raw scores are **larger in magnitude** than the previous
    /// > token-only DP because blank emission costs are now included.
    /// > Callers using token-count normalization (`/ N`) will see
    /// > systematically more negative per-token averages; tune
    /// > `defaultMinSpotterScore` and `defaultMinVocabCtcScore`
    /// > accordingly. A per-frame normalization
    /// > (`raw / (endFrame - startFrame)`) is more stable.
    static func fillDPTable(
        logProbs: [[Float]],
        keywordTokens: [Int],
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> (dp: [[Float]], backtrack: [[Int]], lastMatch: [[Int]]) {
        let T = logProbs.count
        let N = keywordTokens.count
        let neg = -Float.greatestFiniteMagnitude

        var dp = Array(repeating: Array(repeating: neg, count: N + 1), count: T + 1)
        var backtrack = Array(repeating: Array(repeating: 0, count: N + 1), count: T + 1)
        var lastMatch = Array(repeating: Array(repeating: 0, count: N + 1), count: T + 1)

        // Free start: matching zero tokens has score 0 at any frame.
        for t in 0...T { dp[t][0] = 0 }
        if N == 0 { return (dp, backtrack, lastMatch) }

        let s = buildExpandedSequence(keywordTokens)
        let sLen = s.count  // = 2N + 1

        // Internal DP on the expanded graph.
        var dpI = Array(repeating: Array(repeating: neg, count: sLen), count: T + 1)
        var startI = Array(repeating: Array(repeating: 0, count: sLen), count: T + 1)
        var lastTokI = Array(repeating: Array(repeating: 0, count: sLen), count: T + 1)
        // s = 0 is the initial blank; free-start convention says any frame
        // can be the start of the keyword, so dp at the initial state is 0
        // and the candidate start frame is the current frame index.
        for t in 0...T {
            dpI[t][0] = 0
            startI[t][0] = t
        }

        for t in 1...T {
            let frame = logProbs[t - 1]
            for sIdx in 1..<sLen {
                let sym = s[sIdx]
                let emitLogProb = emissionLogProb(symbol: sym, frame: frame, blankId: blankId)
                let isWildcard: Bool = { if case .wildcard = sym { return true } else { return false } }()
                let isToken: Bool = { if case .token = sym { return true } else { return false } }()
                // Wildcards are zero-cost regardless of underlying frame log-probs.
                let added: Float = isWildcard ? 0 : emitLogProb

                let stay = dpI[t - 1][sIdx]
                let advance = dpI[t - 1][sIdx - 1]
                let skipBlank = canSkipBlank(s, at: sIdx) ? dpI[t - 1][sIdx - 2] : neg

                var bestPred = stay
                var predKind = 0  // 0 = stay, 1 = advance, 2 = skip-blank
                if advance > bestPred {
                    bestPred = advance
                    predKind = 1
                }
                if skipBlank > bestPred {
                    bestPred = skipBlank
                    predKind = 2
                }

                if bestPred <= neg / 2 {
                    dpI[t][sIdx] = neg
                    continue
                }

                dpI[t][sIdx] = bestPred + added
                let isMatchFrame = isToken || isWildcard

                switch predKind {
                case 0:
                    startI[t][sIdx] = startI[t - 1][sIdx]
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx]
                case 1:
                    if sIdx == 1 {
                        // First non-blank symbol: keyword starts at this frame.
                        startI[t][sIdx] = t - 1
                    } else {
                        startI[t][sIdx] = startI[t - 1][sIdx - 1]
                    }
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx - 1]
                default:
                    // Skip-blank: predecessor is at sIdx - 2, which by
                    // canSkipBlank must be a token (or wildcard). For
                    // sIdx = 2 we still inherit start from sIdx - 2 = 0,
                    // which records the candidate keyword-start frame.
                    startI[t][sIdx] = startI[t - 1][sIdx - 2]
                    lastTokI[t][sIdx] = isMatchFrame ? t : lastTokI[t - 1][sIdx - 2]
                }
            }
        }

        // Project the expanded states back to the public n-tokens-consumed view.
        for t in 0...T {
            for n in 1...N {
                let sTok = 2 * n - 1
                let sBlank = 2 * n
                let scTok = sTok < sLen ? dpI[t][sTok] : neg
                let scBlank = sBlank < sLen ? dpI[t][sBlank] : neg
                if scTok >= scBlank {
                    dp[t][n] = scTok
                    backtrack[t][n] = startI[t][sTok]
                    lastMatch[t][n] = lastTokI[t][sTok]
                } else {
                    dp[t][n] = scBlank
                    backtrack[t][n] = startI[t][sBlank]
                    lastMatch[t][n] = lastTokI[t][sBlank]
                }
            }
        }

        return (dp, backtrack, lastMatch)
    }

    /// Count non-wildcard tokens for score normalization.
    static func nonWildcardCount(_ keywordTokens: [Int]) -> Int {
        keywordTokens.filter { $0 != wildcardTokenId }.count
    }

    // MARK: - Word Spotting

    /// Constrained CTC word spotting within a temporal window.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword
    ///   - searchStartFrame: Start of search window (inclusive)
    ///   - searchEndFrame: End of search window (exclusive)
    ///   - blankId: Vocabulary index of the CTC blank token
    /// - Returns: `(score, startFrame, endFrame)` in global frame coordinates.
    ///   `score` is normalized by the number of non-wildcard tokens — i.e.
    ///   the *per-token* average log-probability of the best alignment,
    ///   which now includes blank-emission costs along stay paths.
    static func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int,
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> (score: Float, startFrame: Int, endFrame: Int) {
        let T = logProbs.count
        let N = keywordTokens.count

        let clampedStart = max(0, searchStartFrame)
        let clampedEnd = min(T, searchEndFrame)

        if N == 0 || clampedEnd <= clampedStart {
            return (-Float.infinity, clampedStart, clampedStart)
        }

        let windowLogProbs = Array(logProbs[clampedStart..<clampedEnd])
        let windowT = windowLogProbs.count

        if windowT < N {
            return (-Float.infinity, clampedStart, clampedStart)
        }

        let (dp, backtrack, lastMatch) = fillDPTable(
            logProbs: windowLogProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        var bestEnd = 0
        var bestScore = -Float.greatestFiniteMagnitude

        for t in N...windowT {
            if dp[t][N] > bestScore {
                bestScore = dp[t][N]
                bestEnd = t
            }
        }

        let bestStart = backtrack[bestEnd][N]
        let actualEndFrame = lastMatch[bestEnd][N]

        let normFactor = nonWildcardCount(keywordTokens)
        let normalizedScore = normFactor > 0 ? bestScore / Float(normFactor) : bestScore

        let globalStart = clampedStart + bestStart
        let globalEnd = clampedStart + actualEndFrame

        return (normalizedScore, globalStart, globalEnd)
    }

    /// Find ALL occurrences of a keyword in the log-probabilities.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword
    ///   - minScore: Minimum normalized score threshold
    ///   - mergeOverlap: Whether to merge overlapping detections
    ///   - blankId: Vocabulary index of the CTC blank token
    /// - Returns: Array of `(score, startFrame, endFrame)` tuples
    static func ctcWordSpotMultiple(
        logProbs: [[Float]],
        keywordTokens: [Int],
        minScore: Float = ContextBiasingConstants.defaultMinSpotterScore,
        mergeOverlap: Bool = true,
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 {
            return []
        }

        let (dp, backtrack, lastMatch) = fillDPTable(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        let wildcardFreeCount = nonWildcardCount(keywordTokens)
        let normFactor = wildcardFreeCount > 0 ? Float(wildcardFreeCount) : 1.0

        var candidates: [(score: Float, startFrame: Int, endFrame: Int)] = []

        guard T >= N else { return [] }

        for t in N...T {
            let rawScore = dp[t][N]
            let normalizedScore = rawScore / normFactor

            let prevScore = t > N ? dp[t - 1][N] / normFactor : -Float.greatestFiniteMagnitude
            let nextScore = t < T ? dp[t + 1][N] / normFactor : -Float.greatestFiniteMagnitude

            let isLocalMax = normalizedScore >= prevScore && normalizedScore > nextScore
            let meetsThreshold = normalizedScore >= minScore

            if isLocalMax && meetsThreshold {
                let startFrame = backtrack[t][N]
                let actualEndFrame = lastMatch[t][N]
                candidates.append((score: normalizedScore, startFrame: startFrame, endFrame: actualEndFrame))
            }
        }

        if candidates.isEmpty {
            var bestEnd = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for t in N...T {
                let normalizedScore = dp[t][N] / normFactor
                if normalizedScore > bestScore {
                    bestScore = normalizedScore
                    bestEnd = t
                }
            }
            if bestScore >= minScore {
                let startFrame = backtrack[bestEnd][N]
                let actualEndFrame = lastMatch[bestEnd][N]
                candidates.append((score: bestScore, startFrame: startFrame, endFrame: actualEndFrame))
            }
        }

        guard mergeOverlap else { return candidates }

        let sorted = candidates.sorted { $0.startFrame < $1.startFrame }
        var merged: [(score: Float, startFrame: Int, endFrame: Int)] = []

        for candidate in sorted {
            if let last = merged.last {
                if candidate.startFrame <= last.endFrame {
                    var best = candidate.score > last.score ? candidate : last
                    best.endFrame = max(last.endFrame, candidate.endFrame)
                    merged[merged.count - 1] = best
                } else {
                    merged.append(candidate)
                }
            } else {
                merged.append(candidate)
            }
        }

        return merged
    }
}
