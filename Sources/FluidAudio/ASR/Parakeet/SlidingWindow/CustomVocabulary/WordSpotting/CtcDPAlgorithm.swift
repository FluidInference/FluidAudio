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

    /// Terminal keyword states shared by all CTC word spotting variants.
    ///
    /// The internal table is built on the blank-expanded symbol sequence of
    /// length `2N + 1`. Three transitions are evaluated per state:
    ///   - **stay** at `s`: adds `log p_t[symbol_s]` (blank emission cost
    ///     for stays in blank states; token emission for stays in token
    ///     states).
    ///   - **advance** from `s-1`: standard CTC step.
    ///   - **skip blank** from `s-2`: only when `canSkipBlank` permits it.
    ///
    /// Each frame records the better of ending on the keyword's final token
    /// or on its following blank. Earlier token states are needed only by
    /// the next frame, so two working rows suffice. Retained storage is
    /// O(T + N); scores, free starts and predecessor tie-breaking are unchanged.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword (may include `wildcardTokenId`)
    ///   - blankId: Vocabulary index of the CTC blank token
    /// - Returns: Three arrays of length `T+1`:
    ///   - `scores[t]` = best raw log-prob score for consuming the keyword
    ///     by frame `t` (sum of emission log-probs along the path,
    ///     **including** blank emissions).
    ///   - `starts[t]` = inferred keyword start frame (0-indexed) for
    ///     the best path ending at `scores[t]`.
    ///   - `ends[t]` = frame at which the most recent non-blank
    ///     token was emitted along that path.
    ///
    /// > Note: Raw scores are **larger in magnitude** than the previous
    /// > token-only DP because blank emission costs are now included.
    /// > Callers using token-count normalization (`/ N`) will see
    /// > systematically more negative per-token averages; tune
    /// > `defaultMinSpotterScore` and `defaultMinVocabCtcScore`
    /// > accordingly. A per-frame normalization
    /// > (`raw / (endFrame - startFrame)`) is more stable.
    static func terminalStates(
        logProbs: [[Float]],
        keywordTokens: [Int],
        blankId: Int = ContextBiasingConstants.defaultBlankId
    ) -> (scores: [Float], starts: [Int], ends: [Int]) {
        let T = logProbs.count
        let N = keywordTokens.count
        let neg = -Float.greatestFiniteMagnitude

        var scores = Array(repeating: N == 0 ? 0 : neg, count: T + 1)
        var starts = Array(repeating: 0, count: T + 1)
        var ends = Array(repeating: 0, count: T + 1)

        if N == 0 { return (scores, starts, ends) }

        let s = buildExpandedSequence(keywordTokens)
        let sLen = s.count  // = 2N + 1

        // Expanded states depend only on the preceding frame. Keep two rows
        // and retain only the final token/blank pair at each frame.
        var previousScores = Array(repeating: neg, count: sLen)
        var currentScores = Array(repeating: neg, count: sLen)
        var previousStarts = Array(repeating: 0, count: sLen)
        var currentStarts = Array(repeating: 0, count: sLen)
        var previousLastTokens = Array(repeating: 0, count: sLen)
        var currentLastTokens = Array(repeating: 0, count: sLen)
        // s = 0 is the initial blank; free-start convention says any frame
        // can be the start of the keyword, so dp at the initial state is 0
        // and the candidate start frame is the current frame index.
        previousScores[0] = 0

        for t in 1...T {
            currentScores[0] = 0
            currentStarts[0] = t
            currentLastTokens[0] = 0
            let frame = logProbs[t - 1]
            for sIdx in 1..<sLen {
                let sym = s[sIdx]
                let emitLogProb = emissionLogProb(symbol: sym, frame: frame, blankId: blankId)
                let isWildcard: Bool = { if case .wildcard = sym { return true } else { return false } }()
                let isToken: Bool = { if case .token = sym { return true } else { return false } }()
                // Wildcards are zero-cost regardless of underlying frame log-probs.
                let added: Float = isWildcard ? 0 : emitLogProb

                let stay = previousScores[sIdx]
                let advance = previousScores[sIdx - 1]
                let skipBlank = canSkipBlank(s, at: sIdx) ? previousScores[sIdx - 2] : neg

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
                    currentScores[sIdx] = neg
                    currentStarts[sIdx] = 0
                    currentLastTokens[sIdx] = 0
                    continue
                }

                currentScores[sIdx] = bestPred + added
                let isMatchFrame = isToken || isWildcard

                switch predKind {
                case 0:
                    currentStarts[sIdx] = previousStarts[sIdx]
                    currentLastTokens[sIdx] = isMatchFrame ? t : previousLastTokens[sIdx]
                case 1:
                    if sIdx == 1 {
                        // First non-blank symbol: keyword starts at this frame.
                        currentStarts[sIdx] = t - 1
                    } else {
                        currentStarts[sIdx] = previousStarts[sIdx - 1]
                    }
                    currentLastTokens[sIdx] = isMatchFrame ? t : previousLastTokens[sIdx - 1]
                default:
                    // Skip-blank: predecessor is at sIdx - 2, which by
                    // canSkipBlank must be a token (or wildcard). For
                    // sIdx = 2 we still inherit start from sIdx - 2 = 0,
                    // which records the candidate keyword-start frame.
                    currentStarts[sIdx] = previousStarts[sIdx - 2]
                    currentLastTokens[sIdx] = isMatchFrame ? t : previousLastTokens[sIdx - 2]
                }
            }
            let sTok = 2 * N - 1
            let sBlank = 2 * N
            let scTok = currentScores[sTok]
            let scBlank = currentScores[sBlank]
            if scTok >= scBlank {
                scores[t] = scTok
                starts[t] = currentStarts[sTok]
                ends[t] = currentLastTokens[sTok]
            } else {
                scores[t] = scBlank
                starts[t] = currentStarts[sBlank]
                ends[t] = currentLastTokens[sBlank]
            }
            swap(&previousScores, &currentScores)
            swap(&previousStarts, &currentStarts)
            swap(&previousLastTokens, &currentLastTokens)
        }

        return (scores, starts, ends)
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

        let (scores, starts, ends) = terminalStates(
            logProbs: windowLogProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        var bestEnd = 0
        var bestScore = -Float.greatestFiniteMagnitude

        for t in N...windowT {
            if scores[t] > bestScore {
                bestScore = scores[t]
                bestEnd = t
            }
        }

        let bestStart = starts[bestEnd]
        let actualEndFrame = ends[bestEnd]

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

        let (scores, starts, ends) = terminalStates(
            logProbs: logProbs,
            keywordTokens: keywordTokens,
            blankId: blankId
        )

        let wildcardFreeCount = nonWildcardCount(keywordTokens)
        let normFactor = wildcardFreeCount > 0 ? Float(wildcardFreeCount) : 1.0

        var candidates: [(score: Float, startFrame: Int, endFrame: Int)] = []

        guard T >= N else { return [] }

        for t in N...T {
            let rawScore = scores[t]
            let normalizedScore = rawScore / normFactor

            let prevScore = t > N ? scores[t - 1] / normFactor : -Float.greatestFiniteMagnitude
            let nextScore = t < T ? scores[t + 1] / normFactor : -Float.greatestFiniteMagnitude

            let isLocalMax = normalizedScore >= prevScore && normalizedScore > nextScore
            let meetsThreshold = normalizedScore >= minScore

            if isLocalMax && meetsThreshold {
                let startFrame = starts[t]
                let actualEndFrame = ends[t]
                candidates.append((score: normalizedScore, startFrame: startFrame, endFrame: actualEndFrame))
            }
        }

        if candidates.isEmpty {
            var bestEnd = 0
            var bestScore = -Float.greatestFiniteMagnitude
            for t in N...T {
                let normalizedScore = scores[t] / normFactor
                if normalizedScore > bestScore {
                    bestScore = normalizedScore
                    bestEnd = t
                }
            }
            if bestScore >= minScore {
                let startFrame = starts[bestEnd]
                let actualEndFrame = ends[bestEnd]
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
