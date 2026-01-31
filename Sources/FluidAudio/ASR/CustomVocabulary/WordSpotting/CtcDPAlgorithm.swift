/// Pure dynamic programming algorithms for CTC keyword spotting.
///
/// Extracted from `CtcKeywordSpotter` so that the DP logic can be tested
/// independently of CoreML model loading. All methods are static and
/// take only primitive inputs (`[[Float]]` log-prob matrices and `[Int]`
/// token ID arrays).
///
/// Ported from NeMo `ctc_word_spotter.py:ctc_word_spot`.
enum CtcDPAlgorithm {

    /// Wildcard token ID: represents "*" that matches anything at zero cost.
    static let wildcardTokenId = ContextBiasingConstants.wildcardTokenId

    // MARK: - Core DP

    /// Core DP table construction shared by all CTC word spotting variants.
    ///
    /// - Parameters:
    ///   - logProbs: CTC log-probabilities `[T, vocab_size]`
    ///   - keywordTokens: Token IDs for the keyword (may include `wildcardTokenId`)
    /// - Returns: Tuple `(dp, backtrack, lastMatch)` where:
    ///   - `dp[t][n]` = best score to match first `n` tokens by time `t`
    ///   - `backtrack[t][n]` = start frame for the best alignment ending at `t`
    ///   - `lastMatch[t][n]` = frame where token `n` was last matched
    static func fillDPTable(
        logProbs: [[Float]],
        keywordTokens: [Int]
    ) -> (dp: [[Float]], backtrack: [[Int]], lastMatch: [[Int]]) {
        let T = logProbs.count
        let N = keywordTokens.count

        var dp = Array(
            repeating: Array(repeating: -Float.greatestFiniteMagnitude, count: N + 1),
            count: T + 1
        )
        var backtrack = Array(
            repeating: Array(repeating: 0, count: N + 1),
            count: T + 1
        )
        var lastMatch = Array(
            repeating: Array(repeating: 0, count: N + 1),
            count: T + 1
        )

        // Keyword of length 0 has score 0 at any time
        for t in 0...T {
            dp[t][0] = 0.0
        }

        for t in 1...T {
            let frame = logProbs[t - 1]

            for n in 1...N {
                let tokenId = keywordTokens[n - 1]

                // Wildcard token: matches any symbol at zero cost
                if tokenId == wildcardTokenId {
                    let wildcardSkip = dp[t - 1][n - 1]
                    let wildcardStay = dp[t - 1][n]
                    let wildcardScore = max(wildcardSkip, wildcardStay)
                    dp[t][n] = wildcardScore
                    if wildcardScore == wildcardSkip {
                        backtrack[t][n] = t - 1
                        lastMatch[t][n] = t
                    } else {
                        backtrack[t][n] = backtrack[t - 1][n]
                        lastMatch[t][n] = lastMatch[t - 1][n]
                    }
                    continue
                }

                if tokenId < 0 || tokenId >= frame.count {
                    continue
                }

                let tokenScore = frame[tokenId]

                // Option 1: match this token at this timestep
                let matchScore = max(
                    dp[t - 1][n - 1] + tokenScore,
                    dp[t - 1][n] + tokenScore
                )

                // Option 2: skip this timestep
                let skipScore = dp[t - 1][n]

                if matchScore > skipScore {
                    dp[t][n] = matchScore
                    backtrack[t][n] = t - 1
                    lastMatch[t][n] = t
                } else {
                    dp[t][n] = skipScore
                    backtrack[t][n] = backtrack[t - 1][n]
                    lastMatch[t][n] = lastMatch[t - 1][n]
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
    /// - Returns: `(score, startFrame, endFrame)` in global frame coordinates
    static func ctcWordSpotConstrained(
        logProbs: [[Float]],
        keywordTokens: [Int],
        searchStartFrame: Int,
        searchEndFrame: Int
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

        let (dp, backtrack, lastMatch) = fillDPTable(logProbs: windowLogProbs, keywordTokens: keywordTokens)

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
    /// - Returns: Array of `(score, startFrame, endFrame)` tuples
    static func ctcWordSpotMultiple(
        logProbs: [[Float]],
        keywordTokens: [Int],
        minScore: Float = ContextBiasingConstants.defaultMinSpotterScore,
        mergeOverlap: Bool = true
    ) -> [(score: Float, startFrame: Int, endFrame: Int)] {
        let T = logProbs.count
        let N = keywordTokens.count

        if N == 0 || T == 0 {
            return []
        }

        let (dp, backtrack, lastMatch) = fillDPTable(logProbs: logProbs, keywordTokens: keywordTokens)

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
