import Foundation
import OSLog

/// Keyword spotter for hybrid model that uses CTC log-probs from shared encoder.
/// Since CTC and TDT share the same encoder, timestamps are aligned.
public struct HybridKeywordSpotter {
    private let logger = AppLogger(category: "HybridKeywordSpotter")
    private let vocabulary: [Int: String]
    private let reverseVocab: [String: Int]
    private let blankId: Int

    public init(vocabulary: [Int: String], blankId: Int) {
        self.vocabulary = vocabulary
        self.blankId = blankId

        // Build reverse vocabulary
        var reverse: [String: Int] = [:]
        for (id, token) in vocabulary {
            reverse[token] = id
        }
        self.reverseVocab = reverse
    }

    /// Spot keywords using CTC log-probs from hybrid model.
    /// Since these come from the shared encoder, timestamps match TDT output.
    public func spotKeywords(
        ctcLogProbs: [[Float]],
        frameDuration: Double,
        customVocabulary: CustomVocabularyContext
    ) -> [HybridKeywordDetection] {
        var detections: [HybridKeywordDetection] = []

        for term in customVocabulary.terms {
            // Get token IDs for this term
            let tokenIds: [Int]
            if let ctcIds = term.ctcTokenIds, !ctcIds.isEmpty {
                tokenIds = ctcIds
            } else {
                tokenIds = tokenize(term.text)
            }

            guard !tokenIds.isEmpty else { continue }

            // Run DP to find best match
            if let (score, startFrame, endFrame) = findBestMatch(
                logProbs: ctcLogProbs,
                tokenIds: tokenIds,
                blankId: blankId
            ) {
                let startTime = Double(startFrame) * frameDuration
                let endTime = Double(endFrame) * frameDuration

                detections.append(
                    HybridKeywordDetection(
                        term: term,
                        score: score,
                        startFrame: startFrame,
                        endFrame: endFrame,
                        startTime: startTime,
                        endTime: endTime
                    ))
            }
        }

        return detections
    }

    // MARK: - Private Methods

    private func tokenize(_ text: String) -> [Int] {
        // Simple greedy tokenization using vocabulary
        let normalized = text.lowercased()
        var result: [Int] = []
        var position = normalized.startIndex
        var isWordStart = true

        while position < normalized.endIndex {
            var matched = false
            let remaining = normalized.distance(from: position, to: normalized.endIndex)
            var matchLength = min(20, remaining)

            while matchLength > 0 {
                let endPos = normalized.index(position, offsetBy: matchLength)
                let substring = String(normalized[position..<endPos])

                // Try with SentencePiece prefix for word start
                let withPrefix = isWordStart ? "▁" + substring : substring

                if let tokenId = reverseVocab[withPrefix] {
                    result.append(tokenId)
                    position = endPos
                    isWordStart = false
                    matched = true
                    break
                } else if let tokenId = reverseVocab[substring] {
                    result.append(tokenId)
                    position = endPos
                    isWordStart = false
                    matched = true
                    break
                }

                matchLength -= 1
            }

            if !matched {
                // Skip character
                let char = normalized[position]
                position = normalized.index(after: position)
                isWordStart = char == " "
            }
        }

        return result
    }

    private func findBestMatch(
        logProbs: [[Float]],
        tokenIds: [Int],
        blankId: Int
    ) -> (score: Float, startFrame: Int, endFrame: Int)? {
        let T = logProbs.count
        let K = tokenIds.count

        guard T > 0, K > 0 else { return nil }

        // DP: dp[t][k] = (score, startFrame) for best alignment ending at frame t, having emitted k tokens
        var dp: [[(score: Float, startFrame: Int)]] = Array(
            repeating: Array(repeating: (-.infinity, 0), count: K + 1),
            count: T
        )

        // Initialize: can start at any frame
        for t in 0..<T {
            dp[t][0] = (0, t)
        }

        // Fill DP
        for t in 0..<T {
            let V = logProbs[t].count

            for k in 0...K {
                if dp[t][k].score == -.infinity { continue }

                let currentScore = dp[t][k].score
                let startFrame = dp[t][k].startFrame

                // Option 1: Stay (emit blank) - only if we have more frames
                if t + 1 < T {
                    let blankScore = blankId < V ? logProbs[t][blankId] : -Float.infinity
                    let newScore = currentScore + blankScore
                    if newScore > dp[t + 1][k].score {
                        dp[t + 1][k] = (newScore, startFrame)
                    }
                }

                // Option 2: Advance (emit next token)
                if k < K, t + 1 < T {
                    let tokenId = tokenIds[k]
                    let tokenScore = tokenId < V ? logProbs[t][tokenId] : -Float.infinity
                    let newScore = currentScore + tokenScore
                    if newScore > dp[t + 1][k + 1].score {
                        dp[t + 1][k + 1] = (newScore, startFrame)
                    }
                }
            }
        }

        // Find best final state (all K tokens emitted)
        var bestScore: Float = -.infinity
        var bestEnd = 0
        var bestStart = 0

        for t in K..<T {
            if dp[t][K].score > bestScore {
                bestScore = dp[t][K].score
                bestEnd = t
                bestStart = dp[t][K].startFrame
            }
        }

        guard bestScore > -.infinity else { return nil }

        // Normalize by number of tokens
        let normalizedScore = bestScore / Float(K)

        return (normalizedScore, bestStart, bestEnd)
    }
}

/// Detection result from hybrid keyword spotter with aligned timestamps.
public struct HybridKeywordDetection: Sendable {
    public let term: CustomVocabularyTerm
    public let score: Float
    public let startFrame: Int
    public let endFrame: Int
    public let startTime: TimeInterval
    public let endTime: TimeInterval
}
