import Foundation

/// Shared Word Error Rate calculation utilities used by CLI commands.
enum WERCalculator {

    /// Compute word-level edit distance statistics and WER for hypothesis/reference pairs.
    static func calculateWERMetrics(
        hypothesis rawHypothesis: String, reference rawReference: String
    )
        -> (wer: Double, insertions: Int, deletions: Int, substitutions: Int, totalWords: Int)
    {
        let hypothesis = TextNormalizer.normalize(rawHypothesis)
        let reference = TextNormalizer.normalize(rawReference)

        let hypWords = hypothesis.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let refWords = reference.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

        let distance = editDistance(hypWords, refWords)
        let wer = refWords.isEmpty ? 0.0 : Double(distance.total) / Double(refWords.count)

        return (wer, distance.insertions, distance.deletions, distance.substitutions, refWords.count)
    }

    /// Compute character-level CER alongside WER if needed.
    static func calculateWERAndCER(
        hypothesis rawHypothesis: String, reference rawReference: String
    )
        -> (
            wer: Double, cer: Double, insertions: Int, deletions: Int, substitutions: Int, totalWords: Int,
            totalCharacters: Int
        )
    {
        let hypothesis = TextNormalizer.normalize(rawHypothesis)
        let reference = TextNormalizer.normalize(rawReference)

        let hypWords = hypothesis.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let refWords = reference.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

        let wordDistance = editDistance(hypWords, refWords)
        let wer = refWords.isEmpty ? 0.0 : Double(wordDistance.total) / Double(refWords.count)

        let hypChars = Array(hypothesis.replacingOccurrences(of: " ", with: ""))
        let refChars = Array(reference.replacingOccurrences(of: " ", with: ""))
        let charDistance = editDistance(hypChars.map(String.init), refChars.map(String.init))
        let cer = refChars.isEmpty ? 0.0 : Double(charDistance.total) / Double(refChars.count)

        return (
            wer,
            cer,
            wordDistance.insertions,
            wordDistance.deletions,
            wordDistance.substitutions,
            refWords.count,
            refChars.count
        )
    }

    private struct EditDistanceResult {
        let total: Int
        let insertions: Int
        let deletions: Int
        let substitutions: Int
    }

    private static func editDistance<T: Equatable>(_ seq1: [T], _ seq2: [T]) -> EditDistanceResult {
        let m = seq1.count
        let n = seq2.count

        if m == 0 {
            return EditDistanceResult(total: n, insertions: n, deletions: 0, substitutions: 0)
        }
        if n == 0 {
            return EditDistanceResult(total: m, insertions: 0, deletions: m, substitutions: 0)
        }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m {
            dp[i][0] = i
        }
        for j in 0...n {
            dp[0][j] = j
        }

        for i in 1...m {
            for j in 1...n {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = 1 + min(dp[i - 1][j], min(dp[i][j - 1], dp[i - 1][j - 1]))
                }
            }
        }

        var i = m
        var j = n
        var insertions = 0
        var deletions = 0
        var substitutions = 0

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && seq1[i - 1] == seq2[j - 1] {
                i -= 1
                j -= 1
            } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
                substitutions += 1
                i -= 1
                j -= 1
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
                deletions += 1
                i -= 1
            } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
                insertions += 1
                j -= 1
            } else {
                break
            }
        }

        return EditDistanceResult(
            total: dp[m][n],
            insertions: insertions,
            deletions: deletions,
            substitutions: substitutions
        )
    }
}
