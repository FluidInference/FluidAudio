import Foundation

/// String and sequence utility functions
public enum StringUtils {
    /// Levenshtein (edit) distance between two sequences of equatable elements.
    /// Works for both character-level (String → [Character]) and word-level ([String]) comparisons.
    ///
    /// - Parameters:
    ///   - a: First sequence
    ///   - b: Second sequence
    /// - Returns: Minimum number of insertions, deletions, and substitutions to transform `a` into `b`
    public static func levenshteinDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        let m = a.count
        let n = b.count

        guard m > 0 else { return n }
        guard n > 0 else { return m }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                let cost = a[i - 1] == b[j - 1] ? 0 : 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  // deletion
                    dp[i][j - 1] + 1,  // insertion
                    dp[i - 1][j - 1] + cost  // substitution
                )
            }
        }

        return dp[m][n]
    }

    /// Convenience overload for String comparison (character-level distance)
    public static func levenshteinDistance(_ a: String, _ b: String) -> Int {
        return levenshteinDistance(Array(a), Array(b))
    }
}
