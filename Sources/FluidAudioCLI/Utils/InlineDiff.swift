#if os(macOS)
import Foundation

/// Word-level inline diff with ANSI colour highlighting.
///
/// Computes an edit-distance alignment between `reference` and `hypothesis`
/// and backtracks the DP table to classify each token as equal, substituted,
/// deleted, or inserted. Returns two strings ready to print side-by-side:
/// reference tokens that differ are highlighted red; hypothesis tokens that
/// differ are highlighted green.
///
/// Not FLEURS-specific — any ASR benchmark can use this to display high-WER
/// samples.
enum InlineDiff {
    /// - Parameters:
    ///   - reference: Tokens from the ground-truth transcript.
    ///   - hypothesis: Tokens produced by the ASR system.
    /// - Returns: `(referenceWithHighlights, hypothesisWithHighlights)`. When
    ///   the terminal does not expose `TERM`, ANSI escapes are replaced with
    ///   `[...]` brackets so the diff stays readable in log files.
    static func generate(reference: [String], hypothesis: [String]) -> (String, String) {
        let supportsColor = ProcessInfo.processInfo.environment["TERM"] != nil
        let redColor = supportsColor ? "\u{001B}[31m" : "["
        let greenColor = supportsColor ? "\u{001B}[32m" : "["
        let resetColor = supportsColor ? "\u{001B}[0m" : "]"

        let m = reference.count
        let n = hypothesis.count

        if n == 0 {
            let refString = reference.map { "\(redColor)\($0)\(resetColor)" }.joined(separator: " ")
            return (refString, "")
        }
        if m == 0 {
            let hypString = hypothesis.map { "\(greenColor)\($0)\(resetColor)" }.joined(separator: " ")
            return ("", hypString)
        }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }
        for i in 1...m {
            for j in 1...n {
                if reference[i - 1] == hypothesis[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] =
                        1
                        + min(
                            dp[i - 1][j],
                            dp[i][j - 1],
                            dp[i - 1][j - 1]
                        )
                }
            }
        }

        var i = m
        var j = n
        var refDiffWords: [(String, Bool)] = []
        var hypDiffWords: [(String, Bool)] = []

        while i > 0 || j > 0 {
            if i > 0 && j > 0 && reference[i - 1] == hypothesis[j - 1] {
                refDiffWords.insert((reference[i - 1], false), at: 0)
                hypDiffWords.insert((hypothesis[j - 1], false), at: 0)
                i -= 1
                j -= 1
            } else if i > 0 && j > 0 && dp[i][j] == dp[i - 1][j - 1] + 1 {
                refDiffWords.insert((reference[i - 1], true), at: 0)
                hypDiffWords.insert((hypothesis[j - 1], true), at: 0)
                i -= 1
                j -= 1
            } else if i > 0 && dp[i][j] == dp[i - 1][j] + 1 {
                refDiffWords.insert((reference[i - 1], true), at: 0)
                i -= 1
            } else if j > 0 && dp[i][j] == dp[i][j - 1] + 1 {
                hypDiffWords.insert((hypothesis[j - 1], true), at: 0)
                j -= 1
            } else {
                break
            }
        }

        var refString = ""
        var hypString = ""
        for (word, isDifferent) in refDiffWords {
            if !refString.isEmpty { refString += " " }
            refString += isDifferent ? "\(redColor)\(word)\(resetColor)" : word
        }
        for (word, isDifferent) in hypDiffWords {
            if !hypString.isEmpty { hypString += " " }
            hypString += isDifferent ? "\(greenColor)\(word)\(resetColor)" : word
        }

        return (refString, hypString)
    }
}
#endif
