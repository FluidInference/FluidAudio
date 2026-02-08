import Foundation
import NaturalLanguage

/// Shared Word Error Rate calculation utilities used by CLI commands.
enum WERCalculator {

    // MARK: - Tokenization

    /// Tokenize text into words, using NaturalLanguage for Chinese/Japanese/Korean.
    private static func tokenize(_ text: String) -> [String] {
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text

        var words: [String] = []
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let word = String(text[range])
            if !word.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                words.append(word)
            }
            return true
        }
        return words
    }

    /// Check if text contains CJK (Chinese/Japanese/Korean) characters.
    private static func containsCJK(_ text: String) -> Bool {
        text.unicodeScalars.contains { scalar in
            let value = scalar.value
            // CJK Unified Ideographs (Chinese)
            return (0x4E00...0x9FFF).contains(value)
                // CJK Extension A
                || (0x3400...0x4DBF).contains(value)
                // Hiragana (Japanese)
                || (0x3040...0x309F).contains(value)
                // Katakana (Japanese)
                || (0x30A0...0x30FF).contains(value)
                // Hangul Syllables (Korean)
                || (0xAC00...0xD7AF).contains(value)
                // Hangul Jamo (Korean)
                || (0x1100...0x11FF).contains(value)
        }
    }

    /// Tokenize hypothesis and reference, using NL tokenizer for CJK, whitespace for others.
    private static func tokenizePair(
        hypothesis: String,
        reference: String
    ) -> (hypWords: [String], refWords: [String]) {
        if containsCJK(reference) {
            return (tokenize(hypothesis), tokenize(reference))
        } else {
            return (
                hypothesis.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty },
                reference.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
            )
        }
    }

    // MARK: - Public API

    /// Compute word-level edit distance statistics and WER for hypothesis/reference pairs.
    /// Uses NaturalLanguage tokenization for CJK languages (Chinese/Japanese/Korean).
    static func calculateWERMetrics(
        hypothesis rawHypothesis: String, reference rawReference: String
    ) -> (wer: Double, insertions: Int, deletions: Int, substitutions: Int, totalWords: Int) {
        let hypothesis = TextNormalizer.normalize(rawHypothesis)
        let reference = TextNormalizer.normalize(rawReference)

        let (hypWords, refWords) = tokenizePair(hypothesis: hypothesis, reference: reference)
        let distance = editDistance(hypWords, refWords)
        let wer = refWords.isEmpty ? 0.0 : Double(distance.total) / Double(refWords.count)

        return (wer, distance.insertions, distance.deletions, distance.substitutions, refWords.count)
    }

    /// Compute character-level CER alongside WER if needed.
    /// Uses NaturalLanguage tokenization for CJK languages (Chinese/Japanese/Korean).
    static func calculateWERAndCER(
        hypothesis rawHypothesis: String, reference rawReference: String
    ) -> (
        wer: Double, cer: Double, insertions: Int, deletions: Int, substitutions: Int,
        totalWords: Int, totalCharacters: Int
    ) {
        let hypothesis = TextNormalizer.normalize(rawHypothesis)
        let reference = TextNormalizer.normalize(rawReference)

        let (hypWords, refWords) = tokenizePair(hypothesis: hypothesis, reference: reference)
        let wordDistance = editDistance(hypWords, refWords)
        let wer = refWords.isEmpty ? 0.0 : Double(wordDistance.total) / Double(refWords.count)

        // Character-level edit distance (spaces removed)
        let hypChars = Array(hypothesis.filter { $0 != " " })
        let refChars = Array(reference.filter { $0 != " " })
        let charDistance = editDistanceChars(hypChars, refChars)
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

    // MARK: - Edit Distance

    private struct EditDistanceResult {
        let total: Int
        let insertions: Int
        let deletions: Int
        let substitutions: Int
    }

    /// Character-based edit distance (avoids String conversion overhead).
    private static func editDistanceChars(_ seq1: [Character], _ seq2: [Character]) -> EditDistanceResult {
        let m = seq1.count
        let n = seq2.count

        if m == 0 {
            return EditDistanceResult(total: n, insertions: n, deletions: 0, substitutions: 0)
        }
        if n == 0 {
            return EditDistanceResult(total: m, insertions: 0, deletions: m, substitutions: 0)
        }

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if seq1[i - 1] == seq2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                }
            }
        }

        // Backtrack to count insertions, deletions, substitutions
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

    /// Generic edit distance for string arrays.
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
