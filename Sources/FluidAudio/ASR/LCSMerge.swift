import Foundation

/// Longest Common Subsequence (LCS) based text merging for overlapping transcription chunks
struct LCSMerge {

    /// Result of LCS-based text merging
    struct MergeResult {
        let mergedText: String
        let newPortion: String  // Only the new text that wasn't in the previous chunk
        let overlapLength: Int  // Number of overlapping words
        let confidence: Float
    }

    /// Merge two overlapping text segments using LCS algorithm
    /// - Parameters:
    ///   - previousText: Text from the previous chunk
    ///   - currentText: Text from the current chunk
    ///   - previousConfidence: Confidence score of previous chunk
    ///   - currentConfidence: Confidence score of current chunk
    /// - Returns: MergeResult containing merged text and new portion
    static func mergeOverlappingTexts(
        previousText: String,
        currentText: String,
        previousConfidence: Float,
        currentConfidence: Float
    ) -> MergeResult {
        let previousWords = previousText.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }
        let currentWords = currentText.components(separatedBy: .whitespacesAndNewlines).filter { !$0.isEmpty }

        // Find the best overlap using LCS
        let overlap = findBestOverlap(previous: previousWords, current: currentWords)

        if overlap.length > 0 {
            // Found overlap - merge intelligently with bounds checking
            let stablePreviousCount = max(0, previousWords.count - overlap.length)
            let stablePrevious = Array(previousWords.prefix(stablePreviousCount))

            let dropCount = min(overlap.length, currentWords.count)
            let newWords = Array(currentWords.dropFirst(dropCount))

            let mergedWords: [String]
            let confidence: Float

            // Choose the higher confidence version for the overlap region
            if currentConfidence > previousConfidence * 1.1 {  // 10% threshold
                // Current chunk has significantly higher confidence
                mergedWords = stablePrevious + currentWords
                confidence = currentConfidence
            } else {
                // Keep previous chunk's overlap, add only new words
                mergedWords = previousWords + newWords
                confidence = max(previousConfidence, currentConfidence)
            }

            let mergedText = mergedWords.joined(separator: " ")
            let newPortion = newWords.joined(separator: " ")

            return MergeResult(
                mergedText: mergedText,
                newPortion: newPortion,
                overlapLength: overlap.length,
                confidence: confidence
            )
        } else {
            // No overlap found - concatenate with deduplication
            let deduplicatedWords = removeDuplicateWords(previousWords + currentWords)
            let mergedText = deduplicatedWords.joined(separator: " ")

            // All of current text is new
            let newPortion = currentText

            return MergeResult(
                mergedText: mergedText,
                newPortion: newPortion,
                overlapLength: 0,
                confidence: max(previousConfidence, currentConfidence)
            )
        }
    }

    /// Find the best overlap between two word arrays using LCS
    private static func findBestOverlap(previous: [String], current: [String]) -> OverlapInfo {
        let maxOverlapLength = min(previous.count, current.count, 8)  // Limit to smaller overlap to be more conservative
        var bestOverlap = OverlapInfo(length: 0, previousStartIndex: 0, currentStartIndex: 0)

        // Only look for overlaps if we have enough words to avoid range errors
        guard maxOverlapLength >= 2 else {
            return bestOverlap  // No overlap possible
        }

        // Only look for overlaps of at least 2 words to avoid false positives
        for overlapLength in 2...maxOverlapLength {
            // Check suffix of previous against prefix of current
            let previousSuffix = Array(previous.suffix(overlapLength))
            let currentPrefix = Array(current.prefix(overlapLength))

            // Use exact matching for more precision, but allow for minor variations
            if isSequenceSimilar(previousSuffix, currentPrefix, threshold: 0.9) {
                bestOverlap = OverlapInfo(
                    length: overlapLength,
                    previousStartIndex: previous.count - overlapLength,
                    currentStartIndex: 0
                )
                break  // Take first good match to avoid over-merging
            }
        }

        return bestOverlap
    }

    /// Check if two word sequences are similar enough to be considered overlapping
    private static func isSequenceSimilar(_ seq1: [String], _ seq2: [String], threshold: Double) -> Bool {
        guard seq1.count == seq2.count else { return false }

        let matchingWords = zip(seq1, seq2).reduce(0) { count, pair in
            count + (isWordSimilar(pair.0, pair.1) ? 1 : 0)
        }

        let similarity = Double(matchingWords) / Double(seq1.count)
        return similarity >= threshold
    }

    /// Check if two words are similar (handles minor transcription variations)
    private static func isWordSimilar(_ word1: String, _ word2: String) -> Bool {
        let normalized1 = word1.lowercased().trimmingCharacters(in: .punctuationCharacters)
        let normalized2 = word2.lowercased().trimmingCharacters(in: .punctuationCharacters)

        // Exact match
        if normalized1 == normalized2 {
            return true
        }

        // Handle common transcription variations
        let variations = [
            ("and", "an"), ("the", "a"), ("to", "too"), ("for", "four"),
            ("one", "won"), ("two", "to"), ("there", "their"), ("its", "it's"),
        ]

        for (var1, var2) in variations {
            if (normalized1 == var1 && normalized2 == var2) || (normalized1 == var2 && normalized2 == var1) {
                return true
            }
        }

        // Levenshtein distance for similar words (typos, etc.)
        if normalized1.count > 3 && normalized2.count > 3 {
            let distance = levenshteinDistance(normalized1, normalized2)
            let maxLength = max(normalized1.count, normalized2.count)
            return Double(distance) / Double(maxLength) <= 0.3  // 30% threshold
        }

        return false
    }

    /// Calculate Levenshtein distance between two strings
    private static func levenshteinDistance(_ str1: String, _ str2: String) -> Int {
        let arr1 = Array(str1)
        let arr2 = Array(str2)
        let m = arr1.count
        let n = arr2.count

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m { dp[i][0] = i }
        for j in 0...n { dp[0][j] = j }

        for i in 1...m {
            for j in 1...n {
                if arr1[i - 1] == arr2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
                }
            }
        }

        return dp[m][n]
    }

    /// Remove consecutive duplicate words from a word array
    private static func removeDuplicateWords(_ words: [String]) -> [String] {
        guard !words.isEmpty else { return [] }

        var result = [words[0]]
        for i in 1..<words.count {
            if !isWordSimilar(words[i], words[i - 1]) {
                result.append(words[i])
            }
        }
        return result
    }
}

/// Information about overlapping regions between two text sequences
private struct OverlapInfo {
    let length: Int
    let previousStartIndex: Int
    let currentStartIndex: Int
}
