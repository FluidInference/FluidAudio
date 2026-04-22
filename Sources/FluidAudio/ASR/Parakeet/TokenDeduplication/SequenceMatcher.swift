import Foundation

/// Core sequence matching logic extracted from token deduplication implementations.
///
/// This utility consolidates common matching algorithms used across:
/// - `AsrManager+TokenProcessing.swift` - Token-based streaming deduplication
/// - `ChunkProcessor.swift` - Time-based chunk merging
struct SequenceMatcher<Element> {

    // MARK: - Suffix-Prefix Matching

    /// Find longest suffix-prefix overlap between two sequences.
    ///
    /// Searches for the longest sequence where the end of `previous` matches
    /// the beginning of `current`. This is the most common case in streaming
    /// where consecutive chunks overlap at their boundaries.
    ///
    /// Extracted from: `AsrManager+TokenProcessing.swift:128-138`
    ///
    /// - Parameters:
    ///   - previous: The first sequence
    ///   - current: The second sequence
    ///   - maxOverlap: Maximum overlap length to search (default: no limit)
    ///   - matcher: Closure to determine if two elements match
    /// - Returns: Match if found, nil otherwise
    static func findSuffixPrefixMatch(
        previous: [Element],
        current: [Element],
        maxOverlap: Int = Int.max,
        matcher: (Element, Element) -> Bool
    ) -> SequenceMatch? {
        let maxSearchLength = min(maxOverlap, previous.count)
        let maxMatchLength = min(maxOverlap, current.count)

        guard maxSearchLength >= 2 && maxMatchLength >= 2 else {
            return nil
        }

        // Search from longest to shortest for greedy matching
        for overlapLength in (2...min(maxSearchLength, maxMatchLength)).reversed() {
            let prevSuffix = Array(previous.suffix(overlapLength))
            let currPrefix = Array(current.prefix(overlapLength))

            if arraysMatch(prevSuffix, currPrefix, matcher: matcher) {
                return SequenceMatch(
                    leftStartIndex: previous.count - overlapLength,
                    rightStartIndex: 0,
                    length: overlapLength
                )
            }
        }

        return nil
    }

    // MARK: - Bounded Substring Search

    /// Find overlapping subsequence within bounded search window.
    ///
    /// Searches for matching subsequences within a limited window, useful when
    /// overlap might not be at exact boundaries. This handles cases where tokens
    /// might shift due to decoding variations.
    ///
    /// Extracted from: `AsrManager+TokenProcessing.swift:143-165`
    ///
    /// - Parameters:
    ///   - previous: The first sequence
    ///   - current: The second sequence
    ///   - maxSearchLength: Maximum length to search in previous sequence
    ///   - boundarySearchFrames: Maximum offset from start of current to search
    ///   - matcher: Closure to determine if two elements match
    /// - Returns: Match if found, nil otherwise
    static func findBoundedSubstringMatch(
        previous: [Element],
        current: [Element],
        maxSearchLength: Int,
        boundarySearchFrames: Int,
        matcher: (Element, Element) -> Bool
    ) -> SequenceMatch? {
        // Need at least 2 elements in both sequences
        guard previous.count >= 2 && current.count >= 2 else {
            return nil
        }

        for overlapLength in (2...min(maxSearchLength, current.count)).reversed() {
            let prevStart = max(0, previous.count - maxSearchLength)
            let prevEnd = previous.count - overlapLength + 1
            if prevEnd <= prevStart { continue }

            for startIndex in prevStart..<prevEnd {
                let prevSub = Array(previous[startIndex..<(startIndex + overlapLength)])
                let currEnd = max(0, current.count - overlapLength + 1)

                // Limit search window using boundary frames
                let searchLimit = min(boundarySearchFrames, currEnd)
                for currentStart in 0..<searchLimit {
                    let currSub = Array(current[currentStart..<(currentStart + overlapLength)])
                    if arraysMatch(prevSub, currSub, matcher: matcher) {
                        return SequenceMatch(
                            leftStartIndex: startIndex,
                            rightStartIndex: currentStart,
                            length: overlapLength
                        )
                    }
                }
            }
        }

        return nil
    }

    // MARK: - Longest Common Subsequence

    /// Find longest common subsequence using dynamic programming.
    ///
    /// Returns all matching positions found via LCS algorithm. This allows
    /// non-contiguous matches and is useful when sequences might have
    /// insertions or deletions.
    ///
    /// Extracted from: `ChunkProcessor.swift:336-373`
    ///
    /// - Parameters:
    ///   - left: The first sequence
    ///   - right: The second sequence
    ///   - matcher: Closure to determine if two elements match
    /// - Returns: Array of single-element matches (use `consolidateMatches` to merge)
    static func findLongestCommonSubsequence(
        left: [Element],
        right: [Element],
        matcher: (Element, Element) -> Bool
    ) -> [SequenceMatch] {
        let leftCount = left.count
        let rightCount = right.count

        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)

        // Build DP table
        for i in 1...leftCount {
            for j in 1...rightCount {
                if matcher(left[i - 1], right[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        // Backtrack to find matches
        var matches: [SequenceMatch] = []
        var i = leftCount
        var j = rightCount

        while i > 0 && j > 0 {
            if matcher(left[i - 1], right[j - 1]) {
                // Match found - create single-element match
                matches.append(
                    SequenceMatch(
                        leftStartIndex: i - 1,
                        rightStartIndex: j - 1,
                        length: 1
                    ))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return matches.reversed()
    }

    // MARK: - Contiguous Matching

    /// Find longest contiguous matching sequence.
    ///
    /// Searches for the longest run of consecutive matches between two sequences.
    /// Unlike LCS, this requires matches to be adjacent in both sequences.
    ///
    /// Extracted from: `ChunkProcessor.swift:296-334`
    ///
    /// - Parameters:
    ///   - left: The first sequence
    ///   - right: The second sequence
    ///   - matcher: Closure to determine if two elements match
    /// - Returns: Array of single-element matches forming the longest contiguous run
    static func findContiguousMatches(
        left: [Element],
        right: [Element],
        matcher: (Element, Element) -> Bool
    ) -> [SequenceMatch] {
        var best: [SequenceMatch] = []

        for i in 0..<left.count {
            for j in 0..<right.count {
                if matcher(left[i], right[j]) {
                    var current: [SequenceMatch] = []
                    var k = i
                    var l = j

                    while k < left.count && l < right.count {
                        if matcher(left[k], right[l]) {
                            current.append(
                                SequenceMatch(
                                    leftStartIndex: k,
                                    rightStartIndex: l,
                                    length: 1
                                ))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    // MARK: - Match Consolidation

    /// Consolidate adjacent single-element matches into contiguous sequences.
    ///
    /// Takes an array of single-element matches (typically from LCS or contiguous
    /// matching) and merges adjacent matches into longer sequences.
    ///
    /// - Parameter matches: Array of single-element matches
    /// - Returns: Array of consolidated matches
    static func consolidateMatches(_ matches: [SequenceMatch]) -> [SequenceMatch] {
        guard !matches.isEmpty else { return [] }

        var consolidated: [SequenceMatch] = []
        var current = matches[0]

        for i in 1..<matches.count {
            let next = matches[i]
            // Check if next match is contiguous with current
            if next.leftStartIndex == current.leftStartIndex + current.length
                && next.rightStartIndex == current.rightStartIndex + current.length
            {
                // Extend current match
                current = SequenceMatch(
                    leftStartIndex: current.leftStartIndex,
                    rightStartIndex: current.rightStartIndex,
                    length: current.length + next.length
                )
            } else {
                // Non-contiguous, save current and start new
                consolidated.append(current)
                current = next
            }
        }
        consolidated.append(current)

        return consolidated
    }

    // MARK: - Helper Methods

    private static func arraysMatch(
        _ a: [Element],
        _ b: [Element],
        matcher: (Element, Element) -> Bool
    ) -> Bool {
        guard a.count == b.count else { return false }
        return zip(a, b).allSatisfy(matcher)
    }
}
