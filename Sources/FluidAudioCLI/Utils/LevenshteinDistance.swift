#if os(macOS)
/// Compute the Levenshtein edit distance between two sequences.
///
/// Uses the standard dynamic programming approach with O(min(m,n)) space
/// via a two-row optimization.
func levenshteinDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
    let m = a.count
    let n = b.count

    if m == 0 { return n }
    if n == 0 { return m }

    // Ensure `b` is the shorter sequence for space optimization.
    if m < n { return levenshteinDistance(b, a) }

    var prev = Array(0...n)
    var curr = Array(repeating: 0, count: n + 1)

    for i in 1...m {
        curr[0] = i
        for j in 1...n {
            let cost = a[i - 1] == b[j - 1] ? 0 : 1
            curr[j] = min(
                prev[j] + 1,  // deletion
                curr[j - 1] + 1,  // insertion
                prev[j - 1] + cost  // substitution
            )
        }
        swap(&prev, &curr)
    }

    return prev[n]
}
#endif
