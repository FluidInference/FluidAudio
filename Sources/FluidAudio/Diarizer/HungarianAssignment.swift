import Foundation

// MARK: - Hungarian (O(n^3) min-cost assignment on a square cost matrix)

enum HungarianAssignment {
    /// Kuhn-Munkres with potentials. `cost` is row-major `n × n`,
    /// non-negative integers. Returns `assign[row] = col`.
    static func solve(costSquare cost: [Int], n: Int) -> [Int] {
        if n == 0 { return [] }
        // Classic Jonker-Volgenant style implementation adapted for
        // square n×n. 1-based indexing in arrays of size n+1 for the
        // canonical algorithm — cost is read as `cost[(i-1)*n + (j-1)]`.
        let INF = Int.max / 4
        var u = [Int](repeating: 0, count: n + 1)
        var v = [Int](repeating: 0, count: n + 1)
        var p = [Int](repeating: 0, count: n + 1)
        var way = [Int](repeating: 0, count: n + 1)

        for i in 1...n {
            p[0] = i
            var j0 = 0
            var minv = [Int](repeating: INF, count: n + 1)
            var used = [Bool](repeating: false, count: n + 1)
            repeat {
                used[j0] = true
                let i0 = p[j0]
                var delta = INF
                var j1 = 0
                for j in 1...n where !used[j] {
                    let cur = cost[(i0 - 1) * n + (j - 1)] - u[i0] - v[j]
                    if cur < minv[j] {
                        minv[j] = cur
                        way[j] = j0
                    }
                    if minv[j] < delta {
                        delta = minv[j]
                        j1 = j
                    }
                }
                for j in 0...n {
                    if used[j] {
                        u[p[j]] += delta
                        v[j] -= delta
                    } else {
                        minv[j] -= delta
                    }
                }
                j0 = j1
            } while p[j0] != 0
            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }
        var assign = [Int](repeating: -1, count: n)
        for j in 1...n {
            if p[j] != 0 { assign[p[j] - 1] = j - 1 }
        }
        return assign
    }

    /// Max-total-score assignment on a rectangular score matrix
    /// (`scores[row][col]`, higher is better). Returns `assign[row] = col`,
    /// or `-1` for rows left unassigned when there are more rows than columns.
    /// Non-finite scores are treated as worse than any finite score.
    static func maxScoreAssignment(scores: [[Double]]) -> [Int] {
        let rows = scores.count
        guard rows > 0 else { return [] }
        let cols = scores[0].count
        guard cols > 0 else { return Array(repeating: -1, count: rows) }

        let finite = scores.flatMap { $0 }.filter { $0.isFinite }
        let maxScore = finite.max() ?? 0
        let minScore = finite.min() ?? 0
        let sentinel = minScore - 1

        // Pad to square; dummy cells share a constant cost so they cannot
        // influence which real pairs the solver prefers. Maximise score ⇒
        // minimise (maxScore − score), scaled to integers for the solver.
        let n = max(rows, cols)
        let scale = 1e6
        var cost = [Int](repeating: 0, count: n * n)
        for r in 0..<rows {
            precondition(scores[r].count == cols, "Jagged score matrix is not supported")
            for c in 0..<cols {
                let score = scores[r][c].isFinite ? scores[r][c] : sentinel
                cost[r * n + c] = Int(((maxScore - score) * scale).rounded())
            }
        }

        let assign = solve(costSquare: cost, n: n)
        return (0..<rows).map { row in
            let col = assign[row]
            return col < cols ? col : -1
        }
    }
}
