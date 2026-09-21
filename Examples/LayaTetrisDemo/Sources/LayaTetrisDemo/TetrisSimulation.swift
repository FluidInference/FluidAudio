import Foundation

// Tetris simulation shared with `fluidaudiocli laya-tetris` (Sources/FluidAudioCLI/Commands/Decision/
// LayaTetrisCommand.swift). The demo keeps its own copy because examples only link the library product.

// MARK: - Tetris simulation

struct SplitMix64 {
    private var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

struct TetrisGame {
    static let width = 10
    static let height = 20

    struct Piece {
        let name: String
        /// Rotation states as cell offsets (column, row) with row 0 at the top of the piece box.
        let rotations: [[(Int, Int)]]
    }

    struct Features {
        let linesCleared: Int
        let newHoles: Int
        let landingHeight: Int
        let maxHeight: Int
        let bumpiness: Int
        let bumpinessDelta: Int
        let wellDepth: Int
        let flushSides: Int

        /// Dellacherie-style linear evaluation used by the baseline policy.
        var heuristic: Double {
            Double(linesCleared) * 3.4 - Double(newHoles) * 7.9 - Double(landingHeight) * 4.5 - Double(bumpiness) * 1.2
                - Double(wellDepth) * 3.4
        }
    }

    struct Candidate: Identifiable {
        let id: Int
        let rotation: Int
        let column: Int
        let board: [[Bool]]
        /// Board cells (x, y) the piece occupies before any line clears.
        let cells: [(Int, Int)]
        let features: Features
    }

    static let pieces: [Piece] = [
        Piece(name: "I", rotations: [[(0, 0), (1, 0), (2, 0), (3, 0)], [(0, 0), (0, 1), (0, 2), (0, 3)]]),
        Piece(name: "O", rotations: [[(0, 0), (1, 0), (0, 1), (1, 1)]]),
        Piece(
            name: "T",
            rotations: [
                [(0, 0), (1, 0), (2, 0), (1, 1)], [(0, 0), (0, 1), (0, 2), (1, 1)], [(1, 0), (0, 1), (1, 1), (2, 1)],
                [(1, 0), (0, 1), (1, 1), (1, 2)],
            ]),
        Piece(name: "S", rotations: [[(1, 0), (2, 0), (0, 1), (1, 1)], [(0, 0), (0, 1), (1, 1), (1, 2)]]),
        Piece(name: "Z", rotations: [[(0, 0), (1, 0), (1, 1), (2, 1)], [(1, 0), (0, 1), (1, 1), (0, 2)]]),
        Piece(
            name: "J",
            rotations: [
                [(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)],
                [(1, 0), (1, 1), (0, 2), (1, 2)],
            ]),
        Piece(
            name: "L",
            rotations: [
                [(2, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (0, 1), (0, 2), (1, 2)], [(0, 0), (1, 0), (2, 0), (0, 1)],
                [(0, 0), (1, 0), (1, 1), (1, 2)],
            ]),
    ]

    private(set) var board: [[Bool]]
    private(set) var linesCleared = 0
    private(set) var isOver = false
    private var bag: [Int] = []
    private var rng: SplitMix64

    init(seed: UInt64) {
        board = Array(repeating: Array(repeating: false, count: Self.width), count: Self.height)
        rng = SplitMix64(seed: seed)
    }

    /// Next piece from a seven-bag randomizer, or nil once the stack has topped out.
    mutating func spawn() -> Piece? {
        guard !isOver else { return nil }
        if bag.isEmpty {
            bag = Array(0..<Self.pieces.count)
            for index in stride(from: bag.count - 1, to: 0, by: -1) {
                bag.swapAt(index, Int(rng.next() % UInt64(index + 1)))
            }
        }
        return Self.pieces[bag.removeLast()]
    }

    func candidates(for piece: Piece) -> [Candidate] {
        var result: [Candidate] = []
        let heightsBefore = columnHeights(board)
        let bumpinessBefore = bumpiness(heightsBefore)
        for (rotation, cells) in piece.rotations.enumerated() {
            let pieceWidth = cells.map(\.0).max()! + 1
            let pieceHeight = cells.map(\.1).max()! + 1
            for column in 0...(Self.width - pieceWidth) {
                // Hard drop: lowest row offset where every cell is free.
                var row = -pieceHeight
                while fits(cells, column: column, row: row + 1) { row += 1 }
                guard row >= 0 else { continue }
                var next = board
                let occupied = cells.map { (column + $0.0, row + $0.1) }
                for (dx, dy) in cells { next[row + dy][column + dx] = true }
                let landingHeight = Self.height - row - pieceHeight / 2
                let cleared = clearLines(&next)
                let heightsAfter = columnHeights(next)
                let holesBefore = holes(board)
                let holesAfter = holes(next)
                let bump = bumpiness(heightsAfter)
                let flush = flushSides(cells, column: column, row: row)
                result.append(
                    Candidate(
                        id: result.count, rotation: rotation, column: column, board: next, cells: occupied,
                        features: Features(
                            linesCleared: cleared, newHoles: max(0, holesAfter - holesBefore),
                            landingHeight: landingHeight, maxHeight: heightsAfter.max() ?? 0, bumpiness: bump,
                            bumpinessDelta: bump - bumpinessBefore, wellDepth: deepestWell(heightsAfter),
                            flushSides: flush)))
            }
        }
        return result
    }

    mutating func apply(_ candidate: Candidate) {
        board = candidate.board
        linesCleared += candidate.features.linesCleared
        if board[0].contains(true) || board[1].contains(true) { isOver = true }
    }

    /// One natural sentence per landing, the only thing laya sees. Under the short question
    /// "Is this a clean placement?" the multilingual checkpoint ranks these sensibly: holes and a
    /// taller stack pull P(clean) down, a cleared line pushes it up; numeric feature dumps do not.
    func describe(_ candidate: Candidate, piece: Piece) -> String {
        let f = candidate.features
        var clauses: [String] = []
        clauses.append(
            f.newHoles > 0
                ? "leaves \(Self.words(f.newHoles)) hole\(f.newHoles == 1 ? "" : "s") under it" : "leaves no holes")
        if f.bumpinessDelta > 2 {
            clauses.append("makes the surface much bumpier")
        } else if f.bumpinessDelta > 0 {
            clauses.append("makes the surface bumpier")
        } else if f.bumpinessDelta < 0 {
            clauses.append("makes the surface flatter")
        } else {
            clauses.append("keeps the surface flat")
        }
        if f.maxHeight >= 15 {
            clauses.append("the stack is getting dangerously tall")
        } else if f.landingHeight > 8 {
            clauses.append("makes the stack taller")
        } else {
            clauses.append("keeps the stack low")
        }
        if f.wellDepth >= 3 { clauses.append("leaves a deep well") }
        if f.linesCleared > 0 {
            clauses.append("clears \(Self.words(f.linesCleared)) line\(f.linesCleared == 1 ? "" : "s")")
        }
        let body =
            clauses.count > 1 ? clauses.dropLast().joined(separator: ", ") + ", and " + clauses.last! : clauses[0]
        return "The \(piece.name) piece dropped at column \(candidate.column) " + body + "."
    }

    private static func words(_ value: Int) -> String {
        let names = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        return value < names.count ? names[value] : String(value)
    }

    func render() -> String {
        Self.render(board)
    }

    func render(after candidate: Candidate) -> String {
        Self.render(candidate.board)
    }

    private static func render(_ grid: [[Bool]]) -> String {
        grid.map { row in row.map { $0 ? "█" : "·" }.joined() }.joined(separator: "\n")
    }

    // MARK: Geometry

    private func fits(_ cells: [(Int, Int)], column: Int, row: Int) -> Bool {
        for (dx, dy) in cells {
            let x = column + dx
            let y = row + dy
            guard x >= 0, x < Self.width, y < Self.height else { return false }
            if y >= 0, board[y][x] { return false }
        }
        return true
    }

    private func clearLines(_ grid: inout [[Bool]]) -> Int {
        let kept = grid.filter { !$0.allSatisfy { $0 } }
        let cleared = grid.count - kept.count
        grid = Array(repeating: Array(repeating: false, count: Self.width), count: cleared) + kept
        return cleared
    }

    private func columnHeights(_ grid: [[Bool]]) -> [Int] {
        (0..<Self.width).map { x in
            for y in 0..<Self.height where grid[y][x] { return Self.height - y }
            return 0
        }
    }

    private func holes(_ grid: [[Bool]]) -> Int {
        var count = 0
        for x in 0..<Self.width {
            var covered = false
            for y in 0..<Self.height {
                if grid[y][x] { covered = true } else if covered { count += 1 }
            }
        }
        return count
    }

    private func bumpiness(_ heights: [Int]) -> Int {
        zip(heights, heights.dropFirst()).reduce(0) { $0 + abs($1.0 - $1.1) }
    }

    private func deepestWell(_ heights: [Int]) -> Int {
        var deepest = 0
        for x in 0..<Self.width {
            let left = x == 0 ? Self.height : heights[x - 1]
            let right = x == Self.width - 1 ? Self.height : heights[x + 1]
            deepest = max(deepest, min(left, right) - heights[x])
        }
        return deepest
    }

    private func flushSides(_ cells: [(Int, Int)], column: Int, row: Int) -> Int {
        var count = 0
        for (dx, dy) in cells {
            for neighbour in [(column + dx - 1, row + dy), (column + dx + 1, row + dy)] {
                if neighbour.0 < 0 || neighbour.0 >= Self.width {
                    count += 1
                } else if neighbour.1 >= 0, board[neighbour.1][neighbour.0] {
                    count += 1
                }
            }
        }
        return count
    }
}
