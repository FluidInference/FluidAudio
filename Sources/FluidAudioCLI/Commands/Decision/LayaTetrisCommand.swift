#if os(macOS)
import FluidAudio
import Foundation

/// `laya-tetris`: headless Tetris played by laya decisions.
///
/// Every legal landing of the current piece is described in one sentence and laya answers a
/// `noul` question, "Is this a clean placement?"; the highest P(true) wins. The game reports lines
/// cleared, pieces placed, and per-decision latency, which is the point: each decision is one
/// fixed-shape encoder pass on device. A feature-weighted heuristic policy is available as a baseline.
struct LayaTetrisCommand {
    private static let logger = AppLogger(category: "LayaTetris")

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }
        do {
            try await play(arguments: arguments)
        } catch {
            logger.error("laya-tetris failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - Options

    private struct Options {
        var modelDirectory: String?
        var lengths: [Int] = [128]
        var maxPieces = 200
        var seed: UInt64 = 7
        var policy = "laya"
        var showEvery = 0
        var trace = 0
        var json = false
    }

    private static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var index = 0
        func value(_ flag: String) throws -> String {
            index += 1
            guard index < arguments.count else { throw LayaError.invalidAsset("\(flag) needs a value") }
            return arguments[index]
        }
        while index < arguments.count {
            switch arguments[index] {
            case "--model-dir": options.modelDirectory = try value("--model-dir")
            case "--lengths": options.lengths = try value("--lengths").split(separator: ",").compactMap { Int($0) }
            case "--pieces": options.maxPieces = Int(try value("--pieces")) ?? options.maxPieces
            case "--seed": options.seed = UInt64(try value("--seed")) ?? options.seed
            case "--policy": options.policy = try value("--policy")
            case "--show-every": options.showEvery = Int(try value("--show-every")) ?? 0
            case "--trace": options.trace = Int(try value("--trace")) ?? 0
            case "--json": options.json = true
            default: throw LayaError.invalidAsset("Unknown argument \(arguments[index])")
            }
            index += 1
        }
        guard ["laya", "heuristic", "random"].contains(options.policy) else {
            throw LayaError.invalidAsset("--policy must be laya, heuristic, or random")
        }
        return options
    }

    // MARK: - Game

    static let question = LayaQuestion.noul("Is this a clean placement?")

    private static func play(arguments: [String]) async throws {
        let options = try parse(arguments)
        var manager: LayaManager?
        if options.policy == "laya" {
            let configuration = LayaManager.Configuration(lengths: options.lengths)
            if let directory = options.modelDirectory {
                manager = try await LayaManager.load(
                    from: URL(fileURLWithPath: directory), configuration: configuration)
            } else {
                manager = try await LayaManager.load(configuration: configuration)
            }
        }
        var game = TetrisGame(seed: options.seed)
        var rng = SplitMix64(seed: options.seed &+ 1)
        var decisionTimes: [Double] = []
        var tokenCounts: [Int] = []
        let started = Date()
        var pieces = 0
        while pieces < options.maxPieces, let piece = game.spawn() {
            let candidates = game.candidates(for: piece)
            guard !candidates.isEmpty else { break }
            let chosen: TetrisGame.Candidate
            switch options.policy {
            case "laya":
                guard let manager else { throw LayaError.invalidModel("The laya policy needs a loaded model") }
                var best: (score: Float, candidate: TetrisGame.Candidate)?
                for candidate in candidates {
                    let state = game.describe(candidate, piece: piece)
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    let answer = try await manager.answer(state: state, question: question)
                    decisionTimes.append(Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6)
                    tokenCounts.append(answer.tokenCount)
                    let score = answer.noul ?? 0
                    if pieces < options.trace {
                        print(String(format: "  %.3f  %@", score, state))
                    }
                    if best == nil || score > best!.score { best = (score, candidate) }
                }
                chosen = best!.candidate
                if pieces < options.trace {
                    print("  -> column \(chosen.column) rotation \(chosen.rotation)\n\(game.render(after: chosen))")
                }
            case "heuristic":
                chosen = candidates.max { $0.features.heuristic < $1.features.heuristic }!
            default:
                chosen = candidates[Int(rng.next() % UInt64(candidates.count))]
            }
            game.apply(chosen)
            pieces += 1
            if options.showEvery > 0, pieces % options.showEvery == 0 {
                print("piece \(pieces) · lines \(game.linesCleared)\n\(game.render())")
            }
        }
        let elapsed = Date().timeIntervalSince(started)
        decisionTimes.sort()
        let median = decisionTimes.isEmpty ? 0 : decisionTimes[decisionTimes.count / 2]
        let p95 =
            decisionTimes.isEmpty
            ? 0 : decisionTimes[min(decisionTimes.count - 1, Int(Double(decisionTimes.count) * 0.95))]
        let perMinute = elapsed > 0 ? Double(decisionTimes.count) / elapsed * 60 : 0
        if options.json {
            let payload: [String: Any] = [
                "policy": options.policy, "pieces": pieces, "lines": game.linesCleared, "game_over": game.isOver,
                "decisions": decisionTimes.count, "median_ms": median, "p95_ms": p95, "decisions_per_minute": perMinute,
                "elapsed_s": elapsed, "max_tokens": tokenCounts.max() ?? 0,
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            print(String(decoding: data, as: UTF8.self))
            return
        }
        print(game.render())
        print(
            "policy \(options.policy) · pieces \(pieces) · lines cleared \(game.linesCleared) · \(game.isOver ? "topped out" : "stopped at piece cap")"
        )
        if !decisionTimes.isEmpty {
            print(
                String(
                    format:
                        "%d laya decisions · median %.2f ms · p95 %.2f ms · %.0f decisions/min · prompts ≤ %d tokens · %.1f s total",
                    decisionTimes.count, median, p95, perMinute, tokenCounts.max() ?? 0, elapsed))
        } else {
            print(String(format: "%.2f s total", elapsed))
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudiocli laya-tetris [--model-dir DIR] [--lengths 128] [--pieces 200] [--seed 7]
                                             [--policy laya|heuristic|random] [--show-every N] [--trace N] [--json]

            Plays headless 10x20 Tetris. With --policy laya (default) every legal landing is described in
            one sentence and scored by laya's P(clean); the best-scoring landing is played.
            """)
    }
}

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

    struct Candidate {
        let rotation: Int
        let column: Int
        let board: [[Bool]]
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
                        rotation: rotation, column: column, board: next,
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
#endif
