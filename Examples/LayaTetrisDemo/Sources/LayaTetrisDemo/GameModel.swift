import FluidAudio
import Foundation
import SwiftUI

/// One scored landing, shown in the decision console.
struct ScoredCandidate: Identifiable {
    let id: Int
    let candidate: TetrisGame.Candidate
    let sentence: String
    let probability: Float
    let milliseconds: Double
}

/// Drives the game loop: laya scores every legal landing of the current piece, the best wins.
@MainActor
final class GameModel: ObservableObject {
    enum Policy: String, CaseIterable, Identifiable {
        case laya, heuristic, random
        var id: String { rawValue }
    }

    // Model
    @Published private(set) var manager: LayaManager?
    @Published private(set) var loadStatus = "Not loaded"
    @Published private(set) var isLoading = false
    @Published var errorMessage: String?

    // Game
    @Published private(set) var board: [[Bool]] = TetrisGame(seed: 1).board
    @Published private(set) var currentPiece = "–"
    @Published private(set) var candidates: [ScoredCandidate] = []
    @Published private(set) var chosen: TetrisGame.Candidate?
    @Published private(set) var evaluating: TetrisGame.Candidate?
    @Published private(set) var pieces = 0
    @Published private(set) var lines = 0
    @Published private(set) var isOver = false
    @Published private(set) var isRunning = false
    @Published private(set) var log: [String] = []

    // Stats
    @Published private(set) var decisions = 0
    @Published private(set) var medianMs = 0.0
    @Published private(set) var lastMs = 0.0
    @Published private(set) var decisionsPerMinute = 0.0
    @Published private(set) var promptTokens = 0
    @Published private(set) var bucket = 0

    // Controls
    @Published var policy: Policy = .laya
    @Published var seed: UInt64 = 7
    /// Extra delay per scored candidate so the scoring can be watched; 0 runs flat out.
    @Published var stepDelayMs = 0.0
    /// Pause after each placed piece.
    @Published var pieceDelayMs = 120.0

    private var game = TetrisGame(seed: 7)
    private var rng = SplitMix64(seed: 8)
    private var task: Task<Void, Never>?
    private var latencies: [Double] = []
    private var runStart = Date()
    private var runDecisionSeconds = 0.0

    static let question = LayaQuestion.noul("Is this a clean placement?")

    /// `LAYA_DEMO_AUTORUN=1` loads the model and plays without clicks; `LAYA_DEMO_QUIT_AFTER=<s>`
    /// prints the stats and exits, which is how the demo is smoke-tested headlessly.
    func applyEnvironment() {
        let environment = ProcessInfo.processInfo.environment
        if let seedText = environment["LAYA_DEMO_SEED"], let value = UInt64(seedText) { seed = value }
        guard environment["LAYA_DEMO_AUTORUN"] == "1" else { return }
        loadModel()
        Task {
            while manager == nil && isLoading { try? await Task.sleep(nanoseconds: 100_000_000) }
            guard manager != nil else { return }
            reset()
            toggle()
            if let quitText = environment["LAYA_DEMO_QUIT_AFTER"], let seconds = Double(quitText) {
                try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
                print(
                    String(
                        format:
                            "autorun: %d pieces · %d lines · %d decisions · median %.2f ms · %.0f decisions/min · %@",
                        pieces, lines, decisions, medianMs, decisionsPerMinute,
                        (isOver ? "topped out" : "still playing") as NSString))
                exit(0)
            }
        }
    }

    func loadModel() {
        guard !isLoading, manager == nil else { return }
        isLoading = true
        loadStatus = "Loading laya (128-token bucket, CPU + Neural Engine)…"
        Task {
            do {
                let started = Date()
                let configuration = LayaManager.Configuration(lengths: [128])
                let loaded: LayaManager
                if let directory = ProcessInfo.processInfo.environment["LAYA_MODEL_DIR"], !directory.isEmpty {
                    loaded = try await LayaManager.load(
                        from: URL(fileURLWithPath: directory), configuration: configuration)
                } else {
                    loaded = try await LayaManager.load(configuration: configuration)
                }
                // First call pays the Core ML warm-up; keep it out of the game stats.
                _ = try await loaded.answer(state: "The piece leaves no holes.", question: Self.question)
                manager = loaded
                loadStatus = String(
                    format: "laya-multilingual · L128 · CPU + ANE · loaded in %.1f s", Date().timeIntervalSince(started)
                )
            } catch {
                errorMessage = error.localizedDescription
                loadStatus = "Load failed"
            }
            isLoading = false
        }
    }

    func reset() {
        task?.cancel()
        task = nil
        isRunning = false
        game = TetrisGame(seed: seed)
        rng = SplitMix64(seed: seed &+ 1)
        board = game.board
        candidates = []
        chosen = nil
        evaluating = nil
        currentPiece = "–"
        pieces = 0
        lines = 0
        isOver = false
        decisions = 0
        latencies = []
        medianMs = 0
        lastMs = 0
        decisionsPerMinute = 0
        promptTokens = 0
        bucket = 0
        runDecisionSeconds = 0
        log = []
    }

    func toggle() {
        if isRunning {
            task?.cancel()
            task = nil
            isRunning = false
            return
        }
        guard policy != .laya || manager != nil else {
            errorMessage = "Load the model first, or pick the heuristic policy."
            return
        }
        if isOver { reset() }
        isRunning = true
        runStart = Date()
        task = Task { [weak self] in
            await self?.play()
        }
    }

    private func play() async {
        while !Task.isCancelled, let piece = game.spawn() {
            currentPiece = piece.name
            let all = game.candidates(for: piece)
            guard !all.isEmpty else { break }
            candidates = []
            chosen = nil
            var best: (Float, TetrisGame.Candidate)?
            switch policy {
            case .laya:
                guard let manager else { break }
                for candidate in all {
                    if Task.isCancelled {
                        isRunning = false
                        return
                    }
                    evaluating = candidate
                    let sentence = game.describe(candidate, piece: piece)
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    do {
                        let answer = try await manager.answer(state: sentence, question: Self.question)
                        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                        record(ms: ms, tokens: answer.tokenCount, bucket: answer.bucketLength)
                        let p = answer.noul ?? 0
                        candidates.append(
                            ScoredCandidate(
                                id: candidate.id, candidate: candidate, sentence: sentence, probability: p,
                                milliseconds: ms))
                        if best == nil || p > best!.0 { best = (p, candidate) }
                    } catch {
                        errorMessage = error.localizedDescription
                        isRunning = false
                        return
                    }
                    if stepDelayMs > 0 {
                        try? await Task.sleep(nanoseconds: UInt64(stepDelayMs * 1_000_000))
                    }
                }
            case .heuristic:
                let pick = all.max { $0.features.heuristic < $1.features.heuristic }!
                best = (Float(pick.features.heuristic), pick)
                candidates = all.map {
                    ScoredCandidate(
                        id: $0.id, candidate: $0, sentence: game.describe($0, piece: piece),
                        probability: Float($0.features.heuristic), milliseconds: 0)
                }
            case .random:
                let pick = all[Int(rng.next() % UInt64(all.count))]
                best = (0, pick)
            }
            evaluating = nil
            guard let (score, pick) = best else { break }
            chosen = pick
            game.apply(pick)
            pieces += 1
            lines = game.linesCleared
            isOver = game.isOver
            let placed = String(
                format: "%@ → column %d rot %d · %@ %.3f · %d lines", piece.name as NSString, pick.column,
                pick.rotation,
                (policy == .laya ? "P(clean)" : "score") as NSString, score, lines)
            log.insert(placed, at: 0)
            if log.count > 12 { log.removeLast() }
            if pieceDelayMs > 0 {
                try? await Task.sleep(nanoseconds: UInt64(pieceDelayMs * 1_000_000))
            }
            board = game.board
            if isOver { break }
        }
        isRunning = false
        if isOver { log.insert("Topped out after \(pieces) pieces, \(lines) lines.", at: 0) }
    }

    private func record(ms: Double, tokens: Int, bucket: Int) {
        decisions += 1
        lastMs = ms
        latencies.append(ms)
        runDecisionSeconds += ms / 1000
        if latencies.count % 16 == 0 || latencies.count < 16 {
            let sorted = latencies.sorted()
            medianMs = sorted[sorted.count / 2]
        }
        let elapsed = Date().timeIntervalSince(runStart)
        decisionsPerMinute = elapsed > 0 ? Double(decisions) / elapsed * 60 : 0
        promptTokens = max(promptTokens, tokens)
        self.bucket = bucket
    }
}
