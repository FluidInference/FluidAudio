import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var model: GameModel

    var body: some View {
        HStack(alignment: .top, spacing: 16) {
            VStack(alignment: .leading, spacing: 12) {
                header
                BoardView(board: model.board, chosen: model.chosen, evaluating: model.evaluating)
                    .frame(width: 300, height: 600)
                controls
            }
            VStack(alignment: .leading, spacing: 12) {
                statTiles
                consoleView
                logView
            }
            .frame(minWidth: 560)
        }
        .padding(16)
        .alert(
            "laya",
            isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("laya plays Tetris").font(.title2.bold())
            Text(
                "Every legal landing is described in one sentence; laya answers “Is this a clean placement?” and the highest P(true) is played."
            )
            .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text(model.loadStatus).font(.caption.monospaced()).foregroundStyle(
                model.manager == nil ? Color.secondary : Color.green)
        }
        .frame(width: 300, alignment: .leading)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Button(model.manager == nil ? "Load model" : "Loaded") { model.loadModel() }
                    .disabled(model.isLoading || model.manager != nil)
                Button(model.isRunning ? "Pause" : (model.isOver ? "Restart" : "Play")) { model.toggle() }
                    .keyboardShortcut(.space, modifiers: [])
                    .disabled(model.policy == .laya && model.manager == nil)
                Button("Reset") { model.reset() }.disabled(model.isRunning)
            }
            Picker("Policy", selection: $model.policy) {
                ForEach(GameModel.Policy.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .disabled(model.isRunning)
            HStack {
                Text("Seed").font(.caption)
                TextField("seed", value: $model.seed, format: .number).frame(width: 70).disabled(model.isRunning)
                Spacer()
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(String(format: "Delay per scored landing: %.0f ms", model.stepDelayMs)).font(.caption)
                Slider(value: $model.stepDelayMs, in: 0...300, step: 10)
                Text(String(format: "Pause per piece: %.0f ms", model.pieceDelayMs)).font(.caption)
                Slider(value: $model.pieceDelayMs, in: 0...1000, step: 20)
            }
        }
        .frame(width: 300)
    }

    private var statTiles: some View {
        HStack(spacing: 10) {
            StatTile(
                title: "per decision", value: String(format: "%.2f ms", model.lastMs),
                detail: String(format: "median %.2f ms", model.medianMs), accent: .orange)
            StatTile(
                title: "decisions / min", value: String(format: "%.0f", model.decisionsPerMinute),
                detail: "\(model.decisions) total", accent: .blue)
            StatTile(
                title: "lines", value: "\(model.lines)", detail: "\(model.pieces) pieces · \(model.currentPiece)",
                accent: .green)
            StatTile(
                title: "bucket", value: model.bucket > 0 ? "L\(model.bucket)" : "–",
                detail: "≤ \(model.promptTokens) tokens · ANE", accent: .purple)
        }
    }

    private var consoleView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Decisions for the current piece").font(.headline)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 3) {
                    let best = model.candidates.map(\.probability).max() ?? 0
                    ForEach(model.candidates) { scored in
                        HStack(spacing: 8) {
                            ProbabilityBar(
                                value: scored.probability, isBest: scored.probability == best && model.policy == .laya
                            )
                            .frame(width: 90, height: 12)
                            Text(
                                model.policy == .laya
                                    ? String(format: "%.3f", scored.probability)
                                    : String(format: "%.1f", scored.probability)
                            )
                            .font(.caption.monospaced())
                            .frame(width: 44, alignment: .trailing)
                            Text(scored.sentence)
                                .font(.caption)
                                .lineLimit(1)
                                .foregroundStyle(scored.probability == best ? .primary : .secondary)
                            Spacer()
                            if scored.milliseconds > 0 {
                                Text(String(format: "%.1f ms", scored.milliseconds)).font(.caption2.monospaced())
                                    .foregroundStyle(.red)
                            }
                        }
                    }
                }
                .padding(6)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .frame(minHeight: 300)
        }
    }

    private var logView: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Placed").font(.headline)
            VStack(alignment: .leading, spacing: 2) {
                ForEach(Array(model.log.enumerated()), id: \.offset) { _, line in
                    Text(line).font(.caption.monospaced()).lineLimit(1)
                }
            }
            .frame(maxWidth: .infinity, minHeight: 120, alignment: .topLeading)
            .padding(6)
            .background(Color(nsColor: .textBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}

struct StatTile: View {
    let title: String
    let value: String
    let detail: String
    let accent: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title.uppercased()).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.system(size: 22, weight: .semibold, design: .rounded)).foregroundStyle(accent)
            Text(detail).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(accent.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct ProbabilityBar: View {
    let value: Float
    let isBest: Bool

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3).fill(Color.secondary.opacity(0.15))
                RoundedRectangle(cornerRadius: 3)
                    .fill(isBest ? Color.green : Color.blue.opacity(0.6))
                    .frame(width: geometry.size.width * CGFloat(max(0, min(1, value))))
            }
        }
    }
}

struct BoardView: View {
    let board: [[Bool]]
    let chosen: TetrisGame.Candidate?
    let evaluating: TetrisGame.Candidate?

    var body: some View {
        Canvas { context, size in
            let columns = TetrisGame.width
            let rows = TetrisGame.height
            let cell = min(size.width / CGFloat(columns), size.height / CGFloat(rows))
            let originX = (size.width - cell * CGFloat(columns)) / 2
            let originY = (size.height - cell * CGFloat(rows)) / 2
            func rect(_ x: Int, _ y: Int) -> CGRect {
                CGRect(x: originX + CGFloat(x) * cell, y: originY + CGFloat(y) * cell, width: cell, height: cell)
                    .insetBy(dx: 1, dy: 1)
            }
            context.fill(
                Path(CGRect(x: originX, y: originY, width: cell * CGFloat(columns), height: cell * CGFloat(rows))),
                with: .color(Color.black.opacity(0.85)))
            for y in 0..<rows {
                for x in 0..<columns where board[y][x] {
                    context.fill(Path(roundedRect: rect(x, y), cornerRadius: 2), with: .color(Color.cyan.opacity(0.85)))
                }
            }
            if let evaluating {
                for (x, y) in evaluating.cells where y >= 0 {
                    context.stroke(
                        Path(roundedRect: rect(x, y), cornerRadius: 2), with: .color(Color.orange.opacity(0.9)),
                        lineWidth: 1.5)
                }
            }
            if let chosen {
                for (x, y) in chosen.cells where y >= 0 {
                    context.fill(Path(roundedRect: rect(x, y), cornerRadius: 2), with: .color(Color.green.opacity(0.9)))
                }
            }
        }
        .background(Color.black.opacity(0.9))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
