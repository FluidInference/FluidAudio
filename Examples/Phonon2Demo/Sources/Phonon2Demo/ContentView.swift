import AppKit
import SwiftUI
import UniformTypeIdentifiers

private enum Palette {
    static let background = Color(red: 0.957, green: 0.957, blue: 0.945)
    static let card = Color(red: 0.980, green: 0.980, blue: 0.972)
    static let border = Color(red: 0.86, green: 0.86, blue: 0.84)
    static let accent = Color(red: 0.33, green: 0.45, blue: 0.20)
    static let text = Color(red: 0.10, green: 0.10, blue: 0.10)
    static let secondary = Color(red: 0.42, green: 0.42, blue: 0.40)
    static let highlight = Color(red: 0.33, green: 0.45, blue: 0.20).opacity(0.18)
}

struct ContentView: View {
    @EnvironmentObject private var model: DemoModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                header
                filePicker
                actions
                statsRow
                ComparisonCard(
                    runs: model.runs, activeModel: model.activeModel, liveWallSeconds: model.liveWallSeconds)
                if !model.paragraphs.isEmpty {
                    TranscriptCard(paragraphs: model.paragraphs, text: model.transcriptText)
                        .environmentObject(model.player)
                }
            }
            .padding(.horizontal, 44)
            .padding(.vertical, 36)
            .frame(maxWidth: 980, alignment: .leading)
            .frame(maxWidth: .infinity)
        }
        .background(Palette.background)
        .foregroundStyle(Palette.text)
        .task { model.applyLaunchArguments() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Phonon-2 on FluidAudio")
                .font(.system(size: 44, weight: .bold))
            Text("Fermion Research\u{2019}s five-value Parakeet, running on the Neural Engine through Core ML.")
                .font(.system(size: 17))
            Text(
                "This demo runs \(model.choice.title) (\(model.choice.detail)) entirely on-device, then times Parakeet TDT v3 on the same file. Audio never leaves your Mac."
            )
            .font(.system(size: 17))
            Text("Models download the first time you transcribe, then stay cached.")
                .font(.system(size: 17))
                .foregroundStyle(Palette.secondary)
        }
    }

    private var filePicker: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 4) {
                Text("Pick an audio file").foregroundStyle(Palette.secondary)
                if !model.samples.isEmpty {
                    Text("(or try").foregroundStyle(Palette.secondary)
                    ForEach(Array(model.samples.enumerated()), id: \.element.url) { index, sample in
                        if index > 0 { Text("or").foregroundStyle(Palette.secondary) }
                        Button {
                            model.select(sample.url)
                        } label: {
                            Text(sample.label).underline().foregroundStyle(Palette.accent)
                        }
                        .buttonStyle(.plain)
                        .disabled(model.isBusy)
                    }
                    Text(")").foregroundStyle(Palette.secondary)
                }
            }
            .font(.system(size: 16))

            HStack(spacing: 14) {
                Button("Choose File") { chooseFile() }
                    .buttonStyle(SecondaryButtonStyle())
                    .disabled(model.isBusy)
                Text(model.fileURL?.lastPathComponent ?? "No file selected")
                    .font(.system(size: 17))
                    .foregroundStyle(model.fileURL == nil ? Palette.secondary : Palette.text)
            }
        }
    }

    private var actions: some View {
        HStack(spacing: 14) {
            Button {
                model.transcribe()
            } label: {
                HStack(spacing: 8) {
                    if model.isBusy {
                        ProgressView().controlSize(.small).tint(.white)
                    }
                    Text(buttonTitle)
                }
            }
            .buttonStyle(PrimaryButtonStyle())
            .disabled(model.fileURL == nil || model.isBusy)

            Button("Cancel") { model.cancel() }
                .buttonStyle(SecondaryButtonStyle())
                .disabled(!model.isBusy)

            Button("Reset") { model.reset() }
                .buttonStyle(SecondaryButtonStyle())
                .keyboardShortcut("r", modifiers: .command)
                .help("Clear the results and timings (⌘R); models stay loaded")
                .disabled(model.isBusy || !model.hasResults)

            Toggle(isOn: $model.compareWithV3) {
                Text("Also time Parakeet TDT v3 as a reference")
                    .font(.system(size: 15))
                    .foregroundStyle(Palette.secondary)
            }
            .toggleStyle(.checkbox)
            .tint(Palette.accent)
            .disabled(model.isBusy)
            .padding(.leading, 8)

            if case .failed(let message) = model.phase {
                Text(message).foregroundStyle(.red).font(.system(size: 14))
            }
        }
    }

    private var buttonTitle: String {
        switch model.phase {
        case .loadingModel: return "Loading model…"
        case .transcribing: return "Transcribing…"
        default: return "Transcribe"
        }
    }

    private var statsRow: some View {
        HStack(spacing: 0) {
            StatCell(
                title: "Audio length",
                value: Self.clock(model.stats?.audioSeconds ?? model.fileDuration ?? 0))
            Divider().frame(height: 84)
            StatCell(
                title: "Transcription wall time",
                value: String(format: "%.2fs", model.stats?.wallSeconds ?? model.liveWallSeconds))
            Divider().frame(height: 84)
            StatCell(
                title: "Speed",
                value: String(format: "%.1fx real-time", model.stats?.speed ?? 0))
        }
        .background(RoundedRectangle(cornerRadius: 12).fill(Palette.card))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Palette.border, lineWidth: 1))
    }

    private func chooseFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.audio]
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            model.select(url)
        }
    }

    static func clock(_ seconds: TimeInterval) -> String {
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 { return "\(hours)h \(minutes)m \(secs)s" }
        return "\(minutes)m \(secs)s"
    }
}

private struct ComparisonCard: View {
    let runs: [DemoModelChoice: RunStats]
    let activeModel: DemoModelChoice?
    let liveWallSeconds: TimeInterval

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("SAME FILE, SAME MAC")
                .font(.system(size: 13, weight: .semibold))
                .tracking(1.4)
                .foregroundStyle(Palette.secondary)
            ForEach(DemoModelChoice.allCases) { choice in
                if let run = runs[choice] {
                    row(
                        choice.title, wall: String(format: "%.2fs", run.wallSeconds),
                        speed: String(format: "%.1fx real-time", run.speed), note: note(for: choice, run))
                } else {
                    // Not run yet: zeros; while running, the wall clock counts up from zero.
                    let wall = choice == activeModel ? liveWallSeconds : 0
                    row(
                        choice.title, wall: String(format: "%.2fs", wall), speed: "0.0x real-time",
                        note: choice == .v3 ? "reference" : "")
                }
            }
        }
        .padding(.horizontal, 26)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 12).fill(Palette.card))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Palette.border, lineWidth: 1))
    }

    private func note(for choice: DemoModelChoice, _ run: RunStats) -> String {
        guard choice != .v3, let reference = runs[.v3] else { return choice == .v3 ? "reference" : "" }
        let ratio = reference.wallSeconds / max(run.wallSeconds, 1e-6)
        return ratio >= 1
            ? String(format: "%.2f× faster than v3", ratio) : String(format: "%.2f× slower than v3", 1 / ratio)
    }

    private func row(_ title: String, wall: String, speed: String, note: String) -> some View {
        HStack {
            Text(title).font(.system(size: 17, weight: .semibold)).frame(width: 200, alignment: .leading)
            Text(wall).font(.system(size: 17, design: .rounded)).monospacedDigit()
                .frame(width: 110, alignment: .leading)
            Text(speed).font(.system(size: 17, design: .rounded)).monospacedDigit()
                .frame(width: 210, alignment: .leading)
            Text(note).font(.system(size: 15)).foregroundStyle(Palette.secondary)
            Spacer()
        }
    }
}

private struct StatCell: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(title.uppercased())
                .font(.system(size: 13, weight: .semibold))
                .tracking(1.4)
                .foregroundStyle(Palette.secondary)
            Text(value)
                .font(.system(size: 24, weight: .regular, design: .rounded))
                .monospacedDigit()
        }
        .padding(.horizontal, 26)
        .padding(.vertical, 22)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct TranscriptCard: View {
    let paragraphs: [Paragraph]
    let text: String
    @EnvironmentObject private var player: TranscriptPlayer

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Transcript").font(.system(size: 24, weight: .bold))
                    Text("Hover for timestamps. Click a word to seek.")
                        .font(.system(size: 15))
                        .foregroundStyle(Palette.secondary)
                }
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                } label: {
                    Image(systemName: "doc.on.doc").font(.system(size: 17))
                }
                .buttonStyle(SecondaryButtonStyle())
                .help("Copy transcript")
            }
            .padding(24)

            Divider()

            PlayerBar().padding(.horizontal, 24).padding(.top, 18)

            LazyVStack(alignment: .leading, spacing: 14) {
                ForEach(paragraphs) { paragraph in
                    ParagraphView(paragraph: paragraph)
                }
            }
            .padding(24)
        }
        .background(RoundedRectangle(cornerRadius: 12).fill(Palette.card))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Palette.border, lineWidth: 1))
    }
}

private struct PlayerBar: View {
    @EnvironmentObject private var player: TranscriptPlayer

    var body: some View {
        HStack(spacing: 16) {
            Button {
                player.toggle()
            } label: {
                Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                    .font(.system(size: 16))
                    .frame(width: 22)
            }
            .buttonStyle(.plain)
            Text("\(Self.time(player.currentTime)) / \(Self.time(player.duration))")
                .font(.system(size: 15))
                .monospacedDigit()
            Slider(
                value: Binding(
                    get: { player.currentTime },
                    set: { player.seek(to: $0, andPlay: false) }),
                in: 0...max(player.duration, 1))
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 12)
        .background(Capsule().fill(Color(red: 0.93, green: 0.93, blue: 0.92)))
    }

    static func time(_ seconds: TimeInterval) -> String {
        let total = Int(seconds)
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}

private struct ParagraphView: View {
    let paragraph: Paragraph
    @EnvironmentObject private var player: TranscriptPlayer

    var body: some View {
        FlowLayout(spacing: 5, lineSpacing: 5) {
            ForEach(paragraph.words) { word in
                let active =
                    player.isPlaying && player.currentTime >= word.start && player.currentTime < word.end + 0.08
                Text(word.text)
                    .font(.system(size: 16))
                    .padding(.horizontal, 2)
                    .background(RoundedRectangle(cornerRadius: 3).fill(active ? Palette.highlight : .clear))
                    .contentShape(Rectangle())
                    .onTapGesture { player.seek(to: word.start) }
                    .help(String(format: "%.2fs – %.2fs", word.start, word.end))
            }
        }
    }
}

/// Wraps its subviews left to right like inline text.
private struct FlowLayout: Layout {
    var spacing: CGFloat
    var lineSpacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? 800
        var x: CGFloat = 0
        var y: CGFloat = 0
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width {
                y += lineHeight + lineSpacing
                x = 0
                lineHeight = 0
            }
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
        return CGSize(width: width, height: y + lineHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX
        var y = bounds.minY
        var lineHeight: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > bounds.minX, x + size.width > bounds.maxX {
                y += lineHeight + lineSpacing
                x = bounds.minX
                lineHeight = 0
            }
            subview.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
            x += size.width + spacing
            lineHeight = max(lineHeight, size.height)
        }
    }
}

private struct PrimaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 17, weight: .medium))
            .foregroundStyle(.white)
            .padding(.horizontal, 22)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(Palette.accent.opacity(isEnabled ? (configuration.isPressed ? 0.8 : 1) : 0.5)))
    }
}

private struct SecondaryButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 17))
            .foregroundStyle(isEnabled ? Palette.text : Palette.secondary)
            .padding(.horizontal, 18)
            .padding(.vertical, 11)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(configuration.isPressed ? Palette.border : Color(red: 0.92, green: 0.92, blue: 0.91))
            )
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(Palette.border, lineWidth: 1))
    }
}
