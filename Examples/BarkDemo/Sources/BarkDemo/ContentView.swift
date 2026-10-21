import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject var model: DemoModel
    @State private var picking = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            TapeView(
                peaks: model.waveform,
                progress: model.fileDuration > 0 ? model.decodedSeconds / model.fileDuration : 0
            )
            .frame(height: 72)
            stats
            Typewriter(text: model.showFinal ? (model.finalText ?? model.typed) : model.typed, typing: model.isTyping)
        }
        .padding(24)
        .background(Color(nsColor: .windowBackgroundColor))
        .fileImporter(isPresented: $picking, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result { model.select(url) }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 12) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Bark Typewriter").font(.system(size: 22, weight: .semibold))
                Text(
                    model.fileURL.map { "\($0.lastPathComponent) · \(DemoModel.clock(model.fileDuration))" }
                        ?? "no file"
                )
                .font(.callout).foregroundStyle(.secondary)
            }
            Spacer()
            Picker("Model", selection: $model.choice) {
                ForEach(DemoModelChoice.allCases) { Text($0.title).tag($0) }
            }
            .frame(width: 220)
            .disabled(model.isBusy)
            Button("Open…") { picking = true }.disabled(model.isBusy)
            if model.isBusy {
                Button("Stop") { model.cancel() }
            } else {
                Button("Transcribe") { model.start() }
                    .keyboardShortcut(.return)
                    .buttonStyle(.borderedProminent)
                    .disabled(model.fileURL == nil)
            }
        }
    }

    private var stats: some View {
        HStack(spacing: 28) {
            Stat(
                label: "audio done",
                value: "\(DemoModel.clock(model.decodedSeconds)) / \(DemoModel.clock(model.fileDuration))")
            Stat(label: "wall clock", value: String(format: "%.1f s", model.wallSeconds))
            Stat(label: "speed", value: model.wallSeconds > 0.2 ? String(format: "%.0f×", model.liveSpeed) : "–")
            Stat(label: "windows", value: model.windowCount > 0 ? "\(model.windowsDone) / \(model.windowCount)" : "–")
            Spacer()
            Group {
                switch model.phase {
                case .failed(let message): Text(message).foregroundStyle(.red)
                case .loadingModel: Label(model.status, systemImage: "hourglass")
                default: Text(model.status.isEmpty ? model.choice.detail : model.status)
                }
            }
            .font(.callout).foregroundStyle(.secondary).lineLimit(2)
        }
    }
}

private struct Stat: View {
    let label: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label.uppercased()).font(.caption2.weight(.medium)).foregroundStyle(.secondary)
            Text(value).font(.system(size: 20, weight: .semibold, design: .monospaced)).monospacedDigit()
        }
    }
}

/// The file as a strip of peaks; the part the model has finished is lit, with a playhead at its edge.
private struct TapeView: View {
    let peaks: [Float]
    let progress: Double

    var body: some View {
        Canvas { context, size in
            let n = max(peaks.count, 1)
            let barWidth = size.width / CGFloat(n)
            let mid = size.height / 2
            let head = size.width * CGFloat(min(max(progress, 0), 1))
            for (i, peak) in peaks.enumerated() {
                let x = CGFloat(i) * barWidth
                let h = max(2, CGFloat(peak) * (size.height - 8))
                let rect = CGRect(x: x, y: mid - h / 2, width: max(1, barWidth - 1), height: h)
                context.fill(Path(rect), with: .color(x < head ? .accentColor : .secondary.opacity(0.25)))
            }
            if progress > 0 && progress < 1 {
                context.fill(Path(CGRect(x: head - 1, y: 0, width: 2, height: size.height)), with: .color(.primary))
            }
        }
        .animation(.linear(duration: 0.1), value: progress)
        .padding(.vertical, 4)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
    }
}

/// Monospaced text that grows from the end, with a blinking cursor while the model is still running.
/// Only the tail is rendered while typing, so an hour of text stays cheap at 60 frames per second.
private struct Typewriter: View {
    let text: String
    let typing: Bool
    @State private var cursorOn = true
    private let blink = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    (Text(typing ? tail : text) + Text(typing && cursorOn ? "▌" : " ").foregroundColor(.accentColor))
                        .font(.system(size: 17, design: .monospaced))
                        .lineSpacing(5)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Color.clear.frame(height: 1).id("end")
                }
                .padding(18)
            }
            .background(RoundedRectangle(cornerRadius: 10).fill(Color(nsColor: .textBackgroundColor)))
            .onChange(of: text) { if typing { proxy.scrollTo("end", anchor: .bottom) } }
        }
        .onReceive(blink) { _ in cursorOn.toggle() }
    }

    private var tail: String {
        guard text.count > 4000 else { return text }
        let start = text.index(text.endIndex, offsetBy: -4000)
        let cut = text[start...].firstIndex(of: " ") ?? start
        return "…" + text[cut...]
    }
}
