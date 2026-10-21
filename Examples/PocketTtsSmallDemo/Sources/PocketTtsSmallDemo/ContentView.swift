import FluidAudio
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: DemoModel
    private let ticker = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            controls
            textEntry
            statsTable
            logView
        }
        .padding(16)
        .onReceive(ticker) { _ in model.tick() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Pocket TTS small models on the Neural Engine").font(.title2).bold()
            Text(
                "kyutai pocket-tts 4.0.0 · 18m / 7m / 2m English · one Core ML MLState package per model: "
                    + "FlowLM step + flow head + Mimi decoder in a single dispatch per 80 ms frame, 100% ANE. "
                    + "Frames play the moment they are generated."
            )
            .font(.callout).foregroundStyle(.secondary)
        }
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Picker("Model", selection: Binding(get: { model.selectedModel }, set: { model.selectModel($0) })) {
                ForEach(PocketTtsSmallModel.allCases, id: \.self) { m in
                    Text(m.rawValue).tag(m)
                }
            }
            .frame(width: 300)
            Button(model.isReady ? "Reload" : "Load") { model.load() }
                .disabled(model.isLoading)
            Picker("Voice", selection: Binding(get: { model.selectedVoice }, set: { model.changeVoice($0) })) {
                ForEach(model.voices, id: \.self) { Text($0).tag($0) }
            }
            .frame(width: 200)
            .disabled(!model.isReady)
            Spacer()
            if model.isLoading { ProgressView().controlSize(.small) }
            Text(model.status).font(.callout).foregroundStyle(.secondary).lineLimit(1)
        }
    }

    private var textEntry: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextEditor(text: $model.text)
                .font(.body)
                .frame(height: 70)
                .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            HStack {
                Button("Speak (⌘↩)") { model.speak() }
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(!model.isReady || model.text.trimmingCharacters(in: .whitespaces).isEmpty)
                Button("Stop") { model.stop() }.disabled(!model.isSpeaking && model.player.scheduledFrames == 0)
                Button("Clear") { model.clearStats() }
                Spacer()
                if let arch = model.arch {
                    Text(
                        "d_model \(arch.dModel) · \(arch.numLayers) layers · \(arch.numHeads)/\(arch.numKvHeads) heads · "
                            + "flow \(arch.flowType) · EOS > \(String(format: "%.1f", arch.eosThreshold))"
                    )
                    .font(.caption).foregroundStyle(.secondary)
                }
                Text("buffered \(max(0, model.player.scheduledFrames - model.player.playedFrames)) frames")
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Text(
                "Press Speak repeatedly while audio plays: each sentence is enqueued on the same warm session (voice KV snapshot + MLState stay resident)."
            )
            .font(.caption).foregroundStyle(.tertiary)
        }
    }

    private var statsTable: some View {
        Table(model.utterances.reversed()) {
            TableColumn("#") { u in Text("\(u.id)").monospacedDigit() }.width(28)
            TableColumn("text") { u in Text(u.text).lineLimit(1) }
            TableColumn("first audio") { u in
                Text(u.firstAudioMs.map { String(format: "%.0f ms", $0) } ?? "…").monospacedDigit()
            }.width(90)
            TableColumn("frames") { u in Text("\(u.frames)").monospacedDigit() }.width(60)
            TableColumn("audio") { u in Text(String(format: "%.2f s", u.audioSeconds)).monospacedDigit() }.width(70)
            TableColumn("ms / frame") { u in Text(String(format: "%.2f", u.msPerFrame)).monospacedDigit() }.width(80)
            TableColumn("speed") { u in Text(String(format: "%.1fx", u.realTimeFactor)).monospacedDigit() }.width(70)
        }
        .frame(minHeight: 140, maxHeight: 220)
    }

    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(model.log) { line in
                        Text(line.text).font(.system(.caption, design: .monospaced)).id(line.id)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .background(Color(nsColor: .textBackgroundColor))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.quaternary))
            .onChange(of: model.log.count) { _, _ in
                if let last = model.log.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
    }
}
