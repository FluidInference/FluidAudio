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
            OceanWaveView(meter: model.player.meter)
            if !model.references.isEmpty { referenceStrip }
            statsTable
            logView
        }
        .padding(16)
        .foregroundStyle(Nord.text)
        .tint(Nord.frost1)
        .background(Nord.bg)
        .preferredColorScheme(.dark)
        .onReceive(ticker) { _ in model.tick() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Pocket TTS small models on the Neural Engine").font(.title3).bold().foregroundStyle(Nord.textBright)
            Text(
                "kyutai pocket-tts 4.0.0 · 18m / 7m / 2m · FlowLM on the ANE (MLState), Mimi fp32 on the CPU "
                    + "or fused fp16 on the ANE (faster, hisses). Frames play as generated; the sea follows the speaker."
            )
            .font(.caption).foregroundStyle(Nord.textDim)
        }
    }

    private var controls: some View {
        HStack(spacing: 8) {
            Picker("Model", selection: Binding(get: { model.selectedModel }, set: { model.selectModel($0) })) {
                ForEach(PocketTtsSmallModel.allCases, id: \.self) { m in
                    Text(m.rawValue).tag(m)
                }
            }
            .frame(width: 190)
            Button(model.isReady ? "Reload" : "Load") { model.load() }
                .disabled(model.isLoading)
            Picker("Voice", selection: Binding(get: { model.selectedVoice }, set: { model.changeVoice($0) })) {
                ForEach(model.voices, id: \.self) { Text($0).tag($0) }
            }
            .frame(width: 130)
            .disabled(!model.isReady)
            Picker(
                "Mimi", selection: Binding(get: { model.mimiPrecision }, set: { model.setMimiPrecision($0) })
            ) {
                Text("fp32 CPU").tag(PocketTtsSmallMimiPrecision.fp32)
                Text("fp16 ANE (hiss)").tag(PocketTtsSmallMimiPrecision.fp16Fused)
            }
            .frame(width: 190)
            .disabled(model.isLoading)
            Spacer()
            if model.isLoading { ProgressView().controlSize(.small) }
            Text(model.status).font(.caption).foregroundStyle(Nord.green).lineLimit(1).truncationMode(.middle)
        }
    }

    private var textEntry: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextEditor(text: $model.text)
                .font(.body)
                .foregroundStyle(Nord.textBright)
                .frame(height: 70)
                .nordPanel()
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
                    .font(.caption2).foregroundStyle(Nord.purple).lineLimit(1)
                }
                Text("buffered \(max(0, model.player.scheduledFrames - model.player.playedFrames)) frames")
                    .font(.caption.monospacedDigit()).foregroundStyle(Nord.textDim)
            }
            Text("Press Speak again while audio plays: sentences queue on the same warm session.")
                .font(.caption2).foregroundStyle(Nord.textDim)
        }
    }

    /// Bundled upstream PyTorch takes next to a Core ML render of the same
    /// sentence and voice: the fairest A/B the demo can offer.
    private var referenceStrip: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Reference takes — upstream PyTorch fp32 on the CPU (18m) vs Core ML, same voice and text")
                .font(.caption).foregroundStyle(Nord.textDim)
            ForEach(model.references) { clip in
                HStack(spacing: 8) {
                    Text(clip.voice).font(.caption.bold()).foregroundStyle(Nord.purple).frame(
                        width: 60, alignment: .leading)
                    Text(clip.text).font(.caption).lineLimit(1).truncationMode(.tail)
                    Spacer()
                    Button("▶ PyTorch") { model.playReference(clip) }
                    Button("▶ Core ML") { model.speakReference(clip) }.disabled(!model.isReady)
                }
                .controlSize(.small)
            }
        }
    }

    private var statsTable: some View {
        Table(model.utterances.reversed()) {
            TableColumn("#") { u in Text("\(u.id)").monospacedDigit() }.width(28)
            TableColumn("text") { u in Text(u.text).lineLimit(1) }
            TableColumn("1st audio") { u in
                Text(u.firstAudioMs.map { String(format: "%.0f ms", $0) } ?? "…").monospacedDigit()
            }.width(90)
            TableColumn("frames") { u in Text("\(u.frames)").monospacedDigit() }.width(60)
            TableColumn("audio") { u in Text(String(format: "%.2f s", u.audioSeconds)).monospacedDigit() }.width(70)
            TableColumn("ms / frame") { u in Text(String(format: "%.2f", u.msPerFrame)).monospacedDigit() }.width(80)
            TableColumn("speed") { u in Text(String(format: "%.1fx", u.realTimeFactor)).monospacedDigit() }.width(70)
        }
        .frame(minHeight: 110, maxHeight: 180)
        .nordPanel()
    }

    private var logView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(model.log) { line in
                        Text(line.text).font(.system(.caption, design: .monospaced)).foregroundStyle(Nord.text).id(
                            line.id)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .nordPanel()
            .onChange(of: model.log.count) { _, _ in
                if let last = model.log.last { proxy.scrollTo(last.id, anchor: .bottom) }
            }
        }
    }
}
