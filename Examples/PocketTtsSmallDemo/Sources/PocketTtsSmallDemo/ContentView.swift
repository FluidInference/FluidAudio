import FluidAudio
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: DemoModel
    private let ticker = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            controls
            statusLine
            textEntry
            OceanWaveView(meter: model.player.meter)
            if !model.references.isEmpty { referenceStrip }
            statsTable
            Spacer(minLength: 0)
        }
        .padding(16)
        .foregroundStyle(Nord.text)
        .tint(Nord.frost1)
        .background(Nord.bg)
        .preferredColorScheme(.dark)
        .onReceive(ticker) { _ in model.tick() }
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
        }
    }

    private var statusLine: some View {
        Text(model.status).font(.caption).foregroundStyle(Nord.green).lineLimit(1).truncationMode(.middle)
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
                HStack(alignment: .top, spacing: 8) {
                    Text(clip.voice).font(.caption.bold()).foregroundStyle(Nord.purple).frame(
                        width: 60, alignment: .leading)
                    Text(clip.text).font(.caption).fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    Button("▶ PyTorch") { model.playReference(clip) }
                    Button("▶ Core ML") { model.speakReference(clip) }.disabled(!model.isReady)
                }
                .controlSize(.small)
            }
        }
    }

    /// Content-sized rows (no empty filler rows), full text, RTFx.
    private var statsTable: some View {
        VStack(spacing: 0) {
            statsRow(
                id: "#", text: "text", first: "1st audio", frames: "frames", audio: "audio",
                perFrame: "ms / frame", rtfx: "RTFx", header: true)
            Divider().overlay(Nord.border)
            if model.utterances.isEmpty {
                Text("No utterances yet.").font(.caption).foregroundStyle(Nord.textDim).padding(8)
            }
            ForEach(model.utterances.reversed()) { u in
                statsRow(
                    id: "\(u.id)", text: u.text,
                    first: u.firstAudioMs.map { String(format: "%.0f ms", $0) } ?? "…",
                    frames: "\(u.frames)", audio: String(format: "%.2f s", u.audioSeconds),
                    perFrame: String(format: "%.2f", u.msPerFrame),
                    rtfx: String(format: "%.1fx", u.realTimeFactor), header: false)
            }
        }
        .nordPanel()
    }

    private func statsRow(
        id: String, text: String, first: String, frames: String, audio: String, perFrame: String, rtfx: String,
        header: Bool
    ) -> some View {
        let font: Font = header ? .caption.bold() : .caption.monospacedDigit()
        let color = header ? Nord.textDim : Nord.text
        return HStack(alignment: .top, spacing: 8) {
            Text(id).font(font).foregroundStyle(color).frame(width: 24, alignment: .leading)
            Text(text).font(header ? font : .caption).foregroundStyle(header ? color : Nord.textBright)
                .frame(maxWidth: .infinity, alignment: .leading).fixedSize(horizontal: false, vertical: true)
            Text(first).font(font).foregroundStyle(color).frame(width: 62, alignment: .trailing)
            Text(frames).font(font).foregroundStyle(color).frame(width: 48, alignment: .trailing)
            Text(audio).font(font).foregroundStyle(color).frame(width: 54, alignment: .trailing)
            Text(perFrame).font(font).foregroundStyle(color).frame(width: 66, alignment: .trailing)
            Text(rtfx).font(font).foregroundStyle(color).frame(width: 50, alignment: .trailing)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
    }
}
