import Combine
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: DemoModel
    private let ticker = Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            OceanWaveView(meter: model.player.meter)
            if !model.references.isEmpty { referenceStrip }
            statsTable
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.top, 28)  // room for the traffic lights with the title bar hidden
        .padding(.bottom, 12)
        .foregroundStyle(Nord.text)
        .tint(Nord.frost1)
        .background(Nord.bg)
        .preferredColorScheme(.dark)
        .onReceive(ticker) { _ in model.tick() }
    }

    /// Bundled upstream PyTorch takes next to a Core ML render of the same
    /// sentence and voice: the fairest A/B the demo can offer.
    private var referenceStrip: some View {
        VStack(alignment: .leading, spacing: 4) {
            ForEach(model.references) { clip in
                let active = model.activeReference?.file == clip.file ? model.activeReference?.source : nil
                HStack(alignment: .top, spacing: 8) {
                    Text(clip.voice).font(.caption.bold()).foregroundStyle(Nord.purple).frame(
                        width: 60, alignment: .leading)
                    Text(clip.text).font(.caption)
                        .foregroundStyle(active == nil ? Nord.text : Nord.yellow)
                        .fixedSize(horizontal: false, vertical: true)
                    Spacer()
                    if let active {
                        Text("◉ \(active.rawValue)").font(.caption2.bold())
                            .foregroundStyle(active == .pytorch ? Nord.orange : Nord.green)
                    }
                    Button("▶ PyTorch") { model.playReference(clip) }
                        .tint(active == .pytorch ? Nord.orange : Nord.frost1)
                    Button("▶ Core ML") { model.speakReference(clip) }.disabled(!model.isReady || model.isSpeaking)
                        .tint(active == .coreml ? Nord.green : Nord.frost1)
                }
                .controlSize(.small)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .background(active == nil ? Color.clear : Nord.panelRaised)
                .clipShape(RoundedRectangle(cornerRadius: 5))
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
