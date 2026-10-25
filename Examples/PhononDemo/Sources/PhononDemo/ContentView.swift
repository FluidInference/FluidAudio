import Combine
import FluidAudio
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var model: DemoModel
    private let ticker = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    private var palette: [Color] {
        model.selectedVoice.isCharacter
            ? [Nord.purple, Nord.frost0, Nord.yellow]  // aurora for the characters
            : [Nord.frost1, Nord.frost2, Nord.frost3]
    }

    var body: some View {
        VStack(spacing: 14) {
            header
            ZStack {
                RingWaveView(meter: model.player.meter, palette: palette)
                centerLabel
            }
            .frame(height: 320)
            Text(model.spokenText.isEmpty ? " " : model.spokenText)
                .font(.callout)
                .foregroundStyle(model.isSpeaking ? Nord.yellow : Nord.textDim)
                .multilineTextAlignment(.center)
                .lineLimit(3)
                .frame(maxWidth: 480)
                .animation(.easeInOut(duration: 0.3), value: model.isSpeaking)
            voiceRows
            composer
            stats
        }
        .padding(.horizontal, 20)
        .padding(.top, 30)  // room for the traffic lights with the title bar hidden
        .padding(.bottom, 14)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .foregroundStyle(Nord.text)
        .tint(Nord.frost1)
        .background(Nord.bg.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .onReceive(ticker) { _ in model.tick() }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            Text("Phonon").font(.title2.weight(.semibold)).foregroundStyle(Nord.textBright)
            Text("Gradium · 31M · Neural Engine").font(.caption).foregroundStyle(Nord.textDim)
            Spacer()
            Text(model.status).font(.caption).foregroundStyle(model.isReady ? Nord.green : Nord.orange)
        }
    }

    private var centerLabel: some View {
        VStack(spacing: 2) {
            Text(model.selectedVoice.label)
                .font(.system(size: 22, weight: .semibold, design: .rounded))
                .foregroundStyle(Nord.textBright)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 150)
            Text(model.selectedVoice.detail).font(.caption2).foregroundStyle(Nord.textDim)
        }
    }

    private var voiceRows: some View {
        VStack(alignment: .leading, spacing: 6) {
            chipRow(model.featuredVoices, accent: Nord.frost1) {
                Menu("More") {
                    ForEach(model.otherVoices) { v in
                        Button("\(v.label)  ·  \(v.detail)") { model.select(v) }
                    }
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }
            chipRow(model.featuredCharacters, accent: Nord.purple) { EmptyView() }
        }
        .disabled(!model.isReady)
    }

    private func chipRow<Trailing: View>(
        _ choices: [VoiceChoice], accent: Color, @ViewBuilder trailing: () -> Trailing
    ) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(choices) { v in
                    let on = v == model.selectedVoice
                    Button {
                        model.select(v)
                    } label: {
                        Text(v.label)
                            .font(.caption.weight(on ? .semibold : .regular))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 5)
                            .foregroundStyle(on ? Nord.bg : Nord.text)
                            .background(on ? accent : Nord.panel)
                            .clipShape(Capsule())
                            .overlay(Capsule().stroke(on ? accent : Nord.border))
                    }
                    .buttonStyle(.plain)
                }
                trailing()
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 6) {
            TextEditor(text: $model.text)
                .font(.callout)
                .foregroundStyle(Nord.textBright)
                .frame(height: 62)
                .padding(6)
                .nordPanel()
            HStack(spacing: 6) {
                ForEach(DemoModel.presets, id: \.title) { preset in
                    Button(preset.title) { model.text = preset.text }
                        .controlSize(.small)
                }
                Spacer()
                Button(model.isSpeaking ? "■ Stop" : "▶ Speak") { model.toggle() }
                    .keyboardShortcut(.return, modifiers: .command)
                    .buttonStyle(.borderedProminent)
                    .tint(model.isSpeaking ? Nord.red : Nord.green)
                    .disabled(!model.isReady)
            }
        }
    }

    private var stats: some View {
        HStack(spacing: 14) {
            stat("first audio", model.firstAudioMs.map { String(format: "%.0f ms", $0) })
            stat("per 80 ms frame", model.msPerFrame.map { String(format: "%.2f ms", $0) })
            stat("real time", model.realTime.map { String(format: "%.1fx", $0) })
            stat("load", model.loadSeconds.map { String(format: "%.2f s", $0) })
            Spacer()
        }
        .font(.caption.monospacedDigit())
    }

    private func stat(_ label: String, _ value: String?) -> some View {
        HStack(spacing: 4) {
            Text(label).foregroundStyle(Nord.textDim)
            Text(value ?? "—").foregroundStyle(Nord.textBright)
        }
    }
}
