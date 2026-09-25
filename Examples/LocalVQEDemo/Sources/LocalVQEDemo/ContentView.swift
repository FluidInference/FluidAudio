import FluidAudio
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var model: DemoModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            ModelBar()
            TabView {
                FileModeView().tabItem { Text("Files (offline)") }
                LiveModeView().tabItem { Text("Live (mic + speaker)") }
            }
        }
        .padding(20)
        .alert(
            "Error",
            isPresented: Binding(get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

private struct ModelBar: View {
    @EnvironmentObject private var model: DemoModel

    var body: some View {
        HStack(spacing: 12) {
            Text("LocalVQE").font(.title2).bold()
            Picker("Variant", selection: $model.variant) {
                Text("v1.3 (4.8M)").tag(LocalVqeVariant.v13)
                Text("v1.2 (1.3M)").tag(LocalVqeVariant.v12)
            }
            .frame(width: 170)
            Picker("Chunk", selection: $model.chunk) {
                Text("256 ms").tag(LocalVqeChunk.batch256ms)
                Text("16 ms").tag(LocalVqeChunk.realtime16ms)
            }
            .frame(width: 150)
            Button(model.manager == nil ? "Load model" : "Reload") { model.loadModel() }
                .disabled(model.isLoading || model.isLive)
            if model.isLoading { ProgressView().controlSize(.small) }
            Text(model.loadStatus).foregroundStyle(.secondary).lineLimit(1)
            Spacer()
        }
    }
}

// MARK: - File mode

private struct FileModeView: View {
    @EnvironmentObject private var model: DemoModel
    @State private var picking: PickTarget?

    private enum PickTarget: Identifiable {
        case mic, reference
        var id: Self { self }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                FileField(title: "Mic recording", url: model.micURL) { picking = .mic }
                FileField(title: "Far end (optional)", url: model.referenceURL) { picking = .reference }
            }
            HStack {
                Menu {
                    ForEach(DemoModel.samples, id: \.id) { sample in
                        Button("fileid \(sample.id) — \(sample.note)") { model.useSamplePair(id: sample.id) }
                    }
                } label: {
                    Text("Use sample pair")
                } primaryAction: {
                    model.useSamplePair()
                }
                .fixedSize()
                .disabled(!model.hasSamples)
                Button("Enhance") { model.processFiles() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(model.isProcessing || model.manager == nil || model.micURL == nil)
                if model.isProcessing { ProgressView().controlSize(.small) }
                Spacer()
                SaveButton(clips: model.fileClips)
            }
            Text(model.fileStatus).font(.callout).foregroundStyle(.secondary)
            ClipList(clips: model.fileClips)
            if !model.hasSamples {
                Text(
                    "No sample pair found. Run `swift run fluidaudiocli enhance-benchmark --max-files 1` once to fetch the AEC-Challenge synthetic set, or pick your own files."
                )
                .font(.footnote).foregroundStyle(.tertiary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.top, 8)
        .fileImporter(
            isPresented: Binding(get: { picking != nil }, set: { if !$0 { picking = nil } }),
            allowedContentTypes: [.audio]
        ) { result in
            guard let target = picking, case .success(let url) = result else { return }
            switch target {
            case .mic: model.micURL = url
            case .reference: model.referenceURL = url
            }
            picking = nil
        }
    }
}

// MARK: - Live mode

private struct LiveModeView: View {
    @EnvironmentObject private var model: DemoModel
    @State private var pickingFarEnd = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                FileField(title: "Far-end file to play", url: model.farEndURL) { pickingFarEnd = true }
                Button("Use sample") { model.useSampleFarEnd() }.disabled(!model.hasSamples)
            }
            HStack {
                if model.isLive {
                    Button("Stop") { model.stopLive() }.keyboardShortcut(.cancelAction)
                } else {
                    Button("Start") { model.startLive() }
                        .keyboardShortcut(.defaultAction)
                        .disabled(model.manager == nil || model.farEndURL == nil)
                }
                Spacer()
                SaveButton(clips: model.liveClips)
            }
            Text(model.liveStatus).font(.callout).foregroundStyle(.secondary)

            GroupBox("Live levels (dBFS per 256 ms step)") {
                VStack(spacing: 8) {
                    LevelMeter(title: "Mic", db: model.liveLevels.micDb, tint: .orange)
                    LevelMeter(title: "Far end", db: model.liveLevels.referenceDb, tint: .blue)
                    LevelMeter(title: "Enhanced", db: model.liveLevels.enhancedDb, tint: .green)
                    HStack {
                        Text(String(format: "%.1f s captured", model.liveLevels.seconds))
                        Spacer()
                        Text(String(format: "%.1f ms per 256 ms step", model.liveLevels.stepMilliseconds))
                    }
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                .padding(6)
            }
            Text(
                "Use the built-in speaker and mic without headphones so the mic actually hears the playback. While only the far end is playing, the Enhanced meter should sit far below Mic; when you talk, it should follow your voice."
            )
            .font(.footnote).foregroundStyle(.tertiary)
            ClipList(clips: model.liveClips)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .padding(.top, 8)
        .fileImporter(isPresented: $pickingFarEnd, allowedContentTypes: [.audio]) { result in
            if case .success(let url) = result { model.farEndURL = url }
        }
    }
}

// MARK: - Pieces

private struct FileField: View {
    let title: String
    let url: URL?
    let pick: () -> Void

    var body: some View {
        HStack {
            Text(title).frame(width: 130, alignment: .trailing)
            Text(url?.lastPathComponent ?? "—")
                .lineLimit(1).truncationMode(.middle)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 6).padding(.vertical, 3)
                .background(RoundedRectangle(cornerRadius: 4).fill(.quaternary))
            Button("Choose…", action: pick)
        }
    }
}

private struct SaveButton: View {
    @EnvironmentObject private var model: DemoModel
    let clips: [Clip]

    var body: some View {
        Button("Show WAVs in Finder") { model.revealOutput() }
            .disabled(clips.isEmpty || model.lastOutputDirectory == nil)
    }
}

private struct ClipList: View {
    @EnvironmentObject private var model: DemoModel
    let clips: [Clip]

    var body: some View {
        VStack(spacing: 8) {
            if !clips.isEmpty {
                HStack {
                    Button {
                        if model.player.playingLabel != nil {
                            model.player.stop()
                        } else {
                            model.playBeforeAfter(clips)
                        }
                    } label: {
                        Label(
                            model.player.playingLabel == nil ? "Play before → after" : "Stop",
                            systemImage: model.player.playingLabel == nil ? "play.fill" : "stop.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .accessibilityLabel("Play before and after")
                    if let label = model.player.playingLabel {
                        Text("Playing: \(label)").font(.callout).foregroundStyle(.secondary)
                    }
                    Button {
                        model.transcribeAll(clips)
                    } label: {
                        Label("Transcribe all \(clips.count)", systemImage: "text.quote")
                    }
                    .disabled(model.isTranscribing)
                    .accessibilityLabel("Transcribe all")
                    if model.isTranscribing { ProgressView().controlSize(.small) }
                    Text(model.asrStatus).font(.callout).foregroundStyle(.secondary).lineLimit(1)
                    Spacer()
                }
            }
            ForEach(clips) { clip in
                HStack(spacing: 10) {
                    Button {
                        if model.player.playingLabel == clip.label { model.player.stop() } else { model.play(clip) }
                    } label: {
                        Image(systemName: model.player.playingLabel == clip.label ? "stop.fill" : "play.fill")
                            .frame(width: 14)
                    }
                    .buttonStyle(.bordered)
                    Text(clip.label).frame(width: 150, alignment: .leading)
                    WaveformView(samples: clip.samples, tint: tint(for: clip))
                        .frame(height: 44)
                    Text(String(format: "%.1f s", clip.seconds))
                        .font(.caption.monospacedDigit()).foregroundStyle(.secondary).frame(width: 44)
                }
                if let transcript = model.transcripts[clip.id] {
                    Text(transcript)
                        .font(.callout)
                        .foregroundStyle(tint(for: clip))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.leading, 200)
                }
            }
        }
    }

    private func tint(for clip: Clip) -> Color {
        if clip.label.hasPrefix("Enhanced") { return .green }
        if clip.label.hasPrefix("Far") { return .blue }
        return .orange
    }
}

private struct WaveformView: View {
    let samples: [Float]
    let tint: Color

    var body: some View {
        Canvas { context, size in
            let peaks = AudioFiles.peaks(samples, bins: Int(size.width))
            guard !peaks.isEmpty else { return }
            var path = Path()
            let mid = size.height / 2
            for (x, peak) in peaks.enumerated() {
                let h = CGFloat(min(1, peak)) * mid
                path.move(to: CGPoint(x: CGFloat(x), y: mid - h))
                path.addLine(to: CGPoint(x: CGFloat(x), y: mid + h))
            }
            context.stroke(path, with: .color(tint), lineWidth: 1)
        }
        .background(RoundedRectangle(cornerRadius: 4).fill(.quaternary.opacity(0.5)))
    }
}

private struct LevelMeter: View {
    let title: String
    let db: Float
    let tint: Color

    var body: some View {
        HStack {
            Text(title).frame(width: 70, alignment: .trailing)
            GeometryReader { geometry in
                let fraction = CGFloat(max(0, min(1, (db + 80) / 80)))
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3).fill(.quaternary)
                    RoundedRectangle(cornerRadius: 3).fill(tint).frame(width: geometry.size.width * fraction)
                }
            }
            .frame(height: 14)
            .animation(.linear(duration: 0.1), value: db)
            Text(String(format: "%6.1f dB", db)).font(.caption.monospacedDigit()).frame(width: 64)
        }
    }
}
