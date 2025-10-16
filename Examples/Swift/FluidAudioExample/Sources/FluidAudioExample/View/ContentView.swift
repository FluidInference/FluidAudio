import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var viewModel: ExampleViewModel
    @State private var showingImporter = false
    private let supportedTypes: [UTType] = [
        .audio,
        .wav,
        .aiff,
        .mpeg4Audio,
        UTType(filenameExtension: "mp3"),
    ].compactMap { $0 }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                stepCards
                streamingTranscript
                playbackControls
            }
            .padding(24)
        }
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: supportedTypes) { result in
            switch result {
            case .success(let url):
                viewModel.selectFile(url)
            case .failure(let error):
                NSLog("Audio selection failed: \(error.localizedDescription)")
            }
        }
    }

    private var stepCards: some View {
        VStack(alignment: .leading, spacing: 16) {
            StepCard(number: 1, title: "Choose Audio File", caption: "Select a local clip to stream.") {
                HStack(spacing: 16) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(viewModel.selectedFileName)
                            .font(.body.monospaced())
                            .foregroundStyle(viewModel.selectedFileURL == nil ? .secondary : .primary)
                        Text("Supported formats: WAV, MP3, AIFF, M4A. Larger files will take longer to stream.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(action: { showingImporter = true }) {
                        Label("Choose Audio…", systemImage: "folder")
                    }
                }
            }

            StepCard(
                number: 2,
                title: "Stream & Transcribe",
                caption: "Run stabilized streaming ASR over the selected audio."
            ) {
                VStack(alignment: .leading, spacing: 8) {
                    Button(action: viewModel.startTranscription) {
                        Label("Start Streaming", systemImage: "waveform")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!viewModel.canStartTranscription)

                    Text(viewModel.statusText)
                        .font(.footnote)
                        .foregroundStyle(
                            viewModel.stage.errorMessage == nil ? Color.secondary : Color.red
                        )
                }
            }

            if let metadata = viewModel.metadata {
                MetricsCard(metadata: metadata)
            }
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("FluidAudio Streaming Example")
                .font(.largeTitle.weight(.semibold))

            Text(
                "Upload audio, watch streaming transcripts stabilize in real time, then hand the result to Kokoro for playback."
            )
            .foregroundStyle(.secondary)

            HStack(spacing: 12) {
                if viewModel.stage.isBusy {
                    ProgressView()
                        .progressViewStyle(.circular)
                }

                Text(viewModel.statusText)
                    .foregroundStyle(
                        viewModel.stage.errorMessage == nil ? Color.secondary : Color.red
                    )
            }
        }
    }

    private var streamingTranscript: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Live Transcript")
                .font(.headline)

            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if viewModel.displayTranscript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Text("No transcription yet. Start the stream to see real-time updates.")
                            .foregroundStyle(.secondary)
                    } else {
                        Text(viewModel.displayTranscript)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .foregroundStyle(viewModel.stage == .streaming ? .primary : .primary)
                    }
                }
                .padding()
            }
            .frame(minHeight: 180)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(Color.secondary.opacity(0.2), lineWidth: 1)
            )
        }
    }

    private var playbackControls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Kokoro Playback")
                .font(.headline)

            HStack(spacing: 16) {
                Button(action: viewModel.readTranscriptAloud) {
                    Label("Read Back with Kokoro", systemImage: "play.circle")
                }
                .disabled(!viewModel.canPlayTranscript)

                if viewModel.isPlaybackActive {
                    Button(action: viewModel.stopPlayback) {
                        Label("Stop Playback", systemImage: "stop.circle")
                    }
                }
            }

            Text("First run downloads Kokoro models and lexicons. Playback uses the default FluidAudio voice.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
    }
}

private struct StepCard<Content: View>: View {
    let number: Int
    let title: String
    let caption: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text("\(number)")
                    .font(.headline.weight(.semibold))
                    .padding(8)
                    .background(
                        Circle()
                            .fill(Color.accentColor.opacity(0.12))
                    )
                VStack(alignment: .leading, spacing: 4) {
                    Text(title)
                        .font(.headline)
                    Text(caption)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }

            content
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.secondary.opacity(0.06))
        )
    }
}

private struct MetricsCard: View {
    let metadata: ExampleTranscriptionMetadata

    var body: some View {
        StepCard(
            number: 3, title: "Session Metrics", caption: "Quick telemetry captured from the stabilized streaming run."
        ) {
            VStack(alignment: .leading, spacing: 12) {
                MetricsGrid {
                    MetricItem(
                        title: "Processing Time",
                        value: formatSeconds(metadata.wallClockSeconds)
                    )
                    MetricItem(
                        title: "First Confirmed Token",
                        value: formatOptionalSeconds(metadata.firstConfirmedTokenLatency)
                    )
                    MetricItem(
                        title: "Real-Time Factor",
                        value: formatRTF(metadata.realTimeFactor)
                    )
                    MetricItem(
                        title: "Word Count",
                        value: "\(metadata.wordCount)"
                    )
                    MetricItem(
                        title: "First Token",
                        value: formatOptionalSeconds(metadata.firstTokenLatency)
                    )
                    MetricItem(
                        title: "Chunks Processed",
                        value: "\(metadata.chunkCount)"
                    )
                }
            }
        }
    }

    private func formatSeconds(_ value: TimeInterval) -> String {
        String(format: "%.2f s", value)
    }

    private func formatOptionalSeconds(_ value: TimeInterval?) -> String {
        guard let value else { return "n/a" }
        return formatSeconds(value)
    }

    private func formatRTF(_ value: Double?) -> String {
        guard let value else { return "n/a" }
        return String(format: "%.2fx", value)
    }
}

private struct MetricsGrid<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        let columns = [
            GridItem(.flexible(), spacing: 16),
            GridItem(.flexible(), spacing: 16),
        ]
        LazyVGrid(columns: columns, alignment: .leading, spacing: 16) {
            content
        }
    }
}

private struct MetricItem: View {
    let title: String
    let value: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title.uppercased())
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.headline)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.secondary.opacity(0.08))
        )
    }
}
