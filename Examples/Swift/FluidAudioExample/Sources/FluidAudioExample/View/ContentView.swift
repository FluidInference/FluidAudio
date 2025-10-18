import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var viewModel: ExampleViewModel
    @State private var showingImporter = false
    @State private var inputSource: InputSource = .file
    private let supportedTypes: [UTType] = [
        .audio,
        .wav,
        .aiff,
        .mpeg4Audio,
        UTType(filenameExtension: "mp3"),
    ].compactMap { $0 }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: DesignSpacing.xxl) {
                header
                stepCards
                streamingTranscript
                playbackControls
            }
            .padding(DesignSpacing.xl)
        }
        .background(DesignColors.background)
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
        VStack(alignment: .leading, spacing: DesignSpacing.lg) {
            ModernStepCard(number: 1, title: "Choose Input Source", caption: "Select where to stream from.") {
                VStack(alignment: .leading, spacing: DesignSpacing.md) {
                    Menu {
                        Button(action: {
                            inputSource = .microphone
                        }) {
                            Label("Microphone", systemImage: "mic.fill")
                        }

                        Button(action: {
                            inputSource = .file
                            showingImporter = true
                        }) {
                            Label("Audio File", systemImage: "folder")
                        }
                    } label: {
                        HStack(spacing: DesignSpacing.md) {
                            Image(systemName: inputSource == .microphone ? "mic.fill" : "folder")
                                .foregroundColor(DesignColors.accent)

                            Text(inputSource == .microphone ? "Microphone" : "Audio File")
                                .bodyText()

                            Spacer()

                            Image(systemName: "chevron.down")
                                .font(.system(size: 12, weight: .semibold))
                                .foregroundColor(DesignColors.textSecondary)
                        }
                        .padding(DesignSpacing.md)
                        .frame(maxWidth: .infinity)
                        .background(DesignColors.secondaryBackground)
                        .cornerRadius(DesignRadius.medium)
                    }

                    if inputSource == .file {
                        VStack(alignment: .leading, spacing: DesignSpacing.sm) {
                            Text(viewModel.selectedFileName)
                                .font(DesignTypography.monospaceBody)
                                .foregroundColor(
                                    viewModel.selectedFileURL == nil ? DesignColors.textSecondary : DesignColors.text)

                            Text("Supported: WAV, MP3, AIFF, M4A")
                                .secondaryText()
                        }
                    }
                }
            }

            ModernStepCard(
                number: 2,
                title: "Stream & Transcribe",
                caption: inputSource == .microphone
                    ? "Start listening with microphone." : "Start streaming transcription."
            ) {
                VStack(alignment: .leading, spacing: DesignSpacing.md) {
                    HStack(spacing: DesignSpacing.md) {
                        ModernButton(
                            inputSource == .microphone ? "Start Listening" : "Start Streaming",
                            icon: inputSource == .microphone ? "mic.fill" : "waveform",
                            isLoading: viewModel.stage == .streaming,
                            isDisabled: inputSource == .file
                                ? !viewModel.canStartTranscription : !viewModel.canStartMicrophoneStream
                        ) {
                            if inputSource == .microphone {
                                viewModel.startMicrophoneTranscription()
                            } else {
                                viewModel.startTranscription()
                            }
                        }

                        if viewModel.isMicrophoneStreaming {
                            ModernButton(
                                "Stop",
                                icon: "stop.circle",
                                style: .secondary
                            ) {
                                viewModel.stopMicrophoneTranscription()
                            }
                        }
                    }

                    if inputSource == .microphone {
                        let status = microphoneStatusMessage
                        StatusBadge(status: status)

                        if viewModel.isMicrophoneStreaming {
                            WaveformIndicator(isRecording: true)
                                .frame(height: 24)
                        }
                    } else if let status = fileStatusMessage {
                        StatusBadge(status: status)
                    }
                }
            }

            if let metadata = viewModel.metadata {
                ModernMetricsCard(metadata: metadata)
                    .transition(.opacity.combined(with: .scale(scale: 0.95)))
            } else if viewModel.stage.isBusy {
                SkeletonCardLoader()
            }
        }
    }

    private var fileStatusMessage: StatusBadge.StatusType? {
        if viewModel.isMicrophoneStreaming {
            return .idle
        }

        if let error = viewModel.stage.errorMessage,
            !error.lowercased().contains("microphone")
        {
            return .error(error)
        }

        if viewModel.selectedFileURL != nil,
            viewModel.stage.isBusy
        {
            return .loading
        }

        return nil
    }

    private var microphoneStatusMessage: StatusBadge.StatusType {
        if let error = viewModel.stage.errorMessage,
            error.lowercased().contains("microphone")
        {
            return .error(error)
        }

        if viewModel.stage == .requestingMicrophone {
            return .loading
        }

        if viewModel.isMicrophoneStreaming {
            return .recording
        }

        return .idle
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            Text("FluidAudio Streaming")
                .font(DesignTypography.displayMedium)
                .foregroundColor(DesignColors.text)

            Text(
                "Upload audio or speak live, watch streaming transcripts stabilize in real time, then hand the result to Kokoro for playback."
            )
            .secondaryText()
            .lineLimit(3)

            if viewModel.stage.isBusy {
                HStack(spacing: DesignSpacing.md) {
                    ProgressView()
                        .scaleEffect(0.85)

                    Text(viewModel.statusText)
                        .secondaryText()

                    Spacer()
                }
            }
        }
    }

    private var streamingTranscript: some View {
        VStack(alignment: .leading, spacing: DesignSpacing.md) {
            Text("Live Transcript")
                .font(DesignTypography.headingMedium)
                .foregroundColor(DesignColors.text)

            ScrollView {
                VStack(alignment: .leading, spacing: DesignSpacing.lg) {
                    if viewModel.displayTranscript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        Text("No transcription yet. Start the stream to see real-time updates.")
                            .secondaryText()
                            .frame(maxWidth: .infinity, alignment: .center)
                            .padding(.vertical, DesignSpacing.xl)
                    } else {
                        Text(viewModel.displayTranscript)
                            .bodyText()
                            .lineSpacing(4)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .transition(.opacity)
                    }
                }
                .padding(DesignSpacing.lg)
            }
            .frame(minHeight: 180)
            .background(DesignColors.card)
            .cornerRadius(DesignRadius.medium)
            .shadow(
                color: DesignElevation.small.color,
                radius: DesignElevation.small.radius,
                x: DesignElevation.small.x,
                y: DesignElevation.small.y
            )
        }
    }

    private var playbackControls: some View {
        ModernStepCard(
            number: 3,
            title: "Kokoro Playback",
            caption: "Convert transcript to speech with text-to-speech synthesis."
        ) {
            VStack(alignment: .leading, spacing: DesignSpacing.md) {
                HStack(spacing: DesignSpacing.md) {
                    ModernButton(
                        "Read Aloud",
                        icon: "play.circle",
                        isLoading: viewModel.stage == .synthesizing || viewModel.stage == .playing,
                        isDisabled: !viewModel.canPlayTranscript
                    ) {
                        viewModel.readTranscriptAloud()
                    }

                    if viewModel.isPlaybackActive {
                        ModernButton(
                            "Stop",
                            icon: "stop.circle",
                            style: .secondary
                        ) {
                            viewModel.stopPlayback()
                        }
                    }
                }

                Text("First run downloads Kokoro models. Uses the default FluidAudio voice.")
                    .secondaryText()
            }
        }
    }
}

extension ContentView {
    private enum InputSource: Hashable {
        case file
        case microphone
    }
}

private struct ModernMetricsCard: View {
    let metadata: ExampleTranscriptionMetadata

    var body: some View {
        ModernStepCard(
            number: 3, title: "Session Metrics", caption: "Quick telemetry from the streaming run."
        ) {
            VStack(alignment: .leading, spacing: DesignSpacing.lg) {
                ModernMetricsGrid {
                    ModernMetricCard(
                        title: "Processing Time",
                        value: formatSeconds(metadata.wallClockSeconds),
                        icon: "timer"
                    )
                    ModernMetricCard(
                        title: "Inference Speed",
                        value: formatSpeedup(metadata),
                        icon: "bolt.fill"
                    )
                    ModernMetricCard(
                        title: "First Token",
                        value: formatOptionalSeconds(metadata.firstTokenLatency),
                        icon: "hare"
                    )
                    ModernMetricCard(
                        title: "Word Count",
                        value: "\(metadata.wordCount)",
                        icon: "textformat"
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

    private func formatSpeedup(_ metadata: ExampleTranscriptionMetadata) -> String {
        guard metadata.wallClockSeconds > 0 else { return "n/a" }
        let speedup = metadata.audioSeconds / metadata.wallClockSeconds
        return String(format: "%.1fx faster", speedup)
    }
}
