import AVFoundation
import FluidAudio
import SwiftUI

/// Example implementation of streaming ASR in a SwiftUI application
/// This demonstrates reactive patterns with FluidAudio streaming transcription
@available(iOS 16.0, macOS 13.0, *)
struct StreamingTranscriptionView: View {
    @StateObject private var transcriptionModel = TranscriptionViewModel()

    var body: some View {
        VStack(spacing: 20) {
            // Header
            Text("FluidAudio Streaming Transcription")
                .font(.title2)
                .bold()

            // Status
            Text(transcriptionModel.status)
                .foregroundColor(statusColor)
                .font(.caption)

            // Transcription Display
            ScrollView {
                Text(transcriptionModel.displayText)
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.secondary.opacity(0.1))
                    .cornerRadius(8)
                    .font(.body)
            }
            .frame(minHeight: 200)

            // Controls
            HStack {
                Button(transcriptionModel.isRecording ? "Stop" : "Start") {
                    Task {
                        if transcriptionModel.isRecording {
                            await transcriptionModel.stopTranscription()
                        } else {
                            await transcriptionModel.startTranscription()
                        }
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(!transcriptionModel.canToggleRecording)

                Button("Clear") {
                    transcriptionModel.clearTranscription()
                }
                .buttonStyle(.bordered)
                .disabled(transcriptionModel.isRecording)
            }
        }
        .padding()
        .onAppear {
            Task {
                await transcriptionModel.requestMicrophonePermission()
            }
        }
        .onDisappear {
            Task {
                await transcriptionModel.stopTranscription()
            }
        }
    }

    private var statusColor: Color {
        switch transcriptionModel.status {
        case "Recording...":
            return .green
        case "Ready":
            return .blue
        default:
            return .secondary
        }
    }
}

// MARK: - View Model
@available(iOS 16.0, macOS 13.0, *)
@MainActor
class TranscriptionViewModel: ObservableObject {

    // MARK: - Published Properties
    @Published var displayText: String = ""
    @Published var status: String = "Initializing..."
    @Published var isRecording: Bool = false
    @Published var canToggleRecording: Bool = false

    // MARK: - Private Properties
    private let streamingAsr = StreamingAsrManager(config: .realtime)
    private var audioEngine: AVAudioEngine?
    private var transcriptionTask: Task<Void, Never>?

    init() {
        setupAudioEngine()
    }

    // MARK: - Audio Setup
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
    }

    func requestMicrophonePermission() async {
        let granted = await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }

        status = granted ? "Ready" : "Microphone permission required"
        canToggleRecording = granted
    }

    // MARK: - Transcription Control
    func startTranscription() async {
        guard !isRecording, canToggleRecording else { return }

        do {
            isRecording = true
            canToggleRecording = false
            status = "Starting..."
            displayText = ""

            // Start FluidAudio
            try await streamingAsr.start()

            // Start audio capture
            try await startAudioCapture()

            // Listen for updates
            transcriptionTask = Task {
                for await snapshot in streamingAsr.snapshots {
                    await updateTranscription(snapshot: snapshot)
                }
            }

            status = "Recording..."
            canToggleRecording = true

        } catch {
            status = "Error: \(error.localizedDescription)"
            isRecording = false
            canToggleRecording = true
        }
    }

    func stopTranscription() async {
        guard isRecording else { return }

        canToggleRecording = false
        status = "Stopping..."

        // Cancel transcription updates
        transcriptionTask?.cancel()
        transcriptionTask = nil

        // Stop audio engine
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)

        // Get final result
        let finalText = (try? await streamingAsr.finish()) ?? displayText
        displayText = finalText

        isRecording = false
        status = "Ready"
        canToggleRecording = true
    }

    func clearTranscription() {
        displayText = ""
    }

    // MARK: - Audio Capture
    private func startAudioCapture() async throws {
        guard let audioEngine = audioEngine else {
            throw TranscriptionError.audioEngineSetupFailed
        }

        let inputNode = audioEngine.inputNode
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            Task {
                await self?.streamingAsr.streamAudio(buffer)
            }
        }

        try audioEngine.start()
    }

    // MARK: - UI Updates
    private func updateTranscription(snapshot: StreamingTranscriptSnapshot) async {
        await MainActor.run {
            let finalized = String(snapshot.finalized.characters)
            let volatile = snapshot.volatile.map { String($0.characters) } ?? ""

            // Combine finalized and volatile text with visual distinction
            if volatile.isEmpty {
                displayText = finalized
            } else {
                displayText = finalized + (finalized.isEmpty ? "" : " ") + "[\(volatile)]"
            }
        }
    }
}

// MARK: - Preview
@available(iOS 16.0, macOS 13.0, *)
struct StreamingTranscriptionView_Previews: PreviewProvider {
    static var previews: some View {
        StreamingTranscriptionView()
    }
}

// MARK: - App Entry Point Example
@available(iOS 16.0, macOS 13.0, *)
@main
struct FluidAudioExampleApp: App {
    var body: some Scene {
        WindowGroup {
            StreamingTranscriptionView()
        }
    }
}
