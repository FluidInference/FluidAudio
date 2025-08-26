import AVFoundation
import FluidAudio
import UIKit

/// Example implementation of streaming ASR in a UIKit application
/// This shows the recommended pattern for integrating FluidAudio with iOS apps
@available(iOS 16.0, *)
class TranscriptionViewController: UIViewController {

    // MARK: - UI Outlets
    @IBOutlet weak var transcriptionLabel: UILabel!
    @IBOutlet weak var startButton: UIButton!
    @IBOutlet weak var stopButton: UIButton!
    @IBOutlet weak var statusLabel: UILabel!

    // MARK: - FluidAudio Components
    private let streamingAsr = StreamingAsrManager(config: .realtime)
    private var audioEngine: AVAudioEngine?
    private var inputNode: AVAudioInputNode?

    // MARK: - State Management
    private var isRecording = false
    private var transcriptionTask: Task<Void, Never>?

    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupAudioEngine()
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        stopTranscription()
    }

    // MARK: - UI Setup
    private func setupUI() {
        transcriptionLabel.text = ""
        transcriptionLabel.numberOfLines = 0
        stopButton.isEnabled = false

        // Style for volatile text (you might use NSAttributedString for better styling)
        transcriptionLabel.textColor = .label
    }

    // MARK: - Audio Engine Setup
    private func setupAudioEngine() {
        audioEngine = AVAudioEngine()
        inputNode = audioEngine?.inputNode

        // Request microphone permission
        AVAudioSession.sharedInstance().requestRecordPermission { [weak self] granted in
            DispatchQueue.main.async {
                self?.statusLabel.text = granted ? "Ready" : "Microphone permission required"
                self?.startButton.isEnabled = granted
            }
        }
    }

    // MARK: - Transcription Control
    @IBAction func startTranscription(_ sender: UIButton) {
        Task {
            await startStreamingTranscription()
        }
    }

    @IBAction func stopTranscription(_ sender: UIButton) {
        stopTranscription()
    }

    private func startStreamingTranscription() async {
        guard !isRecording else { return }

        do {
            // Update UI
            await MainActor.run {
                isRecording = true
                startButton.isEnabled = false
                stopButton.isEnabled = true
                statusLabel.text = "Starting..."
                transcriptionLabel.text = ""
            }

            // Start FluidAudio streaming ASR
            try await streamingAsr.start()

            // Start audio capture
            try await startAudioCapture()

            // Listen for transcription updates
            transcriptionTask = Task { [weak self] in
                guard let self = self else { return }

                for await snapshot in self.streamingAsr.snapshots {
                    await MainActor.run {
                        self.updateTranscriptionUI(snapshot: snapshot)
                    }
                }
            }

            await MainActor.run {
                statusLabel.text = "Recording..."
            }

        } catch {
            await MainActor.run {
                statusLabel.text = "Error: \(error.localizedDescription)"
                stopTranscription()
            }
        }
    }

    private func stopTranscription() {
        guard isRecording else { return }

        // Cancel transcription task
        transcriptionTask?.cancel()
        transcriptionTask = nil

        // Stop audio engine
        audioEngine?.stop()

        // Get final transcription
        Task {
            let finalText = (try? await streamingAsr.finish()) ?? ""
            await MainActor.run {
                transcriptionLabel.text = finalText
                statusLabel.text = "Stopped"
                isRecording = false
                startButton.isEnabled = true
                stopButton.isEnabled = false
            }
        }
    }

    // MARK: - Audio Capture
    private func startAudioCapture() async throws {
        guard let audioEngine = audioEngine,
            let inputNode = inputNode
        else {
            throw TranscriptionError.audioEngineSetupFailed
        }

        // Configure audio format (FluidAudio expects 16kHz mono)
        let format = AVAudioFormat(standardFormatWithSampleRate: 16000, channels: 1)!

        // Install audio tap
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            Task {
                await self?.streamingAsr.streamAudio(buffer)
            }
        }

        try audioEngine.start()
    }

    // MARK: - UI Updates
    private func updateTranscriptionUI(snapshot: StreamingTranscriptSnapshot) {
        let finalized = String(snapshot.finalized.characters)
        let volatile = snapshot.volatile.map { String($0.characters) } ?? ""

        // Create attributed string to show volatile text differently
        let attributedText = NSMutableAttributedString()

        // Add finalized text in normal color
        if !finalized.isEmpty {
            attributedText.append(
                NSAttributedString(
                    string: finalized,
                    attributes: [.foregroundColor: UIColor.label]
                ))
        }

        // Add volatile text in dimmed color
        if !volatile.isEmpty {
            if !finalized.isEmpty {
                attributedText.append(NSAttributedString(string: " "))
            }
            attributedText.append(
                NSAttributedString(
                    string: volatile,
                    attributes: [.foregroundColor: UIColor.secondaryLabel]
                ))
        }

        transcriptionLabel.attributedText = attributedText

        // Auto-scroll to bottom if needed
        // (Implement scrolling logic if using UITextView instead of UILabel)
    }
}

// MARK: - Error Types
enum TranscriptionError: LocalizedError {
    case audioEngineSetupFailed

    var errorDescription: String? {
        switch self {
        case .audioEngineSetupFailed:
            return "Failed to setup audio engine"
        }
    }
}
