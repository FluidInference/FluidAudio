import AVFoundation
import Combine
import FluidAudio
import Foundation

@MainActor
final class ExampleViewModel: ObservableObject {

    enum Stage: Equatable {
        case idle
        case preparingModels
        case streaming
        case finishing
        case ready
        case synthesizing
        case playing
        case error(String)

        var label: String {
            switch self {
            case .idle:
                return "Select an audio file to begin"
            case .preparingModels:
                return "Downloading & warming models…"
            case .streaming:
                return "Streaming transcription…"
            case .finishing:
                return "Finalizing transcript…"
            case .ready:
                return "Transcript ready"
            case .synthesizing:
                return "Synthesizing Kokoro audio…"
            case .playing:
                return "Playing synthesized audio…"
            case .error:
                return "Error"
            }
        }

        var isBusy: Bool {
            switch self {
            case .idle, .ready, .error:
                return false
            case .preparingModels, .streaming, .finishing, .synthesizing, .playing:
                return true
            }
        }

        var errorMessage: String? {
            if case .error(let message) = self {
                return message
            }
            return nil
        }
    }

    @Published private(set) var stage: Stage = .idle
    @Published private(set) var selectedFileURL: URL?
    @Published private(set) var selectedFileName: String = "No file selected"
    @Published private(set) var confirmedText: String = ""
    @Published private(set) var volatileText: String = ""
    @Published private(set) var transcript: String = ""
    @Published private(set) var isPlaybackActive = false
    @Published private(set) var metadata: ExampleTranscriptionMetadata?

    private let transcriptionCoordinator = StreamingTranscriptionCoordinator()
    private let playbackController = PlaybackController()
    private var asrModels: AsrModels?
    private var ttsManager: TtSManager?
    private var transcriptionTask: Task<Void, Never>?
    private var playbackTask: Task<Void, Never>?

    var statusText: String {
        if let error = stage.errorMessage {
            return error
        }
        return stage.label
    }

    var canStartTranscription: Bool {
        selectedFileURL != nil && !stage.isBusy
    }

    var canPlayTranscript: Bool {
        !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !stage.isBusy
    }

    var displayTranscript: String {
        if stage == .streaming {
            return confirmedText + volatileText
        }
        return transcript.isEmpty ? confirmedText : transcript
    }

    func selectFile(_ url: URL) {
        cancelActivePipeline()
        selectedFileURL = url
        selectedFileName = url.lastPathComponent
        stage = .idle
        confirmedText = ""
        volatileText = ""
        transcript = ""
        metadata = nil
    }

    func startTranscription() {
        guard canStartTranscription, let audioURL = selectedFileURL else { return }

        transcriptionTask?.cancel()
        transcriptionTask = Task { [weak self] in
            await self?.runTranscription(for: audioURL)
        }
    }

    func readTranscriptAloud() {
        guard canPlayTranscript else { return }

        playbackTask?.cancel()
        playbackTask = Task { [weak self] in
            await self?.performPlayback()
        }
    }

    func stopPlayback() {
        playbackTask?.cancel()
        playbackController.stop()
        isPlaybackActive = false
        if stage == .playing {
            stage = .ready
        }
    }

    private func runTranscription(for url: URL) async {
        stage = .preparingModels
        confirmedText = ""
        volatileText = ""
        transcript = ""
        metadata = nil

        do {
            let models = try await ensureAsrModels()

            try Task.checkCancellation()

            stage = .streaming

            let result = try await transcriptionCoordinator.transcribeFile(
                at: url,
                models: models,
                onUpdate: { [weak self] update in
                    guard let self else { return }
                    self.handle(update: update)
                }
            )

            try Task.checkCancellation()

            stage = .finishing
            confirmedText = result.transcript
            volatileText = ""
            transcript = result.transcript
            metadata = ExampleTranscriptionMetadata(
                wallClockSeconds: result.wallClockSeconds,
                audioSeconds: result.audioSeconds,
                firstTokenLatency: result.firstTokenLatency,
                firstConfirmedTokenLatency: result.firstConfirmedTokenLatency,
                wordCount: Self.wordCount(in: result.transcript)
            )
            stage = .ready
        } catch is CancellationError {
            stage = .idle
        } catch {
            stage = .error(readableMessage(for: error))
        }

        transcriptionTask = nil
    }

    private func performPlayback() async {
        guard !transcript.isEmpty else {
            playbackTask = nil
            return
        }

        stage = .synthesizing

        do {
            let manager = try await ensureTtsManager()
            try Task.checkCancellation()

            let audioData = try await manager.synthesize(text: transcript)
            try Task.checkCancellation()

            stage = .playing
            isPlaybackActive = true

            try playbackController.play(data: audioData) { [weak self] in
                Task { @MainActor in
                    guard let self else { return }
                    self.isPlaybackActive = false
                    if self.stage == .playing {
                        self.stage = .ready
                    }
                }
            }
        } catch is CancellationError {
            if !transcript.isEmpty {
                stage = .ready
            } else {
                stage = .idle
            }
        } catch {
            isPlaybackActive = false
            stage = .error(readableMessage(for: error))
        }

        playbackTask = nil
    }

    private func ensureAsrModels() async throws -> AsrModels {
        if let cachedModels = asrModels {
            return cachedModels
        }

        let models = try await AsrModels.downloadAndLoad()
        asrModels = models
        return models
    }

    private func ensureTtsManager() async throws -> TtSManager {
        if let manager = ttsManager, manager.isAvailable {
            return manager
        }

        let manager = TtSManager()
        try await manager.initialize()
        ttsManager = manager
        return manager
    }

    private func handle(update: StreamingTranscriptionUpdate) {
        if update.isConfirmed {
            if !update.text.isEmpty {
                confirmedText.append(update.text)
            }
            volatileText = ""
        } else {
            volatileText = update.text
        }
    }

    private func cancelActivePipeline() {
        transcriptionTask?.cancel()
        transcriptionTask = nil
        playbackTask?.cancel()
        playbackTask = nil
        playbackController.stop()
        isPlaybackActive = false
        metadata = nil

        Task {
            await transcriptionCoordinator.cancelActiveSession()
        }
    }

    private func readableMessage(for error: Error) -> String {
        if let localized = error as? LocalizedError {
            if let description = localized.errorDescription {
                return description
            }
        }
        return error.localizedDescription
    }

    private static func wordCount(in text: String) -> Int {
        let words = text.split { $0.isWhitespace || $0.isNewline }
        return words.count
    }
}

struct ExampleTranscriptionMetadata {
    let wallClockSeconds: TimeInterval
    let audioSeconds: TimeInterval
    let firstTokenLatency: TimeInterval?
    let firstConfirmedTokenLatency: TimeInterval?
    let wordCount: Int

    var realTimeFactor: Double? {
        guard audioSeconds > 0 else { return nil }
        return wallClockSeconds / audioSeconds
    }
}
