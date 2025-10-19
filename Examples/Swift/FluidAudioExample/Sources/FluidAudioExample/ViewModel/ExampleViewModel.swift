import AVFoundation
import Combine
import FluidAudio
import Foundation
import OSLog

@MainActor
final class ExampleViewModel: ObservableObject {

    enum Stage: Equatable {
        case idle
        case preparingModels
        case requestingMicrophone
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
            case .requestingMicrophone:
                return "Requesting microphone access…"
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
            case .preparingModels, .requestingMicrophone, .streaming, .finishing, .synthesizing, .playing:
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

    @Published private(set) var stage: Stage = .idle {
        didSet {
            guard oldValue != self.stage else { return }
            let previousLabel = oldValue.label
            let currentLabel = self.stage.label
            logger.debug("Stage transition \(previousLabel) -> \(currentLabel)")
            if self.stage == .streaming {
                if self.streamingStartTime == nil {
                    self.streamingStartTime = Date()
                    logger.info("Streaming session started")
                }
            } else if self.stage == .idle || self.stage == .ready {
                self.streamingStartTime = nil
            }
        }
    }
    @Published private(set) var selectedFileURL: URL?
    @Published private(set) var selectedFileName: String = "No file selected"
    @Published private(set) var confirmedText: String = ""
    @Published private(set) var volatileText: String = ""
    @Published private(set) var transcript: String = ""
    @Published private(set) var isPlaybackActive = false
    @Published private(set) var isMicrophoneStreaming = false
    @Published private(set) var metadata: ExampleTranscriptionMetadata?
    @Published private(set) var liveSnapshot: StreamingTranscriptSnapshot = .empty
    private var securityScopedFileURL: URL?

    private let transcriptionCoordinator = StreamingTranscriptionCoordinator()
    private let playbackController = PlaybackController()
    private lazy var microphoneController = MicrophoneStreamingController(coordinator: transcriptionCoordinator)
    private var asrModels: AsrModels?
    private var ttsManager: TtSManager?
    private var transcriptionTask: Task<Void, Never>?
    private var playbackTask: Task<Void, Never>?
    private var microphoneTask: Task<Void, Never>?
    private let logger = Logger(subsystem: "com.fluidaudio.example", category: "Streaming")
    private var confirmedUpdateCount: Int = 0
    private var volatileUpdateCount: Int = 0
    private var streamingStartTime: Date?
    private let uiUpdateInterval: TimeInterval = 0.15
    private var pendingSnapshot: StreamingTranscriptSnapshot?
    private var snapshotUpdateTask: Task<Void, Never>?
    private var lastSnapshotTimestamp: Date = .distantPast

    var statusText: String {
        if let error = stage.errorMessage {
            return error
        }
        return stage.label
    }

    var canStartTranscription: Bool {
        selectedFileURL != nil && !stage.isBusy
    }

    var canStartMicrophoneStream: Bool {
        !stage.isBusy && !isMicrophoneStreaming
    }

    var canStopMicrophoneStream: Bool {
        isMicrophoneStreaming && stage == .streaming
    }

    var canPlayTranscript: Bool {
        !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !stage.isBusy
    }

    var displayTranscript: String {
        if stage == .streaming {
            return liveSnapshot.confirmedText + liveSnapshot.volatileText
        }
        return transcript.isEmpty ? confirmedText : transcript
    }

    func selectFile(_ url: URL) {
        cancelActivePipeline()
        stopAccessingSelectedFile()

        if url.startAccessingSecurityScopedResource() {
            securityScopedFileURL = url
        } else {
            securityScopedFileURL = nil
        }

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

    func startMicrophoneTranscription() {
        guard canStartMicrophoneStream else { return }

        cancelActivePipeline(resetStage: false)

        microphoneTask?.cancel()
        microphoneTask = Task { [weak self] in
            await self?.runMicrophoneStreaming()
        }
    }

    func stopMicrophoneTranscription() {
        guard canStopMicrophoneStream else { return }

        microphoneTask?.cancel()
        microphoneTask = Task { [weak self] in
            await self?.finishMicrophoneStreaming()
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
        resetStreamingStateCounters()
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
            logStreamingSummary(reason: "file stream completed")
            stage = .ready
        } catch is CancellationError {
            logStreamingSummary(reason: "file stream cancelled")
            stage = .idle
        } catch {
            logStreamingSummary(reason: "file stream failed: \(readableMessage(for: error))")
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

    private func runMicrophoneStreaming() async {
        resetStreamingStateCounters()
        defer { microphoneTask = nil }

        stage = .requestingMicrophone
        confirmedText = ""
        volatileText = ""
        transcript = ""
        metadata = nil

        do {
            let permissionGranted = await microphoneController.requestPermissionIfNeeded()
            try Task.checkCancellation()

            guard permissionGranted else {
                stage = .error(
                    "Microphone access was denied. Enable access in Settings to stream live audio."
                )
                return
            }

            stage = .preparingModels

            let models = try await ensureAsrModels()
            try Task.checkCancellation()

            try await microphoneController.startStreaming(
                models: models,
                onUpdate: { [weak self] update in
                    guard let self else { return }
                    self.handle(update: update)
                },
                skipPermissionCheck: true
            )

            try Task.checkCancellation()

            isMicrophoneStreaming = true
            stage = .streaming
        } catch is CancellationError {
            isMicrophoneStreaming = false
            if stage != .ready {
                stage = .idle
            }
        } catch {
            isMicrophoneStreaming = false
            logStreamingSummary(reason: "microphone stream failed: \(readableMessage(for: error))")
            stage = .error(readableMessage(for: error))
        }
    }

    private func finishMicrophoneStreaming() async {
        defer { microphoneTask = nil }

        stage = .finishing

        do {
            let result = try await microphoneController.stopStreaming()
            try Task.checkCancellation()

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

            isMicrophoneStreaming = false
            logStreamingSummary(reason: "microphone stream completed")
            stage = .ready
        } catch is CancellationError {
            isMicrophoneStreaming = false
            logStreamingSummary(reason: "microphone stream cancelled")
            if transcript.isEmpty {
                stage = .idle
            } else {
                stage = .ready
            }
        } catch {
            isMicrophoneStreaming = false
            stage = .error(readableMessage(for: error))
        }
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
        let sanitizedPreview = update.text.replacingOccurrences(of: "\n", with: " ")
        let preview = String(sanitizedPreview.prefix(80))
        let elapsed = streamingStartTime.map { update.timestamp.timeIntervalSince($0) }

        if update.isConfirmed {
            self.confirmedUpdateCount += 1
            if let elapsed {
                logger.debug(
                    "Confirmed update #\(self.confirmedUpdateCount, privacy: .public) len=\(update.text.count, privacy: .public) elapsed=\(elapsed, format: .fixed(precision: 3))s preview=\"\(preview, privacy: .public)\""
                )
            } else {
                logger.debug(
                    "Confirmed update #\(self.confirmedUpdateCount, privacy: .public) len=\(update.text.count, privacy: .public) preview=\"\(preview, privacy: .public)\""
                )
            }
            if !update.text.isEmpty {
                self.confirmedText.append(update.text)
            }
            self.volatileText = ""
        } else {
            self.volatileUpdateCount += 1
            if let elapsed {
                logger.debug(
                    "Volatile update #\(self.volatileUpdateCount, privacy: .public) len=\(update.text.count, privacy: .public) elapsed=\(elapsed, format: .fixed(precision: 3))s preview=\"\(preview, privacy: .public)\""
                )
            } else {
                logger.debug(
                    "Volatile update #\(self.volatileUpdateCount, privacy: .public) len=\(update.text.count, privacy: .public) preview=\"\(preview, privacy: .public)\""
                )
            }
            if self.volatileUpdateCount == 1 {
                logger.info("First volatile token surfaced at ~\(elapsed ?? 0.0, format: .fixed(precision: 3))s")
            }
            self.volatileText = update.text
        }
        self.scheduleSnapshotUpdate()
    }

    private func cancelActivePipeline(resetStage: Bool = true) {
        transcriptionTask?.cancel()
        transcriptionTask = nil
        microphoneTask?.cancel()
        microphoneTask = nil
        playbackTask?.cancel()
        playbackTask = nil
        playbackController.stop()
        isPlaybackActive = false
        metadata = nil
        let wasMicrophoneStreaming = isMicrophoneStreaming
        isMicrophoneStreaming = false

        Task {
            if wasMicrophoneStreaming {
                await microphoneController.cancel()
            } else {
                await transcriptionCoordinator.cancelActiveSession()
            }
        }

        if resetStage {
            stage = .idle
        }
        resetStreamingStateCounters()
    }

    deinit {
        stopAccessingSelectedFile()
    }

    private func stopAccessingSelectedFile() {
        if let url = securityScopedFileURL {
            url.stopAccessingSecurityScopedResource()
            securityScopedFileURL = nil
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

    private func resetStreamingStateCounters() {
        self.confirmedUpdateCount = 0
        self.volatileUpdateCount = 0
        self.streamingStartTime = nil
        self.pendingSnapshot = nil
        snapshotUpdateTask?.cancel()
        snapshotUpdateTask = nil
        liveSnapshot = StreamingTranscriptSnapshot(confirmedText: confirmedText, volatileText: volatileText)
    }

    private func logStreamingSummary(reason: String) {
        logger.info(
            "Streaming session finished (\(reason, privacy: .public)); confirmedUpdates=\(self.confirmedUpdateCount, privacy: .public) volatileUpdates=\(self.volatileUpdateCount, privacy: .public)"
        )
        resetStreamingStateCounters()
    }

    private func scheduleSnapshotUpdate() {
        let snapshot = StreamingTranscriptSnapshot(confirmedText: confirmedText, volatileText: volatileText)
        pendingSnapshot = snapshot
        guard snapshotUpdateTask == nil else { return }

        snapshotUpdateTask = Task { [weak self] in
            guard let self else { return }
            defer { self.snapshotUpdateTask = nil }

            while !Task.isCancelled {
                guard let nextSnapshot = self.pendingSnapshot else { break }
                self.pendingSnapshot = nil

                let now = Date()
                let elapsed = now.timeIntervalSince(self.lastSnapshotTimestamp)
                if elapsed < self.uiUpdateInterval {
                    let wait = self.uiUpdateInterval - elapsed
                    let delay = UInt64(wait * 1_000_000_000)
                    do {
                        try await Task.sleep(nanoseconds: delay)
                    } catch {
                        break
                    }
                }

                self.lastSnapshotTimestamp = Date()
                self.liveSnapshot = nextSnapshot
            }

            if let remaining = self.pendingSnapshot {
                self.pendingSnapshot = remaining
                self.scheduleSnapshotUpdate()
            }
        }
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

struct StreamingTranscriptSnapshot: Equatable {
    let confirmedText: String
    let volatileText: String

    static let empty = StreamingTranscriptSnapshot(confirmedText: "", volatileText: "")
}
