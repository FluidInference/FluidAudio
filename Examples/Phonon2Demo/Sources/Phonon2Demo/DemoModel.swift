import AVFoundation
import FluidAudio
import Foundation

/// Phonon-2 is the model under demo; v3 is timed on the same file as the reference.
/// Both run their encoder on the Neural Engine (the FluidAudio default).
enum DemoModelChoice: String, CaseIterable, Identifiable {
    case phonon2
    case v3

    var id: String { rawValue }

    var version: AsrModelVersion {
        switch self {
        case .phonon2: return .phonon2
        case .v3: return .v3
        }
    }

    var title: String {
        switch self {
        case .phonon2: return "Phonon-2"
        case .v3: return "Parakeet TDT v3"
        }
    }

    var detail: String {
        switch self {
        case .phonon2: return "five-value sparse · 321 MB encoder · English"
        case .v3: return "6-bit · 445 MB encoder · 25 languages"
        }
    }

    var repoName: String {
        switch self {
        case .phonon2: return "FluidInference/phonon-2-coreml"
        case .v3: return "FluidInference/parakeet-tdt-0.6b-v3-coreml"
        }
    }

    /// ANSI colour for the console log: Phonon-2 lines red, v3 lines blue.
    var ansi: String {
        switch self {
        case .phonon2: return "\u{1B}[1;31m"
        case .v3: return "\u{1B}[1;34m"
        }
    }
}

struct Word: Identifiable {
    let id: Int
    let text: String
    let start: TimeInterval
    let end: TimeInterval
}

/// A run of words shown as one paragraph; split at long pauses so the transcript
/// renders lazily and stays readable.
struct Paragraph: Identifiable {
    let id: Int
    let words: [Word]
}

struct RunStats {
    let audioSeconds: TimeInterval
    let wallSeconds: TimeInterval
    let model: DemoModelChoice
    var speed: Double { audioSeconds / max(wallSeconds, 1e-6) }
}

@MainActor
final class DemoModel: ObservableObject {
    enum Phase: Equatable {
        case idle
        case loadingModel
        case transcribing
        case done
        case failed(String)
    }

    let choice: DemoModelChoice = .phonon2
    @Published private(set) var fileURL: URL?
    @Published private(set) var fileDuration: TimeInterval?
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var stats: RunStats?
    @Published private(set) var paragraphs: [Paragraph] = []
    @Published private(set) var transcriptText = ""
    @Published private(set) var modelLoadSeconds: TimeInterval?
    /// After the selected model, also time v3 on the same file as a reference.
    @Published var compareWithV3 = true
    /// Every model timed on the current file, including the v3 reference.
    @Published private(set) var runs: [DemoModelChoice: RunStats] = [:]
    @Published private(set) var referenceRunning = false
    /// Which model is transcribing right now, and its wall clock so far (ticks from 0 while it runs).
    @Published private(set) var activeModel: DemoModelChoice?
    @Published private(set) var liveWallSeconds: TimeInterval = 0
    private var liveTimer: Timer?

    let player = TranscriptPlayer()

    private var managers: [DemoModelChoice: AsrManager] = [:]
    private var runTask: Task<Void, Never>?

    var isBusy: Bool { phase == .loadingModel || phase == .transcribing }

    /// `--file <path>` and `--autorun` preselect a run, which makes screen
    /// recordings repeatable.
    func applyLaunchArguments() {
        let args = CommandLine.arguments
        func value(after flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return args[i + 1]
        }
        if let path = value(after: "--file") {
            select(URL(fileURLWithPath: (path as NSString).expandingTildeInPath))
        } else if let sample = samples.first {
            select(sample.url)
        }
        if args.contains("--autorun"), fileURL != nil {
            transcribe()
        }
    }

    /// One hour of stitched Earnings-22 calls, preselected on launch when present.
    var samples: [(label: String, url: URL)] {
        let fm = FileManager.default
        let url = fm.homeDirectoryForCurrentUser.appendingPathComponent(
            "Library/Application Support/FluidAudio/earnings22-1h/earnings22_top4_1h.wav")
        return fm.fileExists(atPath: url.path) ? [("one hour of earnings calls", url)] : []
    }

    func select(_ url: URL) {
        cancel()
        fileURL = url
        stats = nil
        runs = [:]
        liveWallSeconds = 0
        paragraphs = []
        transcriptText = ""
        phase = .idle
        fileDuration = try? Self.duration(of: url)
        player.load(url)
    }

    func transcribe() {
        guard let url = fileURL, !isBusy else { return }
        let choice = choice
        runTask = Task { [weak self] in
            await self?.run(url: url, choice: choice)
        }
    }

    /// Back to the launch state for the current file: zeros everywhere, no transcript.
    /// Loaded models stay cached so the next run is timed without a reload.
    func reset() {
        cancel()
        stats = nil
        runs = [:]
        paragraphs = []
        transcriptText = ""
        liveWallSeconds = 0
        phase = .idle
        if let url = fileURL { player.load(url) }
    }

    var hasResults: Bool { stats != nil || !runs.isEmpty || !paragraphs.isEmpty }

    func cancel() {
        runTask?.cancel()
        runTask = nil
        referenceRunning = false
        stopLiveClock()
        if isBusy { phase = .idle }
    }

    /// Every number on screen starts at zero and counts up while a model runs.
    private func startLiveClock(for model: DemoModelChoice) {
        activeModel = model
        liveWallSeconds = 0
        let start = ContinuousClock.now
        liveTimer?.invalidate()
        liveTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.liveWallSeconds = Self.seconds(ContinuousClock.now - start) }
        }
    }

    private func stopLiveClock() {
        liveTimer?.invalidate()
        liveTimer = nil
        activeModel = nil
    }

    private func run(url: URL, choice: DemoModelChoice) async {
        do {
            if managers[choice] == nil { phase = .loadingModel }
            let manager = try await loadManager(choice)
            try Task.checkCancellation()

            phase = .transcribing
            stats = nil
            paragraphs = []
            transcriptText = ""
            console(
                choice,
                "transcribing \(url.lastPathComponent) (\(Self.clock(fileDuration ?? 0)) of audio), encoder on the Neural Engine"
            )
            let (result, wall) = try await timedTranscribe(url, with: manager, model: choice)
            try Task.checkCancellation()

            let audioSeconds = fileDuration ?? result.duration
            let runStats = RunStats(audioSeconds: audioSeconds, wallSeconds: wall, model: choice)
            stats = runStats
            runs[choice] = runStats
            transcriptText = result.text
            paragraphs = Self.paragraphs(from: result.tokenTimings ?? [], fallbackText: result.text)
            phase = .done
            log(runStats, url)

            // The reference run only times v3; the transcript on screen stays the selected model's.
            guard compareWithV3, choice != .v3, runs[.v3] == nil else { return }
            referenceRunning = true
            defer { referenceRunning = false }
            let reference = try await loadManager(.v3)
            try Task.checkCancellation()
            console(.v3, "transcribing the same file as the reference, encoder on the Neural Engine")
            let (refResult, refWall) = try await timedTranscribe(url, with: reference, model: .v3)
            try Task.checkCancellation()
            let refStats = RunStats(
                audioSeconds: fileDuration ?? refResult.duration, wallSeconds: refWall, model: .v3)
            runs[.v3] = refStats
            log(refStats, url)
        } catch is CancellationError {
            phase = .idle
        } catch {
            phase = .failed(error.localizedDescription)
        }
    }

    private func loadManager(_ choice: DemoModelChoice) async throws -> AsrManager {
        if let cached = managers[choice] { return cached }
        console(
            choice,
            "loading \(choice.repoName) (\(choice.detail)); first load downloads and compiles for the Neural Engine")
        let loadStart = ContinuousClock.now
        let models = try await AsrModels.downloadAndLoad(version: choice.version)
        let loaded = AsrManager(config: .default)
        try await loaded.loadModels(models)
        managers[choice] = loaded
        let seconds = Self.seconds(ContinuousClock.now - loadStart)
        modelLoadSeconds = seconds
        console(choice, String(format: "models ready in %.1f s", seconds))
        return loaded
    }

    private func timedTranscribe(
        _ url: URL, with manager: AsrManager, model: DemoModelChoice
    ) async throws
        -> (ASRResult, TimeInterval)
    {
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        startLiveClock(for: model)
        defer { stopLiveClock() }
        let start = ContinuousClock.now
        let result = try await manager.transcribe(url, decoderState: &state)
        return (result, Self.seconds(ContinuousClock.now - start))
    }

    private func log(_ stats: RunStats, _ url: URL) {
        console(
            stats.model,
            String(
                format: "done: %.1f s of audio in %.2f s wall = %.0fx real-time", stats.audioSeconds,
                stats.wallSeconds, stats.speed))
        if stats.model != .v3, let reference = runs[.v3] {
            let ratio = reference.wallSeconds / max(stats.wallSeconds, 1e-6)
            console(stats.model, String(format: "%.2fx faster than Parakeet TDT v3 on this file", ratio))
        } else if stats.model == .v3, let demo = runs[choice] {
            let ratio = stats.wallSeconds / max(demo.wallSeconds, 1e-6)
            console(choice, String(format: "%.2fx faster than Parakeet TDT v3 on this file", ratio))
        }
    }

    /// Timestamped, colour-coded line for the terminal log pane (stdout is line-buffered and tailed by demo.sh).
    private func console(_ model: DemoModelChoice, _ message: String) {
        let stamp = Self.timestamp.string(from: Date())
        print("\u{1B}[2m\(stamp)\u{1B}[0m \(model.ansi)[\(model.title)]\u{1B}[0m \(message)")
    }

    private static let timestamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    static func clock(_ seconds: TimeInterval) -> String {
        let total = Int(seconds.rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        if hours > 0 { return "\(hours)h \(minutes)m \(secs)s" }
        return "\(minutes)m \(secs)s"
    }

    private static func seconds(_ duration: Duration) -> TimeInterval {
        let parts = duration.components
        return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) / 1e18
    }

    private static func duration(of url: URL) throws -> TimeInterval {
        let file = try AVAudioFile(forReading: url)
        return Double(file.length) / file.processingFormat.sampleRate
    }

    /// Merges SentencePiece tokens into words (a leading space starts a new word)
    /// and starts a new paragraph after a pause of 1.2 s or ~60 words.
    private static func paragraphs(from timings: [TokenTiming], fallbackText: String) -> [Paragraph] {
        guard !timings.isEmpty else {
            let words = fallbackText.split(separator: " ").enumerated().map {
                Word(id: $0.offset, text: String($0.element), start: 0, end: 0)
            }
            return words.isEmpty ? [] : [Paragraph(id: 0, words: words)]
        }

        var words: [Word] = []
        var text = ""
        var start: TimeInterval = 0
        var end: TimeInterval = 0
        func flush() {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                words.append(Word(id: words.count, text: trimmed, start: start, end: end))
            }
            text = ""
        }
        for timing in timings {
            if timing.token.hasPrefix(" ") || text.isEmpty {
                flush()
                start = timing.startTime
            }
            text += timing.token
            end = timing.endTime
        }
        flush()

        var paragraphs: [Paragraph] = []
        var current: [Word] = []
        for word in words {
            if let last = current.last, word.start - last.end > 1.2 || current.count >= 60 {
                paragraphs.append(Paragraph(id: paragraphs.count, words: current))
                current = []
            }
            current.append(word)
        }
        if !current.isEmpty {
            paragraphs.append(Paragraph(id: paragraphs.count, words: current))
        }
        return paragraphs
    }
}
