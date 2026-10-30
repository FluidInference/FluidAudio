import AVFoundation
import FluidAudio
import Foundation

/// The models the typewriter can run. All three download from Hugging Face on first use
/// (`FluidInference/bark-3.0-coreml`, `FluidInference/parakeet-tdt-0.6b-v3-coreml`); `--bark-dir` /
/// `--clean-dir` load a local folder in the same layout instead.
enum DemoModelChoice: String, CaseIterable, Identifiable {
    case bark
    case barkClean
    case v3

    var id: String { rawValue }

    var title: String {
        switch self {
        case .barkClean: return "Bark 3.0 Clean"
        case .bark: return "Bark 3.0"
        case .v3: return "Parakeet TDT v3"
        }
    }

    var detail: String {
        switch self {
        case .barkClean: return "drops fillers and repeats · 410 MB encoder"
        case .bark: return "verbatim · 399 MB encoder"
        case .v3: return "reference · 445 MB encoder · 25 languages"
        }
    }

    /// ANSI colour for the log pane.
    var ansi: String {
        switch self {
        case .bark: return "\u{1B}[1;33m"
        case .barkClean: return "\u{1B}[1;32m"
        case .v3: return "\u{1B}[1;34m"
        }
    }

    /// Where `downloadAndLoad` fetches the model from, for the log.
    var hubPath: String {
        switch self {
        case .bark: return "FluidInference/bark-3.0-coreml/bark-3.0"
        case .barkClean: return "FluidInference/bark-3.0-coreml/bark-3.0-clean"
        case .v3: return "FluidInference/parakeet-tdt-0.6b-v3-coreml"
        }
    }

    var version: AsrModelVersion {
        switch self {
        case .bark: return .bark3
        case .barkClean: return .bark3Clean
        case .v3: return .v3
        }
    }
}

@MainActor
final class DemoModel: ObservableObject {
    static let shared = DemoModel()

    enum Phase: Equatable {
        case idle
        case loadingModel
        case transcribing
        case done
        case failed(String)
    }

    @Published var choice: DemoModelChoice = .bark
    @Published private(set) var fileURL: URL?
    @Published private(set) var fileDuration: TimeInterval = 0
    @Published private(set) var waveform: [Float] = []
    @Published private(set) var phase: Phase = .idle
    @Published private(set) var status = ""

    /// Audio the model has finished (end of the last in-order window), the run's wall clock,
    /// and the text typed so far.
    @Published private(set) var decodedSeconds: TimeInterval = 0
    @Published private(set) var wallSeconds: TimeInterval = 0
    @Published private(set) var typed = ""
    @Published private(set) var windowsDone = 0
    @Published private(set) var windowCount = 0
    /// Set when the run ends: the final, seam-merged transcript replaces the window-by-window text.
    @Published private(set) var finalText: String?
    /// True once the typewriter has caught up after the run: the view then shows `finalText`.
    @Published private(set) var showFinal = false
    @Published private(set) var isTyping = false
    @Published private(set) var lastSpeed: Double?

    var isBusy: Bool { phase == .loadingModel || phase == .transcribing }
    var liveSpeed: Double { decodedSeconds / max(wallSeconds, 1e-3) }

    private var directories: [DemoModelChoice: URL] = [:]
    private var managers: [DemoModelChoice: AsrManager] = [:]
    private var runTask: Task<Void, Never>?

    // Typewriter state: windows arrive out of order, are merged in index order into `target`,
    // and a 60 Hz tick types `target` out, catching up quickly when the model is ahead.
    private var pending: [Int: AsrManager.WindowProgress] = [:]
    private var nextWindow = 0
    private var targetWords: [String] = []
    private var target: [Character] = []
    private var typedCount = 0
    private var tick: Timer?
    private var runStart = ContinuousClock.now
    private var running = false
    private var activeChoice: DemoModelChoice = .bark
    private var nextLogFraction = 0.1

    /// `--file <wav>`, `--model bark|clean|v3` (default bark), `--clean-dir <dir>`, `--bark-dir <dir>`, `--autorun`.
    private var launchArgumentsApplied = false

    func applyLaunchArguments() {
        guard !launchArgumentsApplied else { return }
        launchArgumentsApplied = true
        let args = CommandLine.arguments
        func value(after flag: String) -> String? {
            guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
            return (args[i + 1] as NSString).expandingTildeInPath
        }
        if let dir = value(after: "--clean-dir") { directories[.barkClean] = URL(fileURLWithPath: dir) }
        if let dir = value(after: "--bark-dir") { directories[.bark] = URL(fileURLWithPath: dir) }
        switch value(after: "--model") {
        case "clean": choice = .barkClean
        case "v3": choice = .v3
        default: break
        }
        let sample = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(
            "Library/Application Support/FluidAudio/earnings22-1h/earnings22_top4_1h.wav")
        if let path = value(after: "--file") {
            select(URL(fileURLWithPath: path))
        } else if FileManager.default.fileExists(atPath: sample.path) {
            select(sample)
        }
        if args.contains("--autorun") { start() }
    }

    func select(_ url: URL) {
        cancel()
        fileURL = url
        resetRun()
        phase = .idle
        fileDuration = (try? Self.duration(of: url)) ?? 0
        waveform = []
        Task.detached(priority: .userInitiated) {
            let peaks = (try? Self.peaks(of: url, bins: 600)) ?? []
            await MainActor.run { [weak self] in
                if self?.fileURL == url { self?.waveform = peaks }
            }
        }
    }

    func start() {
        guard let url = fileURL, !isBusy else { return }
        let choice = choice
        runTask = Task { [weak self] in await self?.run(url: url, choice: choice) }
    }

    func cancel() {
        runTask?.cancel()
        runTask = nil
        stopTick()
        if isBusy { phase = .idle }
    }

    private func resetRun() {
        pending = [:]
        nextWindow = 0
        targetWords = []
        target = []
        typedCount = 0
        typed = ""
        finalText = nil
        showFinal = false
        decodedSeconds = 0
        wallSeconds = 0
        windowsDone = 0
        windowCount = 0
    }

    private func run(url: URL, choice: DemoModelChoice) async {
        do {
            if managers[choice] == nil {
                phase = .loadingModel
                status = "loading \(choice.title) (first run downloads and compiles for the Neural Engine)"
            }
            let manager = try await loadManager(choice)
            try Task.checkCancellation()

            resetRun()
            activeChoice = choice
            nextLogFraction = 0.1
            phase = .transcribing
            status = "\(choice.title) · encoder on the Neural Engine"
            console(
                choice,
                "transcribing \(url.lastPathComponent) (\(Self.clock(fileDuration)) of audio), encoder on the Neural Engine"
            )
            await manager.setWindowHandler { [weak self] window in
                Task { @MainActor in self?.receive(window) }
            }
            var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
            startTick()
            let result = try await manager.transcribe(url, decoderState: &state)
            let wall = Self.seconds(ContinuousClock.now - runStart)
            running = false
            await manager.setWindowHandler(nil)
            try Task.checkCancellation()

            wallSeconds = wall
            decodedSeconds = fileDuration
            windowsDone = max(windowsDone, windowCount)
            lastSpeed = fileDuration / max(wall, 1e-3)
            finalText = result.text
            // Short files decode as one window and report none: type the final text instead.
            if target.isEmpty { append(words: Self.words(result.text)) }
            status = String(
                format: "%@ · %@ of audio in %.1f s · %.0f× real time", choice.title,
                Self.clock(fileDuration), wall, lastSpeed ?? 0)
            console(
                choice,
                String(
                    format: "done: %@ of audio in %.2f s wall = %.0fx real time, %d words", Self.clock(fileDuration),
                    wall, lastSpeed ?? 0, Self.words(result.text).count))
            phase = .done
        } catch is CancellationError {
            stopTick()
            phase = .idle
            console(choice, "stopped")
        } catch {
            stopTick()
            phase = .failed(error.localizedDescription)
            console(choice, "failed: \(error.localizedDescription)")
        }
    }

    private func loadManager(_ choice: DemoModelChoice) async throws -> AsrManager {
        if let cached = managers[choice] { return cached }
        let version = choice.version
        let source = directories[choice]?.path ?? choice.hubPath
        console(choice, "loading \(source) (\(choice.detail)); first run downloads and compiles for the Neural Engine")
        let loadStart = ContinuousClock.now
        let models: AsrModels
        if let dir = directories[choice] {
            // loadLocal is synchronous and the first load compiles for the Neural Engine; keep it off the main actor.
            models = try await Task.detached(priority: .userInitiated) {
                try AsrModels.loadLocal(from: dir, version: version)
            }.value
        } else {
            models = try await AsrModels.downloadAndLoad(version: version)
        }
        let manager = AsrManager(config: .default)
        try await manager.loadModels(models)
        // Warm-up: the Neural Engine finishes compiling on the first prediction, keep that out of the timing.
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        _ = try await manager.transcribe([Float](repeating: 0, count: 16_000), decoderState: &state)
        managers[choice] = manager
        console(choice, String(format: "models ready in %.1f s", Self.seconds(ContinuousClock.now - loadStart)))
        return manager
    }

    // MARK: - Typewriter

    private func receive(_ window: AsrManager.WindowProgress) {
        guard running else { return }
        windowCount = window.windowCount
        pending[window.index] = window
        while let next = pending.removeValue(forKey: nextWindow) {
            append(words: Self.words(next.text))
            decodedSeconds = min(next.endSeconds, fileDuration)
            nextWindow += 1
            windowsDone = nextWindow
        }
        while fileDuration > 0, decodedSeconds / fileDuration >= nextLogFraction, nextLogFraction < 1 {
            console(
                activeChoice,
                String(
                    format: "%3.0f%%  %@ of audio after %.1f s  (%.0fx)  window %d/%d", nextLogFraction * 100,
                    Self.clock(decodedSeconds), wallSeconds, liveSpeed, windowsDone, windowCount))
            nextLogFraction += 0.1
        }
    }

    /// Windows overlap by a few seconds, so a window starts by re-decoding the tail of the text so far, not always
    /// word for word. Anchor on the last 3-word run the new window shares with that tail and keep what follows it.
    private func append(words: [String]) {
        let keys = words.map(Self.key)
        let tail = Array(targetWords.suffix(25).map(Self.key))
        let skip = Self.overlap(tail: tail, head: Array(keys.prefix(25)))
        let fresh = words.dropFirst(skip)
        guard !fresh.isEmpty else { return }
        targetWords.append(contentsOf: fresh)
        target.append(contentsOf: (target.isEmpty ? "" : " ") + fresh.joined(separator: " "))
    }

    /// Number of leading `head` words already covered by `tail`: the latest 3-gram of `tail` that also occurs in
    /// `head` aligns the two, and everything in `head` up to the end of `tail` is dropped. 0 when nothing matches.
    static func overlap(tail: [String], head: [String], n: Int = 3) -> Int {
        guard tail.count >= n, head.count >= n else { return 0 }
        for j in stride(from: tail.count - n, through: 0, by: -1) {
            let gram = tail[j..<(j + n)]
            for i in 0...(head.count - n) where head[i..<(i + n)].elementsEqual(gram) {
                return min(head.count, i + (tail.count - j))
            }
        }
        return 0
    }

    private func startTick() {
        runStart = ContinuousClock.now
        running = true
        isTyping = true
        tick?.invalidate()
        tick = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.step() }
        }
    }

    private func stopTick() {
        running = false
        isTyping = false
        tick?.invalidate()
        tick = nil
    }

    private func step() {
        if running { wallSeconds = Self.seconds(ContinuousClock.now - runStart) }
        let behind = target.count - typedCount
        if behind > 0 {
            // At least a few characters per frame; type out whatever is queued within ~0.15 s.
            typedCount += max(3, behind / 9)
            typedCount = min(typedCount, target.count)
            typed = String(target[0..<typedCount])
        } else if !running {
            showFinal = finalText != nil
            stopTick()
        }
    }

    // MARK: - Helpers

    /// Timestamped, colour-coded line for the log pane (demo.sh tails stdout).
    private func console(_ model: DemoModelChoice, _ message: String) {
        let stamp = Self.timestamp.string(from: Date())
        print("\u{1B}[2m\(stamp)\u{1B}[0m \(model.ansi)[\(model.title)]\u{1B}[0m \(message)")
    }

    private static let timestamp: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f
    }()

    private static func words(_ text: String) -> [String] {
        text.split(whereSeparator: \.isWhitespace).map(String.init)
    }

    private static func key(_ word: String) -> String {
        word.lowercased().filter { $0.isLetter || $0.isNumber || $0 == "'" }
    }

    static func clock(_ seconds: TimeInterval) -> String {
        let total = Int(seconds.rounded())
        let h = total / 3600
        let m = (total % 3600) / 60
        let s = total % 60
        return h > 0 ? String(format: "%d:%02d:%02d", h, m, s) : String(format: "%d:%02d", m, s)
    }

    private static func seconds(_ duration: Duration) -> TimeInterval {
        let parts = duration.components
        return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) / 1e18
    }

    nonisolated private static func duration(of url: URL) throws -> TimeInterval {
        let file = try AVAudioFile(forReading: url)
        return Double(file.length) / file.processingFormat.sampleRate
    }

    /// Peak amplitude per bin over the whole file, for the tape view.
    nonisolated private static func peaks(of url: URL, bins: Int) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let total = Int(file.length)
        guard total > 0 else { return [] }
        let perBin = max(1, total / bins)
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(perBin)) else {
            return []
        }
        var out: [Float] = []
        out.reserveCapacity(bins)
        while out.count < bins {
            buffer.frameLength = 0
            try file.read(into: buffer, frameCount: AVAudioFrameCount(perBin))
            guard buffer.frameLength > 0, let data = buffer.floatChannelData?[0] else { break }
            var peak: Float = 0
            for i in 0..<Int(buffer.frameLength) { peak = max(peak, abs(data[i])) }
            out.append(peak)
        }
        let top = max(out.max() ?? 1, 1e-6)
        return out.map { $0 / top }
    }
}
