import FluidAudio
import Foundation

/// One line of the on-screen log (mirrored to stdout for the terminal pane).
struct LogLine: Identifiable {
    let id: Int
    let text: String
}

/// Per-utterance measurements.
struct UtteranceStats: Identifiable {
    let id: Int
    let text: String
    var firstAudioMs: Double?  // text enqueued -> first frame
    var frames = 0
    var generationSeconds: Double = 0  // first frame -> last frame
    var audioSeconds: Double { Double(frames) * 0.08 }
    var realTimeFactor: Double { generationSeconds > 0 ? audioSeconds / generationSeconds : 0 }
    var msPerFrame: Double { frames > 1 ? generationSeconds * 1000 / Double(frames - 1) : 0 }
}

@MainActor
final class DemoModel: ObservableObject {
    @Published var selectedModel: PocketTtsSmallModel = .english7m
    @Published var voices: [String] = ["alba"]
    @Published var selectedVoice = "alba"
    @Published var text =
        "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me."
    @Published private(set) var status = "Pick a model and press Load."
    @Published private(set) var isLoading = false
    @Published private(set) var isReady = false
    @Published private(set) var isSpeaking = false
    @Published private(set) var loadSeconds: Double?
    @Published private(set) var arch: PocketTtsSmallArch?
    @Published private(set) var utterances: [UtteranceStats] = []
    @Published private(set) var log: [LogLine] = []

    let player = StreamPlayer()

    private var manager: PocketTtsSmallManager?
    private var session: PocketTtsSmallSession?
    private var consumer: Task<Void, Never>?
    /// Per-utterance timing, touched on every frame; the published
    /// `utterances` table is refreshed from it on the 250 ms tick so the UI
    /// never re-renders per frame (that alone cost ~6 ms a frame).
    private var pending: [Int: (enqueuedAt: Date, firstFrameAt: Date?, lastFrameAt: Date?, frames: Int)] = [:]
    private var finished: Set<Int> = []
    var playAudio = true
    /// `--mimi-fp16`: the fused all-ANE path (faster, −72 dBFS hiss) instead of
    /// the default fp32 Mimi decoder on the CPU.
    var mimiPrecision: PocketTtsSmallMimiPrecision = .fp32
    private var nextUtterance = 0
    private var logCounter = 0

    /// `--model 7m` picks the model, `--mimi-fp16` the fused all-ANE decoder,
    /// `--autorun` loads it and speaks the text box on launch (used by the
    /// showcase script and for headless verification).
    func applyLaunchArguments(_ args: [String]) {
        var autorun = false
        var i = 1
        while i < args.count {
            switch args[i] {
            case "--model" where i + 1 < args.count:
                if let m = PocketTtsSmallModel(name: args[i + 1]) { selectedModel = m }
                i += 1
            case "--text" where i + 1 < args.count:
                text = args[i + 1]
                i += 1
            case "--autorun":
                autorun = true
            case "--no-audio":
                playAudio = false
            case "--mimi-fp16":
                mimiPrecision = .fp16Fused
            default: break
            }
            i += 1
        }
        guard autorun else { return }
        load()
        Task {
            while !isReady && isLoading { try? await Task.sleep(for: .milliseconds(100)) }
            if isReady { speak() }
        }
    }

    func appendLog(_ line: String) {
        logCounter += 1
        log.append(LogLine(id: logCounter, text: line))
        if log.count > 400 { log.removeFirst(log.count - 400) }
        print(line)
    }

    // MARK: - Model

    func load() {
        guard !isLoading else { return }
        isLoading = true
        isReady = false
        status = "Downloading / loading \(selectedModel.rawValue)…"
        appendLog(
            "── loading \(selectedModel.rawValue) (first load compiles for the Neural Engine, 10–150 s; later loads < 1 s)"
        )
        let model = selectedModel
        let mimi = mimiPrecision
        Task {
            do {
                let t0 = Date()
                let manager = PocketTtsSmallManager(model: model, mimiPrecision: mimi)
                try await manager.initialize()
                let dt = Date().timeIntervalSince(t0)
                let arch = await manager.arch
                let voices = try await manager.availableVoices()
                self.manager = manager
                self.arch = arch
                self.voices = voices
                if !voices.contains(selectedVoice) { selectedVoice = voices.first ?? "alba" }
                loadSeconds = dt
                isReady = true
                status = "\(model.rawValue) ready — \(String(format: "%.2f", dt)) s load, mimi \(mimi.rawValue)"
                if let arch {
                    appendLog(
                        "loaded in \(String(format: "%.2f", dt)) s: d_model \(arch.dModel), \(arch.numLayers) layers, "
                            + "\(arch.numHeads)/\(arch.numKvHeads) heads, flow \(arch.flowType), eos \(arch.eosThreshold), "
                            + "\(voices.count) voices")
                }
            } catch {
                status = "Load failed: \(error)"
                appendLog("load failed: \(error)")
            }
            isLoading = false
        }
    }

    func selectModel(_ model: PocketTtsSmallModel) {
        guard model != selectedModel else { return }
        stop()
        selectedModel = model
        isReady = false
        manager = nil
        arch = nil
        loadSeconds = nil
        status = "Press Load for \(model.rawValue)."
    }

    // MARK: - Speaking

    /// Enqueue the text box on the live session; a session is opened on first
    /// use (voice KV snapshot + MLState), then stays warm for further sentences.
    func speak() {
        let line = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty, let manager else { return }
        Task {
            do {
                if session == nil {
                    try player.begin()
                    let s = try await manager.makeSession(voice: selectedVoice)
                    session = s
                    startConsumer(s)
                    appendLog("── session opened (voice \(selectedVoice))")
                }
                let idx = nextUtterance
                nextUtterance += 1
                pending[idx] = (Date(), nil, nil, 0)
                utterances.append(UtteranceStats(id: idx, text: line))
                isSpeaking = true
                session?.enqueue(line)
                appendLog("[\(idx)] enqueued: \(line)")
            } catch {
                status = "Speak failed: \(error)"
                appendLog("speak failed: \(error)")
            }
        }
    }

    private func startConsumer(_ session: PocketTtsSmallSession) {
        consumer?.cancel()
        consumer = Task { [weak self] in
            do {
                for try await frame in session.frames {
                    guard let self else { return }
                    if self.playAudio { self.player.enqueue(frame.samples) }
                    let now = Date()
                    let idx = frame.utteranceIndex ?? 0
                    if var p = self.pending[idx] {
                        if p.firstFrameAt == nil {
                            p.firstFrameAt = now
                            self.appendLog(
                                "[\(idx)] first audio after \(String(format: "%.0f", now.timeIntervalSince(p.enqueuedAt) * 1000)) ms"
                            )
                        }
                        p.lastFrameAt = now
                        p.frames += 1
                        self.pending[idx] = p
                    }
                }
            } catch {
                self?.appendLog("stream error: \(error)")
            }
            self?.isSpeaking = false
        }
    }

    private func update(_ idx: Int, _ change: (inout UtteranceStats) -> Void) {
        if let i = utterances.firstIndex(where: { $0.id == idx }) {
            change(&utterances[i])
        }
    }

    /// Called from the view on a timer: refresh the table from `pending` and
    /// log a summary once an utterance has been quiet for 300 ms.
    func tick() {
        for (idx, p) in pending {
            update(idx) {
                $0.frames = p.frames
                if let f = p.firstFrameAt {
                    $0.firstAudioMs = f.timeIntervalSince(p.enqueuedAt) * 1000
                    if let l = p.lastFrameAt { $0.generationSeconds = l.timeIntervalSince(f) }
                }
            }
            guard let last = p.lastFrameAt, Date().timeIntervalSince(last) > 0.3,
                let stats = utterances.first(where: { $0.id == idx })
            else { continue }
            pending.removeValue(forKey: idx)
            appendLog(
                "[\(idx)] \(stats.frames) frames = \(String(format: "%.2f", stats.audioSeconds)) s audio generated in "
                    + "\(String(format: "%.0f", stats.generationSeconds * 1000)) ms → \(String(format: "%.1f", stats.realTimeFactor))x real time, "
                    + "\(String(format: "%.2f", stats.msPerFrame)) ms per 80 ms frame")
            if pending.isEmpty {
                isSpeaking = false
                status = "Ready — \(selectedModel.rawValue), voice \(selectedVoice)"
            }
        }
    }

    func stop() {
        consumer?.cancel()
        consumer = nil
        if let s = session {
            Task { await s.cancel() }
        }
        session = nil
        pending.removeAll()
        player.stop()
        isSpeaking = false
        if isReady { status = "Stopped." }
    }

    func changeVoice(_ voice: String) {
        guard voice != selectedVoice else { return }
        stop()
        selectedVoice = voice
    }

    func clearStats() {
        utterances.removeAll()
        log.removeAll()
    }
}
