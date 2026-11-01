import AVFoundation
import FluidAudio
import Foundation

/// A bundled upstream PyTorch take (`Reference/references.json`): one voice,
/// one sentence, for A/B against Core ML on the same text and voice.
struct ReferenceClip: Identifiable, Decodable {
    let voice: String
    let file: String
    let text: String
    var id: String { file }
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
    @Published private(set) var isLoading = false
    @Published private(set) var isReady = false
    /// True from the moment a Core ML utterance is requested until every
    /// pending utterance has finished generating. Set synchronously in
    /// `speak()` so a second click during session creation is rejected.
    @Published private(set) var isSpeaking = false
    @Published private(set) var utterances: [UtteranceStats] = []
    /// PyTorch reference takes shipped with the demo.
    let references: [ReferenceClip] = DemoModel.loadReferences()
    /// The reference row currently playing (file + which renderer), for the highlight.
    @Published private(set) var activeReference: (file: String, source: ReferenceSource)?

    enum ReferenceSource: String {
        case pytorch = "PyTorch"
        case coreml = "Core ML"
    }

    let player = StreamPlayer()

    /// Chosen with `--model`; loaded on launch.
    private(set) var selectedModel: PocketTtsSmallModel = .english7m
    /// `.fp32` (default): FlowLM on the ANE, Mimi fp32 on the CPU. `.fp16Fused`
    /// (`--mimi fp16-fused`): one all-ANE dispatch, faster, with the −72 dBFS hiss.
    private(set) var mimiPrecision: PocketTtsSmallMimiPrecision = .fp32
    private var selectedVoice = "alba"
    /// Sentence spoken by `--autorun`, or the reference row's text.
    private var text =
        "Hello world. I am Kyutai's Pocket TTS. I'm fast enough to run on small CPUs. I hope you'll like me."
    var playAudio = true

    private var manager: PocketTtsSmallManager?
    private var session: PocketTtsSmallSession?
    private var consumer: Task<Void, Never>?
    /// True while `makeSession` is in flight: `pending` is still empty then,
    /// so `tick()` must not treat the model as idle.
    private var openingSession = false
    /// Per-utterance timing, touched on every frame; the published
    /// `utterances` rows are refreshed from it on the 250 ms tick so the UI
    /// never re-renders per frame (that alone cost ~6 ms a frame).
    private var pending: [Int: (enqueuedAt: Date, firstFrameAt: Date?, lastFrameAt: Date?, frames: Int)] = [:]
    private var nextUtterance = 0
    /// Row index of the live session's utterance 0: sessions number their
    /// utterances from 0, rows keep counting across sessions.
    private var sessionBase = 0
    /// Rows kept on screen (older ones are dropped).
    private let maxRows = 4
    /// An utterance that never produced a frame is given up after this long.
    private let firstFrameTimeout: TimeInterval = 10

    /// `--model 7m` picks the model, `--mimi fp32|fp16-fused` the Mimi path,
    /// `--text "…"` the sentence, `--autorun` speaks it once the model is
    /// loaded, `--no-audio` keeps the speaker quiet (headless timing).
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
            case "--mimi" where i + 1 < args.count:
                if let m = PocketTtsSmallMimiPrecision(rawValue: args[i + 1]) { mimiPrecision = m }
                i += 1
            default: break
            }
            i += 1
        }
        load()
        guard autorun else { return }
        Task {
            while !isReady && isLoading { try? await Task.sleep(for: .milliseconds(100)) }
            if isReady { speak() }
        }
    }

    /// The log goes to stdout (run the app from a terminal to see it).
    func log(_ line: String) {
        print(line)
    }

    // MARK: - Model

    func load() {
        guard !isLoading else { return }
        isLoading = true
        isReady = false
        log(
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
                if !voices.contains(selectedVoice) { selectedVoice = voices.first ?? "alba" }
                isReady = true
                if let arch {
                    log(
                        "loaded in \(String(format: "%.2f", dt)) s: d_model \(arch.dModel), \(arch.numLayers) layers, "
                            + "\(arch.numHeads)/\(arch.numKvHeads) heads, flow \(arch.flowType), eos \(arch.eosThreshold), "
                            + "\(voices.count) voices, mimi \(mimi.rawValue)")
                }
            } catch {
                log("load failed: \(error)")
            }
            isLoading = false
        }
    }

    // MARK: - Speaking

    /// Enqueue `text` on the live session; a session is opened on first use
    /// (voice KV snapshot + MLState), then stays warm for further sentences.
    /// Rejects re-entry while a session is being opened or an utterance is
    /// pending, so a double click cannot open two sessions.
    func speak() {
        let line = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty, let manager, !isSpeaking else { return }
        isSpeaking = true
        Task {
            do {
                if session == nil {
                    openingSession = true
                    defer { openingSession = false }
                    try player.begin()
                    let s = try await manager.makeSession(voice: selectedVoice)
                    session = s
                    sessionBase = nextUtterance
                    startConsumer(s)
                    log("── session opened (voice \(selectedVoice))")
                }
                let idx = nextUtterance
                nextUtterance += 1
                pending[idx] = (Date(), nil, nil, 0)
                utterances.append(UtteranceStats(id: idx, text: line))
                if utterances.count > maxRows { utterances.removeFirst(utterances.count - maxRows) }
                session?.enqueue(line)
                log("[\(idx)] enqueued: \(line)")
            } catch {
                log("speak failed: \(error)")
                isSpeaking = false
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
                    let idx = self.sessionBase + (frame.utteranceIndex ?? 0)
                    if var p = self.pending[idx] {
                        if p.firstFrameAt == nil {
                            p.firstFrameAt = now
                            self.log(
                                "[\(idx)] first audio after \(String(format: "%.0f", now.timeIntervalSince(p.enqueuedAt) * 1000)) ms"
                            )
                        }
                        p.lastFrameAt = now
                        p.frames += 1
                        self.pending[idx] = p
                    }
                }
            } catch {
                self?.log("stream error: \(error)")
            }
            // The stream is over (finished, cancelled or failed): drop the dead
            // session so the next speak() opens a fresh one instead of
            // enqueueing into a loop that has exited.
            guard let self, self.session === session else { return }
            self.session = nil
            self.consumer = nil
            self.pending.removeAll()
            self.isSpeaking = false
        }
    }

    private func update(_ idx: Int, _ change: (inout UtteranceStats) -> Void) {
        if let i = utterances.firstIndex(where: { $0.id == idx }) {
            change(&utterances[i])
        }
    }

    /// Called from the view on a timer: refresh the rows from `pending`, log a
    /// summary once an utterance has been quiet for 300 ms, give up on one
    /// that never produced a frame, and clear the highlight when playback is over.
    func tick() {
        let now = Date()
        for (idx, p) in pending {
            update(idx) {
                $0.frames = p.frames
                if let f = p.firstFrameAt {
                    $0.firstAudioMs = f.timeIntervalSince(p.enqueuedAt) * 1000
                    if let l = p.lastFrameAt { $0.generationSeconds = l.timeIntervalSince(f) }
                }
            }
            if let last = p.lastFrameAt, now.timeIntervalSince(last) > 0.3, let first = p.firstFrameAt {
                pending.removeValue(forKey: idx)
                let seconds = Double(p.frames) * 0.08
                let gen = last.timeIntervalSince(first)
                log(
                    "[\(idx)] \(p.frames) frames = \(String(format: "%.2f", seconds)) s audio generated in "
                        + "\(String(format: "%.0f", gen * 1000)) ms → \(String(format: "%.1f", gen > 0 ? seconds / gen : 0))x real time, "
                        + "\(String(format: "%.2f", p.frames > 1 ? gen * 1000 / Double(p.frames - 1) : 0)) ms per 80 ms frame"
                )
            } else if p.firstFrameAt == nil, now.timeIntervalSince(p.enqueuedAt) > firstFrameTimeout {
                pending.removeValue(forKey: idx)
                log("[\(idx)] gave up: no audio after \(Int(firstFrameTimeout)) s")
            }
        }
        if pending.isEmpty, !openingSession { isSpeaking = false }
        if activeReference != nil, pending.isEmpty, !openingSession, !isSpeaking,
            player.playedFrames >= player.scheduledFrames
        {
            activeReference = nil
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
        activeReference = nil
    }

    // MARK: - PyTorch references

    private static func loadReferences() -> [ReferenceClip] {
        guard let url = Bundle.module.url(forResource: "references", withExtension: "json", subdirectory: "Reference"),
            let data = try? Data(contentsOf: url),
            let clips = try? JSONDecoder().decode([ReferenceClip].self, from: data)
        else { return [] }
        return clips
    }

    /// Play a bundled PyTorch take through the same player (and sea view) as
    /// the generated audio, as 80 ms frames.
    func playReference(_ clip: ReferenceClip) {
        guard
            let url = Bundle.module.url(forResource: clip.file, withExtension: nil, subdirectory: "Reference")
        else {
            log("reference \(clip.file) missing from the bundle")
            return
        }
        do {
            stop()
            try player.begin()
            let samples = try Self.readMono24k(url)
            if playAudio {
                let frame = Int(StreamPlayer.sampleRate * 0.08)
                var start = 0
                while start < samples.count {
                    let end = min(samples.count, start + frame)
                    player.enqueue(Array(samples[start..<end]))
                    start = end
                }
            }
            activeReference = (clip.file, .pytorch)
            log(
                "── PyTorch reference (\(clip.voice), \(String(format: "%.2f", Double(samples.count) / StreamPlayer.sampleRate)) s): \(clip.text)"
            )
        } catch {
            log("reference playback failed: \(error)")
        }
    }

    /// Speak the reference sentence with the reference voice through Core ML.
    func speakReference(_ clip: ReferenceClip) {
        guard isReady, !isSpeaking else { return }
        if clip.voice != selectedVoice {
            stop()
            selectedVoice = clip.voice
        }
        text = clip.text
        activeReference = (clip.file, .coreml)
        speak()
    }

    private static func readMono24k(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: StreamPlayer.sampleRate, channels: 1),
            let converter = AVAudioConverter(from: file.processingFormat, to: format),
            let input = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else {
            throw NSError(
                domain: "PocketTtsSmallDemo", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "cannot read \(url.lastPathComponent)"])
        }
        try file.read(into: input)
        let capacity =
            AVAudioFrameCount(Double(input.frameLength) * StreamPlayer.sampleRate / file.processingFormat.sampleRate)
            + 1
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { return [] }
        var consumed = false
        var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if consumed {
                status.pointee = .endOfStream
                return nil
            }
            consumed = true
            status.pointee = .haveData
            return input
        }
        if let error { throw error }
        let n = Int(output.frameLength)
        guard let channel = output.floatChannelData?[0] else { return [] }
        return Array(UnsafeBufferPointer(start: channel, count: n))
    }
}
