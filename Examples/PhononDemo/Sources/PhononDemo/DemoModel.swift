import AppKit
import FluidAudio
import Foundation
import SwiftUI

/// A voice as shown in the picker.
struct VoiceChoice: Identifiable, Hashable {
    let id: String  // pack voice name, e.g. en_us_Easton
    let label: String  // Easton
    let detail: String  // US, UK, character
    let isCharacter: Bool

    init(_ id: String) {
        self.id = id
        if id == "default" {
            label = "Default"
            detail = "checkpoint"
            isCharacter = false
        } else if id.hasPrefix("characters_") {
            label = String(id.dropFirst("characters_".count))
            detail = "character"
            isCharacter = true
        } else {
            let parts = id.split(separator: "_", maxSplits: 2).map(String.init)
            label = parts.count == 3 ? parts[2] : id
            detail = parts.count == 3 ? "\(parts[0])-\(parts[1])".uppercased() : ""
            isCharacter = false
        }
    }
}

@MainActor
final class DemoModel: ObservableObject {
    /// Shown as chips; everything else in the pack is in the "More" menu.
    static let featured = [
        "en_us_Easton", "en_us_Zoey", "en_us_Juniper", "en_us_Miles", "en_gb_Maeve", "en_gb_Hugh", "default",
    ]
    static let featuredCharacters = [
        "characters_Captain Silas Blackwater", "characters_The Snow Queen", "characters_Vyrmathrax the Ancient",
        "characters_Fennick the Young Elf", "characters_Master Aldric", "characters_Grub the Ogre",
    ]
    static let presets: [(title: String, text: String)] = [
        ("Hello", "Hello! I'm Phonon, a thirty-one million parameter voice, running right here on the Neural Engine."),
        (
            "Numbers",
            "Your order of $42.50 ships on 3 boxes. Email support@gradium.ai or visit gradium.ai/phonon for 24/7 help."
        ),
        ("Pauses", "Wait. Let me think about that for a second... Okay. I have an answer, and you won't like it."),
        ("Story", "The dragon opened one golden eye, and the whole mountain held its breath."),
    ]

    @Published var text = DemoModel.presets[0].text
    @Published private(set) var voices: [VoiceChoice] = []
    @Published private(set) var selectedVoice = VoiceChoice("en_us_Easton")
    @Published private(set) var status = "Loading Phonon…"
    @Published private(set) var isReady = false
    @Published private(set) var isSpeaking = false
    @Published private(set) var spokenText = ""
    @Published private(set) var loadSeconds: Double?
    @Published private(set) var firstAudioMs: Double?
    @Published private(set) var msPerFrame: Double?
    @Published private(set) var realTime: Double?

    let player = StreamPlayer()
    var playAudio = true
    private var mimiPrecision: PocketTtsSmallMimiPrecision = .fp32
    private var renderDirectory: URL?
    private var manager: PocketTtsSmallManager?
    private var session: PocketTtsSmallSession?
    private var sessionVoice: String?
    private var consumer: Task<Void, Never>?
    private var enqueuedAt: Date?
    private var firstFrameAt: Date?
    private var lastFrameAt: Date?
    private var frames = 0

    var featuredVoices: [VoiceChoice] { Self.featured.map(VoiceChoice.init).filter { v in voices.contains(v) } }
    var featuredCharacters: [VoiceChoice] {
        Self.featuredCharacters.map(VoiceChoice.init).filter { v in voices.contains(v) }
    }
    var otherVoices: [VoiceChoice] {
        let shown = Set(Self.featured + Self.featuredCharacters)
        return voices.filter { !shown.contains($0.id) }
    }

    /// `--text "…"`, `--voice en_gb_Maeve`, `--autorun` (speak on launch),
    /// `--mute` (play silently, ring still moves), `--no-audio` (no playback at all),
    /// `--mimi fp16-fused`.
    func applyLaunchArguments(_ args: [String]) {
        var autorun = false
        var voice: String?
        var i = 1
        while i < args.count {
            switch args[i] {
            case "--text" where i + 1 < args.count:
                text = args[i + 1]
                i += 1
            case "--voice" where i + 1 < args.count:
                voice = args[i + 1]
                i += 1
            case "--mimi" where i + 1 < args.count:
                if let m = PocketTtsSmallMimiPrecision(rawValue: args[i + 1]) { mimiPrecision = m }
                i += 1
            case "--autorun": autorun = true
            case "--no-audio": playAudio = false
            case "--mute": player.setMuted(true)
            case "--render" where i + 1 < args.count:
                renderDirectory = URL(fileURLWithPath: args[i + 1])
                i += 1
            default: break
            }
            i += 1
        }
        if let voice { selectedVoice = VoiceChoice(voice) }
        Task {
            await load()
            if let dir = renderDirectory, isReady {
                await renderFrames(to: dir)
                NSApplication.shared.terminate(nil)
            }
            if autorun, isReady { speak() }
        }
    }

    private func load() async {
        let t0 = Date()
        print("── loading phonon-31m-en (first run downloads ~171 MB and compiles for the Neural Engine)")
        do {
            let manager = PocketTtsSmallManager(model: .phonon31m, mimiPrecision: mimiPrecision)
            try await manager.initialize()
            let names = try await manager.availableVoices().filter {
                $0 == "default" || $0.hasPrefix("en_") || $0.hasPrefix("characters_")
            }
            self.manager = manager
            voices = names.map(VoiceChoice.init)
            if !voices.contains(selectedVoice) { selectedVoice = voices.first ?? VoiceChoice("default") }
            loadSeconds = Date().timeIntervalSince(t0)
            isReady = true
            status = "Ready"
            print("loaded in \(String(format: "%.2f", loadSeconds ?? 0)) s, \(voices.count) English voices")
        } catch {
            status = "Load failed: \(error.localizedDescription)"
            print("load failed: \(error)")
        }
    }

    func select(_ voice: VoiceChoice) {
        guard voice != selectedVoice else { return }
        stop()
        selectedVoice = voice
    }

    func toggle() { isSpeaking ? stop() : speak() }

    func speak() {
        let line = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !line.isEmpty, let manager, isReady else { return }
        Task {
            do {
                if session == nil || sessionVoice != selectedVoice.id {
                    stop()
                    try player.begin()
                    status = "Preparing \(selectedVoice.label)…"
                    let s = try await manager.makeSession(voice: selectedVoice.id)
                    session = s
                    sessionVoice = selectedVoice.id
                    startConsumer(s)
                }
                enqueuedAt = Date()
                firstFrameAt = nil
                lastFrameAt = nil
                frames = 0
                spokenText = line
                isSpeaking = true
                status = "Speaking"
                session?.enqueue(line)
                print("[\(selectedVoice.id)] \(line)")
            } catch {
                status = "Speak failed: \(error.localizedDescription)"
                print("speak failed: \(error)")
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
                    if self.firstFrameAt == nil {
                        self.firstFrameAt = now
                        if let e = self.enqueuedAt { self.firstAudioMs = now.timeIntervalSince(e) * 1000 }
                    }
                    self.lastFrameAt = now
                    self.frames += 1
                }
            } catch {
                print("stream error: \(error)")
            }
        }
    }

    /// Called on a timer by the view: finish the stats once frames stop arriving
    /// and the speaker has played them out.
    func tick() {
        guard isSpeaking, let first = firstFrameAt, let last = lastFrameAt else { return }
        let gen = last.timeIntervalSince(first)
        if frames > 1, gen > 0 {
            msPerFrame = gen * 1000 / Double(frames - 1)
            realTime = Double(frames) * 0.08 / gen
        }
        if Date().timeIntervalSince(last) > 0.3, !player.isPlaying || !playAudio {
            isSpeaking = false
            status = "Ready"
            print(
                "  \(frames) frames, first audio \(String(format: "%.0f", firstAudioMs ?? 0)) ms, "
                    + "\(String(format: "%.2f", msPerFrame ?? 0)) ms/frame, \(String(format: "%.1f", realTime ?? 0))x real time"
            )
        }
    }

    func stop() {
        consumer?.cancel()
        consumer = nil
        if let s = session { Task { await s.cancel() } }
        session = nil
        sessionVoice = nil
        player.stop()
        isSpeaking = false
        if isReady { status = "Ready" }
    }

    /// `--render <dir>`: synthesize the text box offline, feed the audio through a
    /// meter exactly as the player's tap would, and write ring frames as PNGs
    /// (every 120 ms) plus the WAV. For checking the visual without a visible window
    /// (macOS pauses animations of occluded windows) and for README images.
    private func renderFrames(to dir: URL) async {
        guard let manager else { return }
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let samples = try await manager.synthesizeSamples(
                text: text, voice: selectedVoice.id, temperature: nil, seed: 7, deEss: false)
            let meter = SpectrumMeter()
            let block = SpectrumMeter.blockSamples
            let palette: [Color] =
                selectedVoice.isCharacter
                ? [Nord.purple, Nord.frost0, Nord.yellow] : [Nord.frost1, Nord.frost2, Nord.frost3]
            var frame = 0
            var start = 0
            while start + block <= samples.count {
                samples.withUnsafeBufferPointer { buf in
                    if let base = buf.baseAddress { meter.push(base.advanced(by: start), count: block) }
                }
                start += block
                if (start / block) % 6 == 0 {  // every 120 ms of audio
                    let view = RingWaveView(meter: meter, palette: palette)
                        .frame(width: 420, height: 420)
                        .background(Nord.bg)
                    let renderer = ImageRenderer(content: view)
                    renderer.scale = 2
                    if let image = renderer.nsImage, let tiff = image.tiffRepresentation,
                        let rep = NSBitmapImageRep(data: tiff),
                        let png = rep.representation(using: .png, properties: [:])
                    {
                        try png.write(to: dir.appendingPathComponent(String(format: "ring_%03d.png", frame)))
                    }
                    frame += 1
                }
            }
            let wav = try AudioWAV.data(from: samples, sampleRate: 24_000)
            try wav.write(to: dir.appendingPathComponent("speech.wav"))
            print("rendered \(frame) ring frames and speech.wav to \(dir.path)")
        } catch {
            print("render failed: \(error)")
        }
    }
}
