import AVFoundation
import FluidAudio
import Foundation
import AppKit
import SwiftUI

/// A processed clip ready for A/B listening.
struct Clip: Identifiable {
    let id = UUID()
    let label: String
    let samples: [Float]
    var seconds: Double { Double(samples.count) / AudioFiles.sampleRate }
}

@MainActor
final class DemoModel: ObservableObject {
    // Model
    @Published var variant: LocalVqeVariant = .v13
    @Published var chunk: LocalVqeChunk = .batch256ms
    @Published private(set) var manager: LocalVqeManager?
    @Published private(set) var loadStatus = "Not loaded"
    @Published private(set) var isLoading = false

    // File mode
    @Published var micURL: URL?
    @Published var referenceURL: URL?
    @Published private(set) var fileClips: [Clip] = []
    @Published private(set) var fileStatus = "Pick a mic recording and the far-end (loudspeaker) signal."
    @Published private(set) var isProcessing = false

    // Live mode
    @Published var farEndURL: URL?
    @Published private(set) var isLive = false
    @Published private(set) var liveLevels = LiveLevels()
    @Published private(set) var liveClips: [Clip] = []
    @Published private(set) var liveStatus = "Plays the far-end file through your speakers while capturing the mic."

    @Published var errorMessage: String?

    // Transcription (Parakeet TDT v3, same ASR as `enhance-benchmark`)
    @Published private(set) var transcripts: [UUID: String] = [:]
    @Published private(set) var isTranscribing = false
    @Published private(set) var asrStatus = ""
    private var asr: AsrManager?
    /// Folder the most recent Enhance / live Stop wrote its WAVs to.
    @Published private(set) var lastOutputDirectory: URL?

    /// Every run writes mic / far-end / enhanced WAVs here, one subfolder per run.
    static let outputRoot = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Downloads/LocalVQEDemo", isDirectory: true)

    let player = Player()
    private var live: LiveCapture?

    /// The pinned benchmark dataset, if `enhance-benchmark` has downloaded it.
    static let sampleDirectory = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/FluidAudio/Datasets/aec-synthetic-mini")

    /// Sample clips from the pinned set, chosen from the 200-file benchmark rows.
    static let samples: [(id: String, note: String)] = [
        ("1148", "clear win: 27 near-end words, 0% → 100% recall, 9 → 0 leaked far-end words"),
        ("102", "clear win: 13 near-end words, 0% → 100% recall, 13 → 0 leaked"),
        ("1089", "clear win: 14 near-end words, 0% → 100% recall, 13 → 0 leaked"),
        ("0", "hard case: −7 dB SER, nonlinear echo; near-end speech is lost (0% recall in the benchmark)"),
    ]

    var hasSamples: Bool {
        FileManager.default.fileExists(
            atPath: Self.sampleDirectory.appendingPathComponent("fileid_\(Self.samples[0].id)_mic.wav").path)
    }

    // MARK: - Model

    func loadModel() {
        guard !isLoading else { return }
        isLoading = true
        loadStatus = "Loading \(variant.rawValue) / \(chunk.rawValue)…"
        let config = LocalVqeConfig(variant: variant, chunk: chunk)
        Task {
            do {
                let started = ContinuousClock.now
                let loaded = try await LocalVqeManager(config: config) { [weak self] progress in
                    Task { @MainActor in
                        self?.loadStatus = String(format: "Downloading… %.0f%%", progress.fractionCompleted * 100)
                    }
                }
                let ms = started.duration(to: .now).components
                manager = loaded
                loadStatus = String(
                    format: "%@ / %@ loaded in %.1f s (CPU)", variant.rawValue, chunk.rawValue,
                    Double(ms.seconds) + Double(ms.attoseconds) / 1e18)
            } catch {
                loadStatus = "Load failed"
                errorMessage = error.localizedDescription
            }
            isLoading = false
        }
    }

    // MARK: - File mode

    func useSamplePair(id: String = samples[0].id) {
        micURL = Self.sampleDirectory.appendingPathComponent("fileid_\(id)_mic.wav")
        referenceURL = Self.sampleDirectory.appendingPathComponent("fileid_\(id)_lpb.wav")
        let note = Self.samples.first { $0.id == id }?.note ?? ""
        fileStatus = "AEC-Challenge synthetic set, fileid \(id) — \(note)."
        fileClips = []
    }

    func processFiles() {
        guard let manager else {
            errorMessage = "Load a model first."
            return
        }
        guard let micURL else {
            errorMessage = "Pick a mic recording."
            return
        }
        isProcessing = true
        fileStatus = "Processing…"
        player.stop()
        let referenceURL = referenceURL
        Task {
            do {
                let mic = try AudioFiles.read(micURL)
                let started = ContinuousClock.now
                let enhanced = try await manager.process(micURL: micURL, referenceURL: referenceURL)
                let elapsed = started.duration(to: .now).components
                let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                var clips = [Clip(label: "Mic (unprocessed)", samples: mic)]
                if let referenceURL {
                    clips.append(Clip(label: "Far end (loudspeaker)", samples: try AudioFiles.read(referenceURL)))
                }
                clips.append(Clip(label: "Enhanced", samples: enhanced))
                fileClips = clips
                let stem = micURL.deletingPathExtension().lastPathComponent
                let folder = try save(clips, run: "files-\(stem)")
                let audioSeconds = Double(mic.count) / AudioFiles.sampleRate
                fileStatus = String(
                    format:
                        "%.1f s enhanced in %.2f s (%.0fx real time); level %.1f dB → %.1f dB. WAVs in %@",
                    audioSeconds, seconds, audioSeconds / max(seconds, 1e-6),
                    AudioFiles.dbfs(AudioFiles.rms(mic[...])), AudioFiles.dbfs(AudioFiles.rms(enhanced[...])),
                    folder.path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
            } catch {
                fileStatus = "Processing failed"
                errorMessage = error.localizedDescription
            }
            isProcessing = false
        }
    }

    // MARK: - Live mode

    func useSampleFarEnd() {
        farEndURL = Self.sampleDirectory.appendingPathComponent("fileid_\(Self.samples[0].id)_lpb.wav")
    }

    func startLive() {
        guard let manager else {
            errorMessage = "Load a model first."
            return
        }
        guard let farEndURL else {
            errorMessage = "Pick a far-end file to play."
            return
        }
        player.stop()
        liveClips = []
        liveLevels = LiveLevels()
        Task {
            guard await LiveCapture.requestMicrophoneAccess() else {
                errorMessage = "Microphone access denied. Grant it to the terminal that launched this app."
                return
            }
            do {
                let file = try AVAudioFile(forReading: farEndURL)
                let stream = try await manager.makeStream()
                let capture = LiveCapture()
                live = capture
                try capture.start(farEnd: file, stream: stream, stepSeconds: 0.256) { [weak self] levels in
                    self?.liveLevels = levels
                }
                isLive = true
                liveStatus = "Live. Speak over the playback; stop when the far-end clip ends."
            } catch {
                live = nil
                errorMessage = error.localizedDescription
            }
        }
    }

    func stopLive() {
        guard let capture = live else { return }
        live = nil
        isLive = false
        liveStatus = "Finishing…"
        Task {
            do {
                let recording = try await capture.stop()
                liveClips = [
                    Clip(label: "Mic (unprocessed)", samples: recording.mic),
                    Clip(label: "Far end (played)", samples: recording.reference),
                    Clip(label: "Enhanced", samples: recording.enhanced),
                ]
                let folder = try save(liveClips, run: "live")
                liveStatus = String(
                    format: "Captured %.1f s. Listen to Mic vs Enhanced. WAVs in %@",
                    Double(recording.mic.count) / AudioFiles.sampleRate,
                    folder.path.replacingOccurrences(of: NSHomeDirectory(), with: "~"))
            } catch {
                liveStatus = "Capture failed"
                errorMessage = error.localizedDescription
            }
        }
    }

    // MARK: - Transcription

    /// Transcribes every clip with Parakeet so the echo, the near-end words and
    /// what survives enhancement can be compared as text.
    func transcribeAll(_ clips: [Clip]) {
        guard !isTranscribing, !clips.isEmpty else { return }
        isTranscribing = true
        Task {
            do {
                if asr == nil {
                    asrStatus = "Loading Parakeet TDT v3…"
                    let manager = AsrManager()
                    try await manager.loadModels(
                        try await AsrModels.downloadAndLoad(version: .v3) { [weak self] progress in
                            Task { @MainActor in
                                self?.asrStatus = String(
                                    format: "Downloading Parakeet… %.0f%%", progress.fractionCompleted * 100)
                            }
                        })
                    asr = manager
                }
                guard let asr else { return }
                let started = ContinuousClock.now
                for clip in clips {
                    asrStatus = "Transcribing \(clip.label)…"
                    var state = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
                    let text = try await asr.transcribe(clip.samples, decoderState: &state).text
                    transcripts[clip.id] = text.isEmpty ? "(no speech recognised)" : text
                }
                let elapsed = started.duration(to: .now).components
                asrStatus = String(
                    format: "Transcribed %d clips in %.1f s", clips.count,
                    Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18)
            } catch {
                asrStatus = "Transcription failed"
                errorMessage = error.localizedDescription
            }
            isTranscribing = false
        }
    }

    // MARK: - Shared

    func play(_ clip: Clip) {
        do {
            try player.play(clip.samples, label: clip.label)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Mic input followed by the enhanced output, for a before/after listen.
    func playBeforeAfter(_ clips: [Clip]) {
        let ordered = clips.filter { $0.label.hasPrefix("Mic") } + clips.filter { $0.label.hasPrefix("Enhanced") }
        do {
            try player.playSequence(ordered.map { ($0.samples, $0.label) })
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Writes the clips as numbered WAVs into a new timestamped run folder and returns it.
    @discardableResult
    func save(_ clips: [Clip], run: String) throws -> URL {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        let directory = Self.outputRoot.appendingPathComponent(
            "\(run)-\(variant.rawValue)-\(formatter.string(from: Date()))", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (index, clip) in clips.enumerated() {
            let name = clip.label.lowercased()
                .replacingOccurrences(of: #"[^a-z0-9]+"#, with: "-", options: .regularExpression)
                .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
            try AudioFiles.write(clip.samples, to: directory.appendingPathComponent("\(index + 1)-\(name).wav"))
        }
        lastOutputDirectory = directory
        return directory
    }

    func revealOutput() {
        guard let lastOutputDirectory else { return }
        NSWorkspace.shared.activateFileViewerSelecting([lastOutputDirectory])
    }
}
