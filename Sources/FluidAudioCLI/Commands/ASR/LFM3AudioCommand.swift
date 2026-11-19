#if os(macOS) && arch(arm64)
import CoreML
import FluidAudio
import Foundation

/// `lfm3-audio <transcribe|live|tts|benchmark> ...` — LFM3-Audio-2B (ASR, LiveASR, TTS) on Core ML or Core AI.
enum LFM3AudioCommand {
    private static let logger = AppLogger(category: "LFM3Audio")

    struct Options {
        var backend: LFM3AudioBackend = .coreML
        var modelDir: String?
        var coreAIModelDir: String?
        var language: String? = "English"
        var output: String?
        var voice: LFM3AudioVoice = .female
        var ttsLanguage = "American English"
        var seed: UInt64?
        var chunkMs = 160
        var subset = "test-clean"
        var maxFiles: Int?
        var autoDownload = false
        var json: String?
        var phrases: String?
        var positional: [String] = []
    }

    static func run(arguments: [String]) async {
        guard let command = arguments.first else {
            printUsage()
            return
        }
        let options = parse(Array(arguments.dropFirst()))
        guard #available(macOS 15, *) else {
            logger.error("LFM3-Audio needs macOS 15 or newer")
            exit(1)
        }
        do {
            switch command {
            case "transcribe": try await transcribe(options)
            case "live": try await live(options)
            case "tts": try await speak(options)
            case "benchmark": try await benchmark(options)
            default: printUsage()
            }
        } catch {
            logger.error("lfm3-audio \(command) failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - Subcommands

    @available(macOS 15, *)
    private static func load(_ options: Options, speech: Bool) async throws -> LFM3AudioManager {
        let start = Date()
        let models: LFM3AudioModels
        if let dir = options.modelDir {
            models = try LFM3AudioModels.load(
                from: URL(fileURLWithPath: dir), backend: options.backend,
                coreAIDirectory: options.coreAIModelDir.map { URL(fileURLWithPath: $0) }, includeSpeech: speech)
        } else {
            models = try await LFM3AudioModels.downloadAndLoad(backend: options.backend, includeSpeech: speech)
        }
        let manager = try await LFM3AudioManager(models: models)
        print(
            String(
                format: "Loaded LFM3-Audio-2B (%@) in %.1f s", options.backend.rawValue, Date().timeIntervalSince(start)
            ))
        return manager
    }

    @available(macOS 15, *)
    private static func transcribe(_ options: Options) async throws {
        guard let path = options.positional.first else { return printUsage() }
        let manager = try await load(options, speech: false)
        let result = try await manager.transcribe(URL(fileURLWithPath: path), language: options.language)
        print(result.text)
        print(
            String(
                format: "%.2f s audio in %.2f s (RTFx %.1f), %d tokens", result.audioDuration, result.processingTime,
                result.rtfx, result.tokenCount))
    }

    /// Streams a file through LiveASR in `--chunk-ms` pieces (as a microphone would deliver it).
    @available(macOS 15, *)
    private static func live(_ options: Options) async throws {
        guard let path = options.positional.first else { return printUsage() }
        let manager = try await load(options, speech: false)
        let samples = try AudioConverter().resampleAudioFile(path: path)
        let piece = max(1, options.chunkMs * 16)
        try await manager.startLiveTranscription()
        let start = Date()
        var shown = ""
        var offset = 0
        while offset < samples.count {
            let end = min(offset + piece, samples.count)
            let text = try await manager.appendLiveAudio(Array(samples[offset..<end]))
            if text != shown {
                print(String(format: "[%6.2f s] %@", Double(end) / 16_000, text))
                shown = text
            }
            offset = end
        }
        let final = try await manager.finishLiveTranscription()
        let elapsed = Date().timeIntervalSince(start)
        print(final)
        print(
            String(
                format: "%.2f s audio in %.2f s (RTFx %.1f)", Double(samples.count) / 16_000, elapsed,
                Double(samples.count) / 16_000 / elapsed))
    }

    /// One phrase -> `--output` wav, or `--phrases <file>` (one per line) -> `<output dir>/phrase_%03d.wav`
    /// (score the directory with `tts-asr-verify --score-only`).
    @available(macOS 15, *)
    private static func speak(_ options: Options) async throws {
        let texts: [String]
        if let file = options.phrases {
            texts = try String(contentsOfFile: file, encoding: .utf8).split(separator: "\n").map(String.init)
                .filter { !$0.isEmpty && !$0.hasPrefix("#") }
        } else if let text = options.positional.first {
            texts = [text]
        } else {
            return printUsage()
        }
        let manager = try await load(options, speech: true)
        let batch = options.phrases != nil
        let output = options.output ?? (batch ? "lfm3_tts" : "lfm3_tts.wav")
        if batch { try FileManager.default.createDirectory(atPath: output, withIntermediateDirectories: true) }
        var audio = 0.0
        var wall = 0.0
        var firstAudio: [Double] = []
        var cpu = 0.0
        for (index, text) in texts.enumerated() {
            let cpuStart = processCPUSeconds()  // synthesis only, not the WAV encoding / writes below
            let result = try await manager.synthesize(
                text, voice: options.voice, language: options.ttsLanguage,
                seed: options.seed.map { $0 &+ UInt64(index) } ?? (batch ? UInt64(index) : nil))
            cpu += processCPUSeconds() - cpuStart
            let path =
                batch
                ? (output as NSString).appendingPathComponent(String(format: "phrase_%03d.wav", index + 1)) : output
            let data = try AudioWAV.data(from: result.samples, sampleRate: Double(result.sampleRate), normalize: false)
            try data.write(to: URL(fileURLWithPath: path))
            audio += result.duration
            wall += result.processingTime
            if let latency = result.firstAudioLatency { firstAudio.append(latency * 1000) }
            print(
                String(
                    format: "%.2f s audio (%d frames%@) in %.2f s (RTFx %.2f), first audio %.0f ms -> %@",
                    result.duration, result.frames, result.finished ? "" : ", truncated", result.processingTime,
                    result.rtfx, (result.firstAudioLatency ?? 0) * 1000, path))
        }
        if batch {
            let sorted = firstAudio.sorted()
            print(
                String(
                    format:
                        "LFM3-Audio TTS (%@): %d phrases, %.1f s audio in %.1f s (RTFx %.2f), "
                        + "first audio median %.0f ms, process CPU %.2f s (%.3f s per audio s)",
                    options.backend.rawValue, texts.count, audio, wall, audio / max(wall, 1e-9),
                    sorted.isEmpty ? 0 : sorted[sorted.count / 2], cpu, cpu / max(audio, 1e-9)))
        }
    }

    /// User + system CPU seconds of this process (all threads).
    private static func processCPUSeconds() -> Double {
        var usage = rusage()
        guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
        func seconds(_ t: timeval) -> Double { Double(t.tv_sec) + Double(t.tv_usec) / 1e6 }
        return seconds(usage.ru_utime) + seconds(usage.ru_stime)
    }

    /// LibriSpeech WER + RTFx (offline ASR, or LiveASR with `--live`).
    @available(macOS 15, *)
    private static func benchmark(_ options: Options) async throws {
        let live = options.positional.contains("--live")
        let downloader = ASRBenchmark()
        if options.autoDownload { try await downloader.downloadLibriSpeech(subset: options.subset) }
        let dataset = downloader.getLibriSpeechDirectory().appendingPathComponent(options.subset)
        var files = try libriSpeechFiles(dataset)
        if let cap = options.maxFiles { files = Array(files.prefix(cap)) }
        guard !files.isEmpty else {
            throw LFM3AudioError.modelNotFound("LibriSpeech \(options.subset) at \(dataset.path) (--auto-download)")
        }
        let manager = try await load(options, speech: false)
        var errors = 0
        var words = 0
        var audioSeconds = 0.0
        var wallSeconds = 0.0
        var rows: [[String: Any]] = []
        for (index, file) in files.enumerated() {
            let samples = try AudioConverter().resampleAudioFile(file.audio)
            let start = Date()
            let text: String
            if live {
                try await manager.startLiveTranscription()
                var offset = 0
                while offset < samples.count {
                    let end = min(offset + options.chunkMs * 16, samples.count)
                    try await manager.appendLiveAudio(Array(samples[offset..<end]))
                    offset = end
                }
                text = try await manager.finishLiveTranscription()
            } else {
                text = try await manager.transcribe(samples, language: options.language).text
            }
            let elapsed = Date().timeIntervalSince(start)
            let metrics = WERCalculator.calculateWERMetrics(hypothesis: text, reference: file.reference)
            errors += metrics.insertions + metrics.deletions + metrics.substitutions
            words += metrics.totalWords
            audioSeconds += Double(samples.count) / 16_000
            wallSeconds += elapsed
            rows.append([
                "file": file.audio.lastPathComponent, "hypothesis": text, "reference": file.reference,
                "wer": metrics.wer, "seconds": elapsed,
            ])
            if (index + 1) % 20 == 0 || index + 1 == files.count {
                print(
                    String(
                        format: "[%d/%d] WER %.2f%%  RTFx %.1f", index + 1, files.count,
                        100 * Double(errors) / Double(max(words, 1)), audioSeconds / wallSeconds))
            }
        }
        let wer = 100 * Double(errors) / Double(max(words, 1))
        print(
            String(
                format: "LFM3-Audio %@ %@ (%@): %d files, WER %.2f%%, RTFx %.1f (%.1f s audio in %.1f s)",
                live ? "LiveASR" : "ASR", options.subset, options.backend.rawValue, files.count, wer,
                audioSeconds / wallSeconds, audioSeconds, wallSeconds))
        if let json = options.json {
            let summary: [String: Any] = [
                "backend": options.backend.rawValue, "mode": live ? "live" : "offline", "subset": options.subset,
                "files": files.count, "wer": wer, "rtfx": audioSeconds / wallSeconds, "results": rows,
            ]
            try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
                .write(to: URL(fileURLWithPath: json))
        }
    }

    // MARK: - Helpers

    private static func libriSpeechFiles(_ directory: URL) throws -> [(audio: URL, reference: String)] {
        var files: [(URL, String)] = []
        let enumerator = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "txt", url.lastPathComponent.contains(".trans.") else { continue }
            for line in try String(contentsOf: url, encoding: .utf8).split(separator: "\n") {
                let parts = line.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { continue }
                let audio = url.deletingLastPathComponent().appendingPathComponent("\(parts[0]).flac")
                if FileManager.default.fileExists(atPath: audio.path) { files.append((audio, String(parts[1]))) }
            }
        }
        return files.sorted { $0.0.lastPathComponent < $1.0.lastPathComponent }
    }

    private static func parse(_ arguments: [String]) -> Options {
        var options = Options()
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            func next() -> String? {
                i += 1
                return i < arguments.count ? arguments[i] : nil
            }
            switch arg {
            case "--backend": options.backend = LFM3AudioBackend(rawValue: next() ?? "") ?? .coreML
            case "--model-dir": options.modelDir = next()
            case "--coreai-model-dir": options.coreAIModelDir = next()
            case "--language":
                let value = next()
                options.language = value == "auto" ? nil : value
                if let value, value != "auto" { options.ttsLanguage = value }
            case "--voice": options.voice = LFM3AudioVoice(rawValue: next() ?? "") ?? .female
            case "--seed": options.seed = next().flatMap { UInt64($0) }
            case "--output", "--out", "-o": options.output = next()
            case "--chunk-ms": options.chunkMs = next().flatMap { Int($0) } ?? 160
            case "--subset": options.subset = next() ?? "test-clean"
            case "--max-files": options.maxFiles = next().flatMap { Int($0) }
            case "--auto-download": options.autoDownload = true
            case "--json": options.json = next()
            case "--phrases": options.phrases = next()
            default: options.positional.append(arg)
            }
            i += 1
        }
        return options
    }

    private static func printUsage() {
        print(
            """
            LFM3-Audio-2B (Liquid AI): ASR, LiveASR and TTS from one audio LM (macOS 15+; Core AI: macOS 27+)

            Usage:
                fluidaudiocli lfm3-audio transcribe <audio> [--language English|auto]
                fluidaudiocli lfm3-audio live <audio> [--chunk-ms 160]
                fluidaudiocli lfm3-audio tts "<text>" [--voice female|male] [--language "American English"]
                                             [--seed N] [--output out.wav]
                fluidaudiocli lfm3-audio tts --phrases phrases.txt --output dir   (phrase_%03d.wav per line)
                fluidaudiocli lfm3-audio benchmark [--live] [--subset test-clean] [--max-files N]
                                                   [--auto-download] [--json out.json]

            Options:
                --backend coreml|coreai   LM runtime (default coreml)
                --model-dir <dir>         local Core ML repo layout (coreml/, assets/) instead of downloading
                --coreai-model-dir <dir>  local Core AI repo layout (coreai/) for --backend coreai
            """)
    }
}
#endif
