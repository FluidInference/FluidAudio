#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// CLI surface for the Magpie TTS Multilingual Swift port.
///
/// Subcommands:
///   - `download`             Fetch models + constants + tokenizer data from HuggingFace.
///   - `text`                 Synthesize text → WAV.
///   - `bench`                Multi-shot in-process synthesis benchmark.
public enum MagpieCommand {

    private static let logger = AppLogger(category: "MagpieCommand")

    public static func run(arguments: [String]) async {
        guard let sub = arguments.first else {
            printUsage()
            return
        }
        let rest = Array(arguments.dropFirst())
        switch sub {
        case "download":
            await runDownload(arguments: rest)
        case "text":
            await runText(arguments: rest)
        case "bench":
            await runBench(arguments: rest)
        case "help", "--help", "-h":
            printUsage()
        default:
            logger.error("Unknown magpie subcommand: \(sub)")
            printUsage()
            exit(1)
        }
    }

    // MARK: - download

    private static func runDownload(arguments: [String]) async {
        var languageCodes: [String] = ["en"]
        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            if arg == "--languages" || arg == "-l", i + 1 < arguments.count {
                languageCodes = arguments[i + 1].split(separator: ",").map(String.init)
                i += 1
            }
            i += 1
        }
        let langs: Set<MagpieLanguage> = Set(languageCodes.compactMap { MagpieLanguage(rawValue: $0) })
        if langs.isEmpty {
            logger.error("No valid language codes provided")
            exit(1)
        }
        do {
            let repoDir = try await MagpieResourceDownloader.ensureAssets(languages: langs)
            logger.info("Magpie assets ready at: \(repoDir.path)")
        } catch {
            logger.error("Magpie download failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - text

    private static func runText(arguments: [String]) async {
        var text: String? = nil
        var output = "magpie.wav"
        var speakerIdx = MagpieSpeaker.john.rawValue
        var languageCode = "en"
        var cfg: Float = MagpieConstants.defaultCfgScale
        var topK = MagpieConstants.defaultTopK
        var temperature = MagpieConstants.defaultTemperature
        var seed: UInt64? = nil
        var allowIpa = true
        var streaming = false

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--output", "-o":
                if i + 1 < arguments.count {
                    output = arguments[i + 1]
                    i += 1
                }
            case "--speaker":
                if i + 1 < arguments.count, let idx = Int(arguments[i + 1]) {
                    speakerIdx = idx
                    i += 1
                }
            case "--language", "-L":
                if i + 1 < arguments.count {
                    languageCode = arguments[i + 1]
                    i += 1
                }
            case "--cfg":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    cfg = v
                    i += 1
                }
            case "--topk":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    topK = v
                    i += 1
                }
            case "--temperature":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    temperature = v
                    i += 1
                }
            case "--seed":
                if i + 1 < arguments.count, let v = UInt64(arguments[i + 1]) {
                    seed = v
                    i += 1
                }
            case "--no-ipa-override":
                allowIpa = false
            case "--stream":
                streaming = true
            case "--text":
                if i + 1 < arguments.count {
                    text = arguments[i + 1]
                    i += 1
                }
            default:
                if text == nil { text = arg }
            }
            i += 1
        }

        guard let text = text, !text.isEmpty else {
            logger.error("Missing text argument")
            printUsage()
            exit(1)
        }
        guard let speaker = MagpieSpeaker(rawValue: speakerIdx) else {
            logger.error("Invalid speaker index \(speakerIdx); valid range 0..<\(MagpieConstants.numSpeakers)")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language code '\(languageCode)'")
            exit(1)
        }

        do {
            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            let opts = MagpieSynthesisOptions(
                temperature: temperature,
                topK: topK,
                maxSteps: MagpieConstants.maxSteps,
                minFrames: MagpieConstants.minFrames,
                cfgScale: cfg,
                seed: seed,
                peakNormalize: true,
                allowIpaOverride: allowIpa)
            let outURL = URL(fileURLWithPath: output)
            try FileManager.default.createDirectory(
                at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)

            if streaming {
                try await runStreaming(
                    manager: manager, text: text, speaker: speaker,
                    language: language, options: opts, outURL: outURL)
            } else {
                let start = Date()
                let result = try await manager.synthesize(
                    text: text, speaker: speaker, language: language, options: opts)
                let elapsed = Date().timeIntervalSince(start)

                let wav = try AudioWAV.data(
                    from: result.samples,
                    sampleRate: Double(result.sampleRate))
                try wav.write(to: outURL)

                let audioSecs = result.durationSeconds
                let rtfx = elapsed > 0 ? audioSecs / elapsed : 0
                let t = result.timings
                let stepCount = result.codeCount > 0 ? result.codeCount : 1
                let perStepDecoderMs = t.decoderStepSeconds * 1000.0 / Double(stepCount)
                let perStepSamplerMs = t.samplerSeconds * 1000.0 / Double(stepCount)
                let lines = [
                    "Magpie synthesis complete",
                    "  Speaker: \(speaker.displayName), Language: \(language.rawValue)",
                    "  Codes: \(result.codeCount), EOS: \(result.finishedOnEos)",
                    "  Audio: \(String(format: "%.3f", audioSecs))s, "
                        + "Synthesis: \(String(format: "%.3f", elapsed))s, "
                        + "RTFx: \(String(format: "%.2f", rtfx))x",
                    "  Stages:",
                    "    text_encoder: \(String(format: "%.0f", t.textEncoderSeconds * 1000))ms",
                    "    prefill:      \(String(format: "%.0f", t.prefillSeconds * 1000))ms",
                    "    AR loop:      \(String(format: "%.2f", t.arLoopSeconds))s "
                        + "(decoder=\(String(format: "%.2f", t.decoderStepSeconds))s "
                        + "@ \(String(format: "%.1f", perStepDecoderMs))ms/step, "
                        + "sampler=\(String(format: "%.2f", t.samplerSeconds))s "
                        + "@ \(String(format: "%.1f", perStepSamplerMs))ms/step)",
                    "    nanocodec:    \(String(format: "%.0f", t.nanocodecSeconds * 1000))ms",
                    "  Output: \(outURL.path)",
                ]
                FileHandle.standardError.write(Data((lines.joined(separator: "\n") + "\n").utf8))
            }
        } catch {
            logger.error("Magpie synthesis failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    /// Streaming mode: consume `synthesizeStream` chunk-by-chunk, log
    /// time-to-first-audio + per-chunk arrival times, then write the
    /// concatenated waveform to `outURL` so the produced audio is comparable
    /// to the offline path.
    private static func runStreaming(
        manager: MagpieTtsManager,
        text: String,
        speaker: MagpieSpeaker,
        language: MagpieLanguage,
        options: MagpieSynthesisOptions,
        outURL: URL
    ) async throws {
        FileHandle.standardError.write(
            Data("Magpie streaming synthesis (chunk-level)\n".utf8))
        FileHandle.standardError.write(
            Data(
                "  Speaker: \(speaker.displayName), Language: \(language.rawValue)\n"
                    .utf8))

        let stream = try await manager.synthesizeStream(
            text: text, speaker: speaker, language: language, options: options)

        let start = Date()
        var combined: [Float] = []
        var ttfa: Double? = nil
        var chunkCount = 0
        var totalCodes = 0
        var sampleRate = MagpieConstants.audioSampleRate

        for try await chunk in stream {
            let now = Date().timeIntervalSince(start)
            if ttfa == nil {
                ttfa = now
                let ttfaLine =
                    "  TTFA: \(String(format: "%.3f", now))s "
                    + "(first chunk of \(chunk.codeCount) codes "
                    + "= \(String(format: "%.2f", chunk.durationSeconds))s audio)\n"
                FileHandle.standardError.write(Data(ttfaLine.utf8))
            }
            chunkCount += 1
            totalCodes += chunk.codeCount
            sampleRate = chunk.sampleRate
            let preview =
                chunk.text.count > 60
                ? String(chunk.text.prefix(57)) + "..." : chunk.text
            let line =
                "    [chunk \(chunk.sequenceIndex)] +\(String(format: "%.3f", now))s "
                + "audio=\(String(format: "%.2f", chunk.durationSeconds))s "
                + "codes=\(chunk.codeCount) "
                + "eos=\(chunk.finishedOnEos) "
                + "final=\(chunk.isFinal) "
                + "\"\(preview)\"\n"
            FileHandle.standardError.write(Data(line.utf8))
            combined.append(contentsOf: chunk.samples)
        }
        let elapsed = Date().timeIntervalSince(start)

        // Optional peak-normalize once we have the full buffer (matches the
        // offline default).
        if options.peakNormalize {
            var peak: Float = 0
            for s in combined where abs(s) > peak { peak = abs(s) }
            if peak > 0 {
                let scale = MagpieConstants.peakTarget / peak
                for i in 0..<combined.count { combined[i] *= scale }
            }
        }

        let wav = try AudioWAV.data(
            from: combined, sampleRate: Double(sampleRate))
        try wav.write(to: outURL)

        let audioSecs = Double(combined.count) / Double(sampleRate)
        let rtfx = elapsed > 0 ? audioSecs / elapsed : 0
        let summary = [
            "  Chunks: \(chunkCount), Codes: \(totalCodes)",
            "  TTFA: \(String(format: "%.3f", ttfa ?? 0))s "
                + "(\(String(format: "%.0f", (ttfa ?? 0) * 1000))ms)",
            "  Audio: \(String(format: "%.3f", audioSecs))s, "
                + "Total synthesis: \(String(format: "%.3f", elapsed))s, "
                + "RTFx: \(String(format: "%.2f", rtfx))x",
            "  Output: \(outURL.path)",
        ]
        FileHandle.standardError.write(Data((summary.joined(separator: "\n") + "\n").utf8))
    }

    // MARK: - bench

    /// Multi-shot in-process synthesis bench. Loads the manager once, then runs
    /// `--runs N` synthesize() calls back-to-back on the same actor and reports
    /// per-run + median + min/max RTFx and per-stage statistics. This bypasses
    /// the launch-to-launch Metal scheduler variance you get from invoking
    /// `magpie text` in a loop from the shell.
    private static func runBench(arguments: [String]) async {
        var text =
            "Hello world. This is a test of the Magpie text to speech system, "
            + "running on Apple Silicon with the Swift port."
        var speakerIdx = MagpieSpeaker.john.rawValue
        var languageCode = "en"
        var runs = 5
        var warmup = 1
        var seed: UInt64? = 42

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--text":
                if i + 1 < arguments.count {
                    text = arguments[i + 1]
                    i += 1
                }
            case "--runs":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]), v > 0 {
                    runs = v
                    i += 1
                }
            case "--warmup":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]), v >= 0 {
                    warmup = v
                    i += 1
                }
            case "--speaker":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    speakerIdx = v
                    i += 1
                }
            case "--language", "-L":
                if i + 1 < arguments.count {
                    languageCode = arguments[i + 1]
                    i += 1
                }
            case "--seed":
                if i + 1 < arguments.count, let v = UInt64(arguments[i + 1]) {
                    seed = v
                    i += 1
                }
            case "--no-seed":
                seed = nil
            default:
                break
            }
            i += 1
        }

        guard let speaker = MagpieSpeaker(rawValue: speakerIdx) else {
            logger.error("Invalid speaker index \(speakerIdx)")
            exit(1)
        }
        guard let language = MagpieLanguage(rawValue: languageCode) else {
            logger.error("Invalid language code '\(languageCode)'")
            exit(1)
        }

        do {
            let loadStart = Date()
            let manager = try await MagpieTtsManager.downloadAndCreate(languages: [language])
            let loadElapsed = Date().timeIntervalSince(loadStart)

            let opts = MagpieSynthesisOptions(
                seed: seed,
                peakNormalize: true,
                allowIpaOverride: true)

            var header = [
                "Magpie bench",
                "  Text: \"\(text)\" (\(text.count) chars)",
                "  Speaker: \(speaker.displayName), Language: \(language.rawValue)",
                "  Seed: \(seed.map { String($0) } ?? "random")",
                "  Manager load: \(String(format: "%.2f", loadElapsed))s",
                "  Warmup runs: \(warmup), Measured runs: \(runs)",
            ]
            if warmup > 0 { header.append("") }
            FileHandle.standardError.write(Data((header.joined(separator: "\n") + "\n").utf8))

            for w in 0..<warmup {
                let r = try await manager.synthesize(
                    text: text, speaker: speaker, language: language, options: opts)
                let line =
                    "  [warmup \(w + 1)/\(warmup)] codes=\(r.codeCount) "
                    + "decoder=\(String(format: "%.2f", r.timings.decoderStepSeconds))s "
                    + "nano=\(String(format: "%.2f", r.timings.nanocodecSeconds))s"
                FileHandle.standardError.write(Data((line + "\n").utf8))
            }

            // Measured runs.
            var rtfxs: [Double] = []
            var totals: [Double] = []
            var encoders: [Double] = []
            var prefills: [Double] = []
            var arLoops: [Double] = []
            var decoders: [Double] = []
            var samplers: [Double] = []
            var nanocodecs: [Double] = []
            var perStepDecoderMs: [Double] = []
            var codeCounts: [Int] = []

            FileHandle.standardError.write(Data("\n  per-run results:\n".utf8))
            for run in 0..<runs {
                let start = Date()
                let r = try await manager.synthesize(
                    text: text, speaker: speaker, language: language, options: opts)
                let elapsed = Date().timeIntervalSince(start)
                let audio = r.durationSeconds
                let rtfx = elapsed > 0 ? audio / elapsed : 0
                let steps = max(r.codeCount, 1)
                let perStepMs = r.timings.decoderStepSeconds * 1000.0 / Double(steps)

                rtfxs.append(rtfx)
                totals.append(elapsed)
                encoders.append(r.timings.textEncoderSeconds)
                prefills.append(r.timings.prefillSeconds)
                arLoops.append(r.timings.arLoopSeconds)
                decoders.append(r.timings.decoderStepSeconds)
                samplers.append(r.timings.samplerSeconds)
                nanocodecs.append(r.timings.nanocodecSeconds)
                perStepDecoderMs.append(perStepMs)
                codeCounts.append(r.codeCount)

                let line =
                    "    [\(run + 1)/\(runs)] "
                    + "RTFx=\(String(format: "%.2f", rtfx))x "
                    + "synth=\(String(format: "%.2f", elapsed))s "
                    + "audio=\(String(format: "%.2f", audio))s "
                    + "codes=\(r.codeCount) "
                    + "decoder=\(String(format: "%.2f", r.timings.decoderStepSeconds))s "
                    + "(\(String(format: "%.1f", perStepMs))ms/step) "
                    + "nano=\(String(format: "%.2f", r.timings.nanocodecSeconds))s"
                FileHandle.standardError.write(Data((line + "\n").utf8))
            }

            // Summary stats.
            func stats(_ xs: [Double]) -> (median: Double, min: Double, max: Double, mean: Double) {
                let s = xs.sorted()
                let median = s.isEmpty ? 0 : s[s.count / 2]
                let mean = xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count)
                return (median, s.first ?? 0, s.last ?? 0, mean)
            }
            let rs = stats(rtfxs)
            let ts = stats(totals)
            let ds = stats(decoders)
            let ns = stats(nanocodecs)
            let ps = stats(perStepDecoderMs)

            let summary = [
                "",
                "  summary (n=\(runs)):",
                "    RTFx          median=\(String(format: "%.2f", rs.median))x  "
                    + "min=\(String(format: "%.2f", rs.min))x  "
                    + "max=\(String(format: "%.2f", rs.max))x  "
                    + "mean=\(String(format: "%.2f", rs.mean))x",
                "    synth         median=\(String(format: "%.2f", ts.median))s  "
                    + "min=\(String(format: "%.2f", ts.min))s  "
                    + "max=\(String(format: "%.2f", ts.max))s",
                "    decoder_step  median=\(String(format: "%.2f", ds.median))s  "
                    + "min=\(String(format: "%.2f", ds.min))s  "
                    + "max=\(String(format: "%.2f", ds.max))s  "
                    + "(\(String(format: "%.1f", ps.median))ms/step median)",
                "    nanocodec     median=\(String(format: "%.2f", ns.median))s  "
                    + "min=\(String(format: "%.2f", ns.min))s  "
                    + "max=\(String(format: "%.2f", ns.max))s",
            ]
            FileHandle.standardError.write(Data((summary.joined(separator: "\n") + "\n").utf8))
        } catch {
            logger.error("Magpie bench failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    // MARK: - usage

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudio magpie <subcommand> [options]

            ⚠️  EXPERIMENTAL — quite slow on Apple Silicon, needs further perf work.
                Cold first synth ~30 s (model load + ANE compile). Warm synth ~96 s
                wall for an 8-word English sentence on M-series (RTFx ≈ 0.04, i.e.
                ~25× slower than realtime). For real-time use prefer
                `fluidaudio tts` (Kokoro, ~20× RTFx) or PocketTTS (~1.5–2× RTFx).
                See Documentation/TTS/Magpie.md.

            Subcommands:
              download                Download Magpie models + constants + tokenizers
                --languages en,es,de    Comma-separated language codes (default: en)

              text "<text>"           Synthesize text and write a WAV file
                --output, -o PATH       Output WAV path (default: magpie.wav)
                --speaker N             Speaker index 0-4 (default: 0 = John)
                --language CODE         Language code (en, es, de, fr, it, vi, zh, hi)
                --cfg FLOAT             CFG guidance scale (default: 1.0 = off)
                --topk N                Top-K sampling (default: 80)
                --temperature FLOAT     Sampling temperature (default: 0.6)
                --seed N                Deterministic RNG seed
                --no-ipa-override       Disable `|…|` IPA pass-through

              bench                   In-process multi-shot synthesis benchmark
                --runs N                Measured runs (default: 5)
                --warmup N              Unmeasured warmup runs (default: 1)
                --text "<text>"         Override the bench text
                --speaker N             Speaker index (default: 0)
                --language CODE         Language (default: en)
                --seed N                Deterministic seed (default: 42)
                --no-seed               Use a random seed each run

            IPA override example:
              fluidaudio magpie text "Hello | ˈ n ɛ m o ʊ | Text." --output demo.wav

            """
        )
    }
}
#endif
