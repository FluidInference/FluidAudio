#if os(macOS)
import FluidAudio
import Foundation

/// `fluidaudio tts-benchmark` — quantitative TTS benchmark harness.
///
/// Differs from `tts-asr-verify` (which only measures WER + RTFx) by also
/// reporting **TTFT / cold-start / warm-start latency, per-stage timings,
/// peak RSS, WER + CER per category** and a configurable compute-unit
/// preset (`--compute-units default|all-ane|cpu-and-gpu|cpu-only`).
///
/// Slice 1 only wires the **Kokoro ANE** backend; PocketTTS / StyleTTS2 /
/// Magpie / CosyVoice3 land in follow-ups.
///
/// Usage:
///   fluidaudio tts-benchmark --backend kokoro-ane \
///       --corpus prose-en \
///       --voice af_heart \
///       --compute-units default \
///       --output-json bench.json
///
/// Categories ship in `Benchmarks/tts/corpus/`:
///   prose-en   — 20 Harvard sentences (clean prose)
///   numbers-en — 10 phrases with numbers / dates / currencies
///   names-en   — 10 phrases with proper nouns / acronyms
public enum TtsBenchmarkCommand {

    private static let logger = AppLogger(category: "TtsBenchmarkCommand")

    public static func run(arguments: [String]) async {
        var backendName = "kokoro-ane"
        var corpusName: String?
        var corpusPath: String?
        var voice = TtsConstants.recommendedVoice
        var computeUnitsName = "default"
        var outputJson: String?
        var audioDir: String?
        var skipAsr = false

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--backend":
                if i + 1 < arguments.count {
                    backendName = arguments[i + 1]
                    i += 1
                }
            case "--corpus":
                if i + 1 < arguments.count {
                    corpusName = arguments[i + 1]
                    i += 1
                }
            case "--corpus-path":
                if i + 1 < arguments.count {
                    corpusPath = arguments[i + 1]
                    i += 1
                }
            case "--voice":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            case "--compute-units":
                if i + 1 < arguments.count {
                    computeUnitsName = arguments[i + 1]
                    i += 1
                }
            case "--output-json":
                if i + 1 < arguments.count {
                    outputJson = arguments[i + 1]
                    i += 1
                }
            case "--audio-dir":
                if i + 1 < arguments.count {
                    audioDir = arguments[i + 1]
                    i += 1
                }
            case "--skip-asr":
                skipAsr = true
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arg)")
            }
            i += 1
        }

        // Resolve corpus.
        let phrases: [(category: String, text: String)]
        let corpusLabel: String
        do {
            if let corpusPath {
                let url = resolveURL(corpusPath, isDirectory: false)
                let raw = try String(contentsOf: url, encoding: .utf8)
                phrases = parseCorpus(raw, category: url.deletingPathExtension().lastPathComponent)
                corpusLabel = url.lastPathComponent
            } else {
                let resolved = corpusName ?? "prose-en"
                phrases = try loadShippedCorpus(resolved)
                corpusLabel = resolved
            }
        } catch {
            logger.error("Failed to load corpus: \(error.localizedDescription)")
            exit(1)
        }
        guard !phrases.isEmpty else {
            logger.error("Corpus is empty after parsing")
            exit(1)
        }
        logger.info("Loaded \(phrases.count) phrase(s) from corpus '\(corpusLabel)'")

        guard let preset = TtsComputeUnitPreset(cliValue: computeUnitsName) else {
            logger.error(
                "Unknown --compute-units value: \(computeUnitsName). Expected default | all-ane | cpu-and-gpu | cpu-only."
            )
            exit(1)
        }

        let backend = parseBackend(backendName)
        guard backend == .kokoroAne else {
            logger.error(
                "tts-benchmark slice-1 only supports --backend kokoro-ane (got '\(backendName)')")
            exit(1)
        }

        let resolvedVoice =
            voice == TtsConstants.recommendedVoice ? KokoroAneConstants.defaultVoice : voice

        do {
            try await runKokoroAne(
                phrases: phrases,
                corpusLabel: corpusLabel,
                voice: resolvedVoice,
                preset: preset,
                outputJson: outputJson,
                audioDir: audioDir,
                skipAsr: skipAsr
            )
        } catch {
            logger.error("tts-benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - Kokoro ANE driver

    private static func runKokoroAne(
        phrases: [(category: String, text: String)],
        corpusLabel: String,
        voice: String,
        preset: TtsComputeUnitPreset,
        outputJson: String?,
        audioDir: String?,
        skipAsr: Bool
    ) async throws {
        let units = KokoroAneComputeUnits(preset: preset)
        let manager = KokoroAneManager(
            defaultVoice: voice,
            computeUnits: units
        )

        // -- Cold start = initialize() time (first download + ANE compile).
        let coldStart = Date()
        try await manager.initialize()
        let coldStartS = Date().timeIntervalSince(coldStart)
        logger.info(
            String(format: "Cold start (initialize): %.2fs", coldStartS))

        // -- First synth (post-init, still cold-ish on the synthesis path).
        let firstSynthStart = Date()
        let firstWarmup = try await manager.synthesizeDetailed(
            text: "Initialization warm-up.", voice: voice, speed: 1.0)
        let firstSynthMs = Date().timeIntervalSince(firstSynthStart) * 1000
        let firstWarmupAudioMs =
            Double(firstWarmup.samples.count)
            / Double(firstWarmup.sampleRate) * 1000
        logger.info(
            String(
                format: "First synth: %.0f ms (audio %.0f ms)",
                firstSynthMs, firstWarmupAudioMs))

        // Optional output dir for WAVs.
        var audioDirURL: URL? = nil
        if let audioDir {
            let url = resolveURL(audioDir, isDirectory: true)
            try FileManager.default.createDirectory(
                at: url, withIntermediateDirectories: true)
            audioDirURL = url
        }

        // Optional ASR.
        var asr: AsrManager? = nil
        var decoderLayers = 0
        if !skipAsr {
            let asrModels = try await AsrModels.downloadAndLoad()
            let asrInstance = AsrManager()
            try await asrInstance.loadModels(asrModels)
            decoderLayers = await asrInstance.decoderLayerCount
            asr = asrInstance
        }

        // -- Per-phrase warm runs.
        var perPhrase: [[String: Any]] = []
        var byCategory: [String: [Int]] = [:]  // category → indexes into perPhrase

        for (idx, item) in phrases.enumerated() {
            let label = String(format: "[%02d/%02d]", idx + 1, phrases.count)
            logger.info("\(label) [\(item.category)] \(item.text)")

            let synth0 = Date()
            let result = try await manager.synthesizeDetailed(
                text: item.text, voice: voice, speed: 1.0)
            let synthMs = Date().timeIntervalSince(synth0) * 1000
            let audioMs = Double(result.samples.count) / Double(result.sampleRate) * 1000
            let rtfx = synthMs > 0 ? audioMs / synthMs : 0

            // Persist WAV (audioDir if set, else temp file for ASR step).
            let wavURL: URL
            if let audioDirURL {
                wavURL = audioDirURL.appendingPathComponent(
                    String(format: "phrase_%03d.wav", idx + 1))
            } else {
                wavURL = FileManager.default.temporaryDirectory
                    .appendingPathComponent("tts-benchmark-\(UUID().uuidString).wav")
            }
            let wavData = try AudioWAV.data(
                from: result.samples, sampleRate: Double(result.sampleRate))
            try wavData.write(to: wavURL)

            // ASR roundtrip.
            var werValue = Double.nan
            var cerValue = Double.nan
            var hypothesis = ""
            var asrMs = 0.0
            if let asr {
                let asr0 = Date()
                var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                let transcription = try await asr.transcribe(
                    wavURL, decoderState: &decoderState)
                asrMs = Date().timeIntervalSince(asr0) * 1000
                hypothesis = transcription.text

                let m = WERCalculator.calculateWERAndCER(
                    hypothesis: hypothesis, reference: item.text)
                werValue = m.wer
                cerValue = m.cer
            }

            if audioDirURL == nil {
                try? FileManager.default.removeItem(at: wavURL)
            }

            logger.info(
                String(
                    format:
                        "  ttft=%.0f ms  audio=%.0f ms  rtfx=%.2fx  wer=%.1f%%  cer=%.1f%%",
                    synthMs, audioMs, rtfx,
                    werValue.isNaN ? 0 : werValue * 100,
                    cerValue.isNaN ? 0 : cerValue * 100))

            byCategory[item.category, default: []].append(perPhrase.count)
            perPhrase.append([
                "index": idx + 1,
                "category": item.category,
                "reference": item.text,
                "hypothesis": hypothesis,
                "ttft_ms": synthMs,  // one-shot backend → TTFT == total synth.
                "synth_ms": synthMs,
                "audio_ms": audioMs,
                "rtfx": rtfx,
                "wer": werValue.isNaN ? NSNull() : werValue as Any,
                "cer": cerValue.isNaN ? NSNull() : cerValue as Any,
                "asr_ms": asrMs,
                "encoder_tokens": result.encoderTokens,
                "acoustic_frames": result.acousticFrames,
                "stage_ms": [
                    "albert": result.timings.albert,
                    "post_albert": result.timings.postAlbert,
                    "alignment": result.timings.alignment,
                    "prosody": result.timings.prosody,
                    "noise": result.timings.noise,
                    "vocoder": result.timings.vocoder,
                    "tail": result.timings.tail,
                    "total": result.timings.totalMs,
                ],
                "wav_path": audioDirURL == nil ? "" : wavURL.path,
            ])
        }

        if let asr {
            await asr.cleanup()
        }

        // -- Aggregate.
        let totalSynthMs = perPhrase.reduce(0.0) { $0 + ($1["synth_ms"] as? Double ?? 0) }
        let totalAudioMs = perPhrase.reduce(0.0) { $0 + ($1["audio_ms"] as? Double ?? 0) }
        let aggRtfx = totalSynthMs > 0 ? totalAudioMs / totalSynthMs : 0

        let synthMsValues = perPhrase.compactMap { $0["synth_ms"] as? Double }.sorted()
        let p50 = percentile(synthMsValues, 0.5)
        let p95 = percentile(synthMsValues, 0.95)

        // Per-category aggregates.
        var categories: [[String: Any]] = []
        for (cat, indexes) in byCategory.sorted(by: { $0.key < $1.key }) {
            let werVals = indexes.compactMap { perPhrase[$0]["wer"] as? Double }
            let cerVals = indexes.compactMap { perPhrase[$0]["cer"] as? Double }
            let synthVals = indexes.compactMap { perPhrase[$0]["synth_ms"] as? Double }
            let audioVals = indexes.compactMap { perPhrase[$0]["audio_ms"] as? Double }
            let synthSum = synthVals.reduce(0, +)
            let audioSum = audioVals.reduce(0, +)
            let macroWer =
                werVals.isEmpty ? Double.nan : werVals.reduce(0, +) / Double(werVals.count)
            let macroCer =
                cerVals.isEmpty ? Double.nan : cerVals.reduce(0, +) / Double(cerVals.count)
            categories.append([
                "category": cat,
                "phrase_count": indexes.count,
                "macro_wer": macroWer.isNaN ? NSNull() : macroWer as Any,
                "macro_cer": macroCer.isNaN ? NSNull() : macroCer as Any,
                "synth_ms_p50": percentile(synthVals.sorted(), 0.5),
                "synth_ms_p95": percentile(synthVals.sorted(), 0.95),
                "rtfx": synthSum > 0 ? audioSum / synthSum : 0,
            ])
        }

        let peakRssMb =
            Double(FluidAudioCLI.fetchPeakMemoryUsageBytes() ?? 0) / 1024 / 1024

        // Banner.
        logger.info("--- Summary ---")
        logger.info("  backend:        kokoro-ane")
        logger.info("  voice:          \(voice)")
        logger.info("  corpus:         \(corpusLabel) (n=\(phrases.count))")
        logger.info("  compute units:  \(preset.rawValue)")
        logger.info(String(format: "  cold start:     %.2fs (initialize)", coldStartS))
        logger.info(String(format: "  first synth:    %.0f ms", firstSynthMs))
        logger.info(String(format: "  warm synth p50: %.0f ms", p50))
        logger.info(String(format: "  warm synth p95: %.0f ms", p95))
        logger.info(String(format: "  agg RTFx:       %.2fx", aggRtfx))
        logger.info(String(format: "  peak RSS:       %.0f MB", peakRssMb))
        if !skipAsr {
            let werVals = perPhrase.compactMap { $0["wer"] as? Double }
            let cerVals = perPhrase.compactMap { $0["cer"] as? Double }
            let macroWer =
                werVals.isEmpty ? 0 : werVals.reduce(0, +) / Double(werVals.count)
            let macroCer =
                cerVals.isEmpty ? 0 : cerVals.reduce(0, +) / Double(cerVals.count)
            logger.info(String(format: "  macro WER:      %.2f%%", macroWer * 100))
            logger.info(String(format: "  macro CER:      %.2f%%", macroCer * 100))
        } else {
            logger.info("  WER/CER:        skipped (--skip-asr)")
        }

        // -- JSON.
        if let outputJson {
            let summary: [String: Any] = [
                "backend": "kokoro-ane",
                "voice": voice,
                "corpus": corpusLabel,
                "phrase_count": phrases.count,
                "compute_units": preset.rawValue,
                "cold_start_s": coldStartS,
                "first_synth_ms": firstSynthMs,
                "warm_synth_ms_p50": p50,
                "warm_synth_ms_p95": p95,
                "agg_rtfx": aggRtfx,
                "peak_rss_mb": peakRssMb,
                "asr_skipped": skipAsr,
            ]
            let report: [String: Any] = [
                "summary": summary,
                "categories": categories,
                "phrases": perPhrase,
            ]
            let url = resolveURL(outputJson, isDirectory: false)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let data = try JSONSerialization.data(
                withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: url)
            logger.info("Report written: \(url.path)")
        }
    }

    // MARK: - Corpus loading

    private static func loadShippedCorpus(
        _ name: String
    ) throws -> [(category: String, text: String)] {
        // Look for `Benchmarks/tts/corpus/<name>.txt` relative to the
        // working directory (the convention is to invoke `swift run` from
        // the package root).
        let cwd = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        let url = cwd.appendingPathComponent(
            "Benchmarks/tts/corpus/\(name).txt", isDirectory: false)
        let raw = try String(contentsOf: url, encoding: .utf8)
        return parseCorpus(raw, category: name)
    }

    private static func parseCorpus(
        _ raw: String, category: String
    ) -> [(category: String, text: String)] {
        return
            raw
            .split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && !$0.hasPrefix("#") }
            .map { (category: category, text: $0) }
    }

    // MARK: - Helpers

    private static func percentile(_ sorted: [Double], _ p: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        let idx = Int((Double(sorted.count - 1) * p).rounded())
        return sorted[max(0, min(sorted.count - 1, idx))]
    }

    private static func parseBackend(_ name: String) -> TtsBackend {
        switch name.lowercased() {
        case "kokoro": return .kokoro
        case "pocket", "pockettts", "pocket-tts": return .pocketTts
        case "kokoro-ane", "kokoroane", "lai": return .kokoroAne
        default: return .kokoroAne
        }
    }

    private static func resolveURL(_ path: String, isDirectory: Bool) -> URL {
        let expanded = (path as NSString).expandingTildeInPath
        if expanded.hasPrefix("/") {
            return URL(fileURLWithPath: expanded, isDirectory: isDirectory)
        }
        let cwd = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return cwd.appendingPathComponent(expanded, isDirectory: isDirectory)
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio tts-benchmark [options]

            Quantitative TTS benchmark — TTFT, cold/warm split, per-stage timings,
            peak RSS, WER + CER per category, configurable compute-unit preset.

            Slice 1 only wires the `kokoro-ane` backend; PocketTTS / StyleTTS2 /
            Magpie / CosyVoice3 land in follow-up PRs.

            Options:
              --backend <name>          TTS backend (kokoro-ane, default)
              --corpus <name>           Shipped corpus name: prose-en | numbers-en
                                        | names-en (default: prose-en)
              --corpus-path <path>      Custom corpus file (overrides --corpus)
              --voice <name>            Voice name (default: af_heart)
              --compute-units <preset>  default | all-ane | cpu-and-gpu | cpu-only
              --output-json <path>      Write JSON report
              --audio-dir <path>        Keep generated WAVs under this dir
              --skip-asr                Skip Parakeet roundtrip (no WER/CER)
              --help, -h                Show this help

            Examples:
              fluidaudio tts-benchmark --corpus prose-en --output-json bench.json
              fluidaudio tts-benchmark --corpus numbers-en --compute-units cpu-and-gpu
              fluidaudio tts-benchmark --corpus-path my-phrases.txt --skip-asr
            """
        )
    }
}
#endif
