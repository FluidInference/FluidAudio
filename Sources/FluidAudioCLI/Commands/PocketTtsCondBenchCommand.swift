import FluidAudio
import Foundation

/// Benchmark `cond_step` prefill latency for two PocketTTS dispatch modes
/// (legacy chunk-1 vs hybrid chunk-N + chunk-1) on the same text and voice.
///
/// Run via:
/// ```
/// swift run -c release fluidaudio pocket-tts-cond-bench \
///     --text "Hello world." --voice alba --iters 30 --warmup 3
/// ```
///
/// Requires the chunk-N model file (`cond_step_chunk16.mlmodelc`) to be
/// placed manually under `<pocketTtsCacheDir>/v2/<lang>/` — the file is
/// not yet published on HuggingFace. See
/// `PocketTtsCondStepMode.chunked` for placement details.
public enum PocketTtsCondBenchCommand {

    private static let logger = AppLogger(category: "PocketTtsCondBench")

    public static func run(arguments: [String]) async {
        var text = "Hello world, this is a test of the pocket TTS system."
        var voice = "alba"
        var languageRaw = "english"
        var iters = 30
        var warmup = 3
        var chunk = 16
        var alsoSynth = false

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--text":
                if i + 1 < arguments.count {
                    text = arguments[i + 1]
                    i += 1
                }
            case "--voice":
                if i + 1 < arguments.count {
                    voice = arguments[i + 1]
                    i += 1
                }
            case "--language":
                if i + 1 < arguments.count {
                    languageRaw = arguments[i + 1]
                    i += 1
                }
            case "--iters":
                if i + 1 < arguments.count, let n = Int(arguments[i + 1]), n > 0 {
                    iters = n
                    i += 1
                }
            case "--warmup":
                if i + 1 < arguments.count, let n = Int(arguments[i + 1]), n >= 0 {
                    warmup = n
                    i += 1
                }
            case "--chunk":
                if i + 1 < arguments.count, let n = Int(arguments[i + 1]), n > 0 {
                    chunk = n
                    i += 1
                }
            case "--also-synth":
                alsoSynth = true
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arg)")
            }
            i += 1
        }

        guard let language = PocketTtsLanguage(rawValue: languageRaw) else {
            let supported = PocketTtsLanguage.allCases.map { $0.rawValue }.joined(separator: ", ")
            logger.error("Unknown language '\(languageRaw)'. Supported: \(supported)")
            return
        }

        do {
            try await runBenchmark(
                text: text,
                voice: voice,
                language: language,
                iters: iters,
                warmup: warmup,
                chunk: chunk,
                alsoSynth: alsoSynth
            )
        } catch {
            logger.error("pocket-tts-cond-bench failed: \(error)")
        }
    }

    private static func printUsage() {
        let usage = """
            Usage: fluidaudio pocket-tts-cond-bench [options]

            Options:
              --text <string>     Text to prefill (default: short test sentence)
              --voice <name>      Voice id (default: alba)
              --language <id>     Language pack id (default: english)
              --iters <n>         Timed iterations per config (default: 30)
              --warmup <n>        Warmup iterations per config, not recorded (default: 3)
              --chunk <n>         Chunk size for the hybrid path (default: 16)
              --also-synth        Also synthesize one WAV per config to /tmp
            """
        print(usage)
    }

    private static func runBenchmark(
        text: String,
        voice: String,
        language: PocketTtsLanguage,
        iters: Int,
        warmup: Int,
        chunk: Int,
        alsoSynth: Bool
    ) async throws {
        print("PocketTTS cond_step prefill benchmark")
        print("  language: \(language.rawValue)")
        print("  voice:    \(voice)")
        print("  text:     \"\(text)\"")
        print("  warmup:   \(warmup) iters")
        print("  iters:    \(iters)")
        print("  chunk:    \(chunk)")
        print("")

        // --- Legacy (chunk-1) ---
        let legacyManager = PocketTtsManager(
            defaultVoice: voice, language: language, condStepMode: .legacy
        )
        print("Initializing legacy manager (chunk-1 dispatch)...")
        let legacyInitStart = Date()
        try await legacyManager.initialize()
        let legacyInitElapsed = String(format: "%.2f", Date().timeIntervalSince(legacyInitStart))
        print("  done in \(legacyInitElapsed)s")

        print("Running legacy benchmark...")
        let legacy = try await legacyManager.benchmarkCondStepPrefill(
            text: text, voice: voice, warmup: warmup, iters: iters
        )

        // --- Hybrid (chunk-N + chunk-1) ---
        let chunkedManager = PocketTtsManager(
            defaultVoice: voice, language: language, condStepMode: .chunked(chunk: chunk)
        )
        print("Initializing chunked manager (chunk-\(chunk) + chunk-1 hybrid)...")
        let chunkedInitStart = Date()
        do {
            try await chunkedManager.initialize()
            let chunkedInitElapsed = String(format: "%.2f", Date().timeIntervalSince(chunkedInitStart))
            print("  done in \(chunkedInitElapsed)s")
        } catch {
            logger.error(
                "chunked manager init failed (is cond_step_chunk\(chunk).mlmodelc placed under the language root?): \(error)"
            )
            throw error
        }

        print("Running chunked benchmark...")
        let chunked = try await chunkedManager.benchmarkCondStepPrefill(
            text: text, voice: voice, warmup: warmup, iters: iters
        )

        printSummary(legacy: legacy, chunked: chunked)

        if alsoSynth {
            print("")
            print("Synthesizing one WAV per config...")
            let legacyURL = URL(fileURLWithPath: "/tmp/pocket-tts-cond-bench-legacy.wav")
            let chunkedURL = URL(fileURLWithPath: "/tmp/pocket-tts-cond-bench-chunked.wav")
            try await legacyManager.synthesizeToFile(text: text, outputURL: legacyURL, voice: voice)
            try await chunkedManager.synthesizeToFile(text: text, outputURL: chunkedURL, voice: voice)
            print("  wrote \(legacyURL.path)")
            print("  wrote \(chunkedURL.path)")
        }
    }

    private static func printSummary(
        legacy: PocketTtsManager.CondStepPrefillBenchmarkResult,
        chunked: PocketTtsManager.CondStepPrefillBenchmarkResult
    ) {
        let legacyMs = stats(legacy.durations)
        let chunkedMs = stats(chunked.durations)
        let speedup = legacyMs.median / max(chunkedMs.median, 1e-9)

        print("")
        print(
            "SUMMARY — text tokens=\(legacy.textTokens), voice tokens=\(legacy.voiceTokens) (legacy) / \(chunked.voiceTokens) (chunked)"
        )
        print("  config   |  median   |   min     |   stdev   |  iters")
        print("  ---------+-----------+-----------+-----------+-------")
        print(
            "  legacy   | \(fmt(legacyMs.median)) | \(fmt(legacyMs.min)) | \(fmt(legacyMs.stdev)) | \(legacy.durations.count)"
        )
        let chunkLabel = chunked.chunkSize.map { "chunk-\($0)" } ?? "chunked"
        print(
            "  \(pad(chunkLabel, 8)) | \(fmt(chunkedMs.median)) | \(fmt(chunkedMs.min)) | \(fmt(chunkedMs.stdev)) | \(chunked.durations.count)"
        )
        print("")
        let speedupStr = String(format: "%.2f", speedup)
        print("  speedup (legacy median / chunked median): \(speedupStr)x")
    }

    private static func fmt(_ seconds: TimeInterval) -> String {
        // Render in milliseconds, fixed-width 8 chars (e.g. " 12.34 ms").
        let ms = seconds * 1000.0
        return String(format: "%6.2f ms", ms)
    }

    private static func pad(_ s: String, _ width: Int) -> String {
        if s.count >= width { return s }
        return s + String(repeating: " ", count: width - s.count)
    }

    private static func stats(
        _ values: [TimeInterval]
    ) -> (
        median: TimeInterval, min: TimeInterval, stdev: TimeInterval
    ) {
        guard !values.isEmpty else { return (0, 0, 0) }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        let median: TimeInterval
        if sorted.count.isMultiple(of: 2) {
            median = (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            median = sorted[mid]
        }
        let minVal = sorted.first ?? 0
        let mean = values.reduce(0, +) / Double(values.count)
        let variance =
            values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
        let stdev = variance.squareRoot()
        return (median, minVal, stdev)
    }
}
