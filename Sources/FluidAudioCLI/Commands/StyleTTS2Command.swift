import FluidAudio
import Foundation

/// `fluidaudio styletts2 "text" --voice ref_s.bin --output out.wav`
///
/// Drives the StyleTTS2 4-stage diffusion synthesizer end-to-end and writes
/// a 24 kHz mono 16-bit WAV. The `--voice` flag points at a precomputed
/// `ref_s.bin` style+prosody blob (see `mobius-styletts2/scripts/06_dump_ref_s.py`).
public enum StyleTTS2Command {

    private static let logger = AppLogger(category: "StyleTTS2Command")

    public static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            printUsage()
            return
        }

        var text: String?
        var voicePath: String?
        var outputPath = "styletts2.wav"
        var diffusionSteps = 5
        var alpha: Float = 0.3
        var beta: Float = 0.7
        var seed: UInt64?
        var tokenizeOnly = false
        var corpusPath: String?

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--voice":
                guard i + 1 < arguments.count else {
                    fputs("--voice requires a path\n", stderr)
                    exit(2)
                }
                voicePath = arguments[i + 1]
                i += 2
            case "--output", "-o":
                guard i + 1 < arguments.count else {
                    fputs("--output requires a path\n", stderr)
                    exit(2)
                }
                outputPath = arguments[i + 1]
                i += 2
            case "--steps":
                if i + 1 < arguments.count, let n = Int(arguments[i + 1]) {
                    diffusionSteps = n
                    i += 2
                } else {
                    fputs("--steps requires an integer\n", stderr)
                    exit(2)
                }
            case "--alpha":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    alpha = v
                    i += 2
                } else {
                    fputs("--alpha requires a float\n", stderr)
                    exit(2)
                }
            case "--beta":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    beta = v
                    i += 2
                } else {
                    fputs("--beta requires a float\n", stderr)
                    exit(2)
                }
            case "--seed":
                if i + 1 < arguments.count, let v = UInt64(arguments[i + 1]) {
                    seed = v
                    i += 2
                } else {
                    fputs("--seed requires an integer\n", stderr)
                    exit(2)
                }
            case "--tokenize-only":
                tokenizeOnly = true
                i += 1
            case "--corpus":
                guard i + 1 < arguments.count else {
                    fputs("--corpus requires a path\n", stderr)
                    exit(2)
                }
                corpusPath = arguments[i + 1]
                i += 2
            case "--help", "-h":
                printUsage()
                return
            default:
                if text == nil {
                    text = arg
                } else {
                    fputs("Unexpected argument: \(arg)\n", stderr)
                    exit(2)
                }
                i += 1
            }
        }

        if tokenizeOnly {
            await runTokenizeOnly(text: text, corpusPath: corpusPath)
            return
        }

        guard let text else {
            fputs("Missing required text argument\n", stderr)
            printUsage()
            exit(2)
        }
        guard let voicePath else {
            fputs("Missing --voice <path/to/ref_s.bin>\n", stderr)
            printUsage()
            exit(2)
        }

        let voiceURL = expand(voicePath)
        let outputURL = expand(outputPath)

        do {
            logger.notice("Initializing StyleTTS2 (will download models on first run)...")
            let manager = StyleTTS2Manager()
            try await manager.initialize { progress in
                logger.debug(
                    "Download \(progress.phase): "
                        + "\(Int(progress.fractionCompleted * 100))%")
            }

            logger.notice("Synthesizing text (\(text.count) chars, steps=\(diffusionSteps))...")
            let (phonemes, ids) = try await manager.tokenize(text: text)
            print("PHONEMES: \(phonemes)")
            print("TOKEN_IDS (\(ids.count)): \(ids)")
            let start = Date()
            let wav = try await manager.synthesize(
                text: text,
                voiceStyleURL: voiceURL,
                diffusionSteps: diffusionSteps,
                alpha: alpha,
                beta: beta,
                randomSeed: seed
            )
            let elapsed = Date().timeIntervalSince(start)

            try wav.write(to: outputURL)
            logger.notice(
                "Wrote \(outputURL.path) (\(wav.count) bytes) in "
                    + "\(String(format: "%.2f", elapsed))s")
        } catch {
            logger.error("StyleTTS2 synthesis failed: \(error)")
            exit(1)
        }
    }

    /// `--tokenize-only`: phonemize + encode without invoking the diffusion
    /// pipeline. Reports phoneme string, token id sequence, and any scalars
    /// that the 178-token espeak-ng vocab silently dropped. With `--corpus`
    /// runs over every line of a phrase file and aggregates a histogram of
    /// dropped scalars for the whole corpus.
    private static func runTokenizeOnly(text: String?, corpusPath: String?) async {
        do {
            let manager = StyleTTS2Manager()
            try await manager.initialize { _ in }

            var totalScalars = 0
            var totalIds = 0
            var totalDropped = 0
            var dropHist: [Unicode.Scalar: Int] = [:]
            var phraseCount = 0

            func process(_ phrase: String) async throws {
                let (phonemes, ids, dropped) =
                    try await manager.tokenizeWithReport(text: phrase)
                let scalars = phonemes.unicodeScalars.count
                totalScalars += scalars
                totalIds += ids.count
                let phraseDropCount = dropped.values.reduce(0, +)
                totalDropped += phraseDropCount
                for (k, v) in dropped { dropHist[k, default: 0] += v }
                phraseCount += 1

                if corpusPath == nil {
                    print("INPUT      : \(phrase)")
                    print("PHONEMES   : \(phonemes)")
                    print("TOKEN_IDS  (\(ids.count)): \(ids)")
                    let formatted =
                        dropped
                        .sorted { $0.value > $1.value }
                        .map {
                            "U+\(String($0.key.value, radix: 16, uppercase: true))"
                                + " '\($0.key)' ×\($0.value)"
                        }
                        .joined(separator: ", ")
                    print(
                        "DROPPED    (\(phraseDropCount) of \(scalars) scalars):"
                            + " \(formatted)")
                }
            }

            if let corpusPath {
                let url = expand(corpusPath)
                let raw = try String(contentsOf: url, encoding: .utf8)
                let phrases = raw.split(separator: "\n", omittingEmptySubsequences: true)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty && !$0.hasPrefix("#") }
                for (idx, phrase) in phrases.enumerated() {
                    do {
                        try await process(phrase)
                        let dropPct =
                            Double(totalDropped) / Double(max(totalScalars, 1)) * 100
                        if (idx + 1) % 10 == 0 || idx + 1 == phrases.count {
                            fputs(
                                "  [\(idx + 1)/\(phrases.count)] running drop rate "
                                    + "\(String(format: "%.2f", dropPct))%\n",
                                stderr)
                        }
                    } catch {
                        fputs("  [\(idx + 1)] phrase failed: \(error)\n", stderr)
                    }
                }
            } else if let text {
                try await process(text)
            } else {
                fputs("--tokenize-only requires either text or --corpus\n", stderr)
                exit(2)
            }

            let dropPct = Double(totalDropped) / Double(max(totalScalars, 1)) * 100
            let kept = totalScalars - totalDropped
            print("")
            print("=== StyleTTS2 vocab coverage ===")
            print("phrases                : \(phraseCount)")
            print("phoneme scalars total  : \(totalScalars)")
            print("encoded token ids      : \(totalIds)  (== kept scalars: \(kept))")
            print(
                "dropped scalars        : \(totalDropped)  "
                    + "(\(String(format: "%.2f", dropPct))%)")
            print("distinct dropped chars : \(dropHist.count)")
            if !dropHist.isEmpty {
                print("")
                print("dropped histogram (most → least frequent):")
                for (scalar, count) in dropHist.sorted(by: { $0.value > $1.value }) {
                    let hex = String(scalar.value, radix: 16, uppercase: true)
                    print(
                        "  \(String(format: "%6d", count))  U+\(hex)  '\(scalar)'")
                }
            }
        } catch {
            fputs("StyleTTS2 tokenize-only failed: \(error)\n", stderr)
            exit(1)
        }
    }

    private static func expand(_ path: String) -> URL {
        let exp = (path as NSString).expandingTildeInPath
        if exp.hasPrefix("/") {
            return URL(fileURLWithPath: exp)
        }
        let cwd = URL(
            fileURLWithPath: FileManager.default.currentDirectoryPath, isDirectory: true)
        return cwd.appendingPathComponent(exp)
    }

    private static func printUsage() {
        let msg = """
            Usage:
              fluidaudio styletts2 "<text>" --voice <ref_s.bin> [options]

            Options:
              --voice <path>    Required for synthesis. Path to precomputed ref_s.bin (256 fp32 LE).
              --output <path>   Output WAV path (default: styletts2.wav).
              --steps <int>     ADPM2 sampler steps (default: 5).
              --alpha <float>   Acoustic style mix weight (default: 0.3).
              --beta <float>    Prosody style mix weight (default: 0.7).
              --seed <uint>     Deterministic noise seed (default: system RNG).
              --tokenize-only   Run G2P + vocab encode only; report dropped scalars.
                                No --voice needed. Use with text or --corpus.
              --corpus <path>   Phrase-per-line corpus file (with --tokenize-only).

            Example:
              fluidaudio styletts2 "Hello world" \\
                  --voice /tmp/styletts2-ref_s.bin \\
                  --output /tmp/styletts2-hello.wav
            """
        print(msg)
    }
}
