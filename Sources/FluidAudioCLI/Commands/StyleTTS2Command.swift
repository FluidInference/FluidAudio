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
              --voice <path>    Required. Path to precomputed ref_s.bin (256 fp32 LE).
              --output <path>   Output WAV path (default: styletts2.wav).
              --steps <int>     ADPM2 sampler steps (default: 5).
              --alpha <float>   Acoustic style mix weight (default: 0.3).
              --beta <float>    Prosody style mix weight (default: 0.7).
              --seed <uint>     Deterministic noise seed (default: system RNG).

            Example:
              fluidaudio styletts2 "Hello world" \\
                  --voice /tmp/styletts2-ref_s.bin \\
                  --output /tmp/styletts2-hello.wav
            """
        print(msg)
    }
}
