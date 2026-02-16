#if os(macOS)
import FluidAudio
import Foundation

/// Command to run forced alignment on audio files using Qwen3-ForcedAligner.
enum AlignCommand {
    private static let logger = AppLogger(category: "Align")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var text: String?
        var modelDir: String?

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--text", "-t":
                if i + 1 < arguments.count {
                    text = arguments[i + 1]
                    i += 1
                }
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        guard let transcript = text else {
            logger.error("--text is required")
            printUsage()
            exit(1)
        }

        await alignAudio(audioFile: audioFile, text: transcript, modelDir: modelDir)
    }

    private static func alignAudio(audioFile: String, text: String, modelDir: String?) async {
        guard #available(macOS 14, iOS 17, *) else {
            logger.error("ForcedAligner requires macOS 14 or later")
            return
        }

        do {
            let manager = ForcedAlignerManager()

            if let dir = modelDir {
                logger.info("Loading ForcedAligner models from: \(dir)")
                let dirURL = URL(fileURLWithPath: dir)
                try await manager.loadModels(from: dirURL)
            } else {
                logger.info("Downloading ForcedAligner models from HuggingFace...")
                try await manager.downloadAndLoadModels()
            }

            // Load and resample audio to 16kHz mono
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(samples.count) / Double(ForcedAlignerConfig.sampleRate)
            logger.info(
                "Audio: \(String(format: "%.2f", duration))s, \(samples.count) samples at 16kHz"
            )

            logger.info("Aligning text: \(text.prefix(80))...")
            let result = try await manager.align(audioSamples: samples, text: text)

            // Output results
            logger.info(String(repeating: "=", count: 50))
            logger.info("FORCED ALIGNMENT RESULTS (\(String(format: "%.0f", result.latencyMs))ms)")
            logger.info(String(repeating: "=", count: 50))

            for alignment in result.alignments {
                let line = String(
                    format: "  %-15s %8.1f - %8.1f ms",
                    (alignment.word as NSString).utf8String ?? "",
                    alignment.startMs,
                    alignment.endMs
                )
                print(line)
            }

            let rtfx = duration / (result.latencyMs / 1000.0)
            print("")
            print("Performance:")
            print("  Audio duration: \(String(format: "%.2f", duration))s")
            print("  Alignment time: \(String(format: "%.0f", result.latencyMs))ms")
            print("  RTFx: \(String(format: "%.2f", rtfx))x")
            print("  Words aligned: \(result.alignments.count)")

        } catch {
            logger.error("Forced alignment failed: \(error)")
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Forced Alignment Command (Qwen3-ForcedAligner-0.6B)

            Usage: fluidaudio align <audio_file> --text "transcript" [options]

            Options:
                --help, -h              Show this help message
                --text, -t <text>       Transcript text to align (required)
                --model-dir <path>      Path to local model directory (skips download)

            Examples:
                fluidaudio align audio.wav --text "hello world how are you"
                fluidaudio align speech.wav -t "the quick brown fox" --model-dir /path/to/models
            """
        )
    }
}
#endif
