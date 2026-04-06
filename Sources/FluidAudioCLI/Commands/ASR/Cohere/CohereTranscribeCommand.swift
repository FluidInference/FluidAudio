#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// Command to transcribe audio files using Cohere Transcribe.
enum CohereTranscribeCommand {
    private static let logger = AppLogger(category: "CohereTranscribe")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var modelDir: String?
        var variant: CohereAsrVariant = .int8
        var maxTokens = 200
        var cpuOnly = false

        // Parse options
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--variant":
                if i + 1 < arguments.count {
                    let v = arguments[i + 1].lowercased()
                    if let parsed = CohereAsrVariant(rawValue: v) {
                        variant = parsed
                    } else {
                        logger.error("Unknown variant '\(arguments[i + 1])'. Use 'fp16' or 'int8'.")
                        exit(1)
                    }
                    i += 1
                }
            case "--max-tokens":
                if i + 1 < arguments.count, let tokens = Int(arguments[i + 1]) {
                    maxTokens = tokens
                    i += 1
                }
            case "--cpu-only":
                cpuOnly = true
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        await transcribe(
            audioFile: audioFile,
            modelDir: modelDir,
            variant: variant,
            maxTokens: maxTokens,
            cpuOnly: cpuOnly
        )
    }

    private static func transcribe(
        audioFile: String,
        modelDir: String?,
        variant: CohereAsrVariant,
        maxTokens: Int,
        cpuOnly: Bool = false
    ) async {
        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Cohere Transcribe requires macOS 14 or later")
            return
        }

        do {
            // Load models
            let manager = CohereAsrManager()
            let computeUnits: MLComputeUnits = cpuOnly ? .cpuAndGPU : .all

            if let dir = modelDir {
                logger.info(
                    "Loading Cohere Transcribe models from: \(dir) (compute units: \(cpuOnly ? "CPU+GPU" : "All"))")
                let dirURL = URL(fileURLWithPath: dir)
                try await manager.loadModels(from: dirURL, computeUnits: computeUnits)
            } else {
                logger.info("Downloading Cohere Transcribe \(variant.rawValue) models from HuggingFace...")
                let cacheDir = try await CohereAsrModels.download(variant: variant)
                try await manager.loadModels(from: cacheDir, computeUnits: computeUnits)
            }

            // Load and resample audio to 16kHz mono
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(samples.count) / Double(CohereAsrConfig.sampleRate)
            logger.info(
                "Audio: \(String(format: "%.2f", duration))s, \(samples.count) samples at 16kHz"
            )

            // Transcribe
            logger.info("Transcribing...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let text = try await manager.transcribe(
                audioSamples: samples,
                maxNewTokens: maxTokens
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            let rtfx = duration / elapsed

            // Output
            logger.info(String(repeating: "=", count: 50))
            logger.info("COHERE TRANSCRIBE")
            logger.info(String(repeating: "=", count: 50))
            print(text)
            logger.info("")
            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", duration))s")
            logger.info("  Processing time: \(String(format: "%.2f", elapsed))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")

        } catch {
            logger.error("Cohere Transcribe failed: \(error)")
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Cohere Transcribe Command

            Usage: fluidaudio cohere-transcribe <audio_file> [options]

            Options:
                --help, -h              Show this help message
                --model-dir <path>      Path to local model directory (skips download)
                --variant <fp16|int8>   Model variant (default: int8). int8 uses 2.6x less disk/RAM.
                --max-tokens <n>        Maximum tokens to generate (default: 200)
                --cpu-only              Use CPU+GPU only (skip ANE compilation, faster startup)

            Supported languages (14 total):
                en   English             fr   French              de   German
                es   Spanish             it   Italian             pt   Portuguese
                nl   Dutch               pl   Polish              el   Greek
                ar   Arabic              ja   Japanese            zh   Chinese
                ko   Korean              vi   Vietnamese

            Examples:
                fluidaudio cohere-transcribe audio.wav
                fluidaudio cohere-transcribe meeting.wav --variant fp16
                fluidaudio cohere-transcribe long.wav --max-tokens 500
                fluidaudio cohere-transcribe audio.wav --model-dir /path/to/cohere-models
            """
        )
    }
}
#endif
