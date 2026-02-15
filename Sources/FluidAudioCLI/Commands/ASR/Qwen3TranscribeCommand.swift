#if os(macOS)
import FluidAudio
import Foundation

/// Command to transcribe audio files using Qwen3-ASR.
enum Qwen3TranscribeCommand {
    private static let logger = AppLogger(category: "Qwen3Transcribe")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var modelDir: String?
        var language: Qwen3AsrConfig.Language?
        var variant: Qwen3AsrVariant = .f32

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
            case "--language", "-l":
                if i + 1 < arguments.count {
                    let langStr = arguments[i + 1]
                    if let lang = Qwen3AsrConfig.Language(from: langStr) {
                        language = lang
                    } else {
                        logger.warning(
                            "Unknown language '\(langStr)'. Use --help to see supported languages."
                        )
                    }
                    i += 1
                }
            case "--variant":
                if i + 1 < arguments.count {
                    let v = arguments[i + 1].lowercased()
                    if let parsed = Qwen3AsrVariant(rawValue: v) {
                        variant = parsed
                    } else {
                        logger.error("Unknown variant '\(arguments[i + 1])'. Use 'f32' or 'int8'.")
                        exit(1)
                    }
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        await transcribe(audioFile: audioFile, modelDir: modelDir, language: language, variant: variant)
    }

    private static func transcribe(
        audioFile: String,
        modelDir: String?,
        language: Qwen3AsrConfig.Language?,
        variant: Qwen3AsrVariant = .f32
    ) async {
        guard #available(macOS 15, iOS 18, *) else {
            logger.error("Qwen3-ASR requires macOS 15 or later")
            return
        }

        do {
            // Load models
            let manager = Qwen3AsrManager()

            if let dir = modelDir {
                logger.info("Loading Qwen3-ASR models from: \(dir)")
                let dirURL = URL(fileURLWithPath: dir)
                try await manager.loadModels(from: dirURL)
            } else {
                logger.info("Downloading Qwen3-ASR \(variant.rawValue) models from HuggingFace...")
                let cacheDir = try await Qwen3AsrModels.download(variant: variant)
                try await manager.loadModels(from: cacheDir)
            }

            // Load and resample audio to 16kHz mono
            let samples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Double(samples.count) / Double(Qwen3AsrConfig.sampleRate)
            logger.info(
                "Audio: \(String(format: "%.2f", duration))s, \(samples.count) samples at 16kHz"
            )

            // Transcribe
            let langDesc = language?.englishName ?? "auto-detect"
            logger.info("Transcribing (language: \(langDesc))...")
            let startTime = CFAbsoluteTimeGetCurrent()
            let text = try await manager.transcribe(
                audioSamples: samples,
                language: language,
                maxNewTokens: 512
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - startTime

            let rtfx = duration / elapsed

            // Output
            logger.info(String(repeating: "=", count: 50))
            logger.info("QWEN3-ASR TRANSCRIPTION")
            logger.info(String(repeating: "=", count: 50))
            print(text)
            logger.info("")
            logger.info("Performance:")
            logger.info("  Audio duration: \(String(format: "%.2f", duration))s")
            logger.info("  Processing time: \(String(format: "%.2f", elapsed))s")
            logger.info("  RTFx: \(String(format: "%.2f", rtfx))x")

        } catch {
            logger.error("Qwen3-ASR transcription failed: \(error)")
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Qwen3-ASR Transcribe Command

            Usage: fluidaudio qwen3-transcribe <audio_file> [options]

            Options:
                --help, -h              Show this help message
                --model-dir <path>      Path to local model directory (skips download)
                --variant <f32|int8>    Model variant (default: f32). int8 uses ~50% less RAM.
                --language, -l <code>   Language hint (e.g., zh, en, ja, ko, yue, ar, fr, de)

            Supported languages (30 total):
                zh   Chinese (Mandarin)   yue  Cantonese          en   English
                ja   Japanese             ko   Korean             vi   Vietnamese
                th   Thai                 id   Indonesian         ms   Malay
                hi   Hindi                ar   Arabic             tr   Turkish
                ru   Russian              de   German             fr   French
                es   Spanish              pt   Portuguese         it   Italian
                nl   Dutch                pl   Polish             sv   Swedish
                da   Danish               fi   Finnish            cs   Czech
                fil  Filipino             fa   Persian            el   Greek
                hu   Hungarian            mk   Macedonian         ro   Romanian

            Examples:
                fluidaudio qwen3-transcribe audio.wav
                fluidaudio qwen3-transcribe chinese.wav --language zh
                fluidaudio qwen3-transcribe meeting.wav --model-dir /path/to/qwen3-asr
            """
        )
    }
}
#endif
