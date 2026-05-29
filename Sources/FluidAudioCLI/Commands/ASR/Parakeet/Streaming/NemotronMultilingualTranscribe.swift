#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Nemotron Speech Streaming Multilingual transcription for custom audio files.
///
/// Local-path-only: the multilingual model is not yet uploaded to HuggingFace,
/// so the caller must supply `--model-dir` pointing at a directory that
/// contains the compiled `.mlmodelc` (or uncompiled `.mlpackage`) bundles plus
/// `metadata.json` and `tokenizer.json`.
public class NemotronMultilingualTranscribe {
    private let logger = AppLogger(category: "NemotronMultilingualTranscribe")

    public struct Config {
        var inputFiles: [URL] = []
        var modelDir: URL?
        /// Language code passed to `setLanguage(_:)` (e.g. `"en-US"`, `"zh-CN"`,
        /// `"auto"`). When `nil`, the manager uses its `default_prompt_id`.
        var language: String?
        /// Raw prompt id override. Takes precedence over `language` if set.
        var promptId: Int?
        /// Chunk-size tier in ms (560 / 1120 / 2240 / 4480) for auto-download.
        var chunkMs: Int = 2240

        public init() {}
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Run CLI transcription
    public static func run(arguments: [String]) async {
        let logger = AppLogger(category: "NemotronMultilingualTranscribe")

        var config = Config()

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]

            switch arg {
            case "--input", "-i":
                i += 1
                if i < arguments.count {
                    let path = arguments[i]
                    let url = URL(fileURLWithPath: path)
                    config.inputFiles.append(url)
                }
            case "--model-dir", "-m":
                i += 1
                if i < arguments.count {
                    config.modelDir = URL(fileURLWithPath: arguments[i])
                }
            case "--language", "-l":
                i += 1
                if i < arguments.count {
                    config.language = arguments[i]
                }
            case "--prompt-id":
                i += 1
                if i < arguments.count, let pid = Int(arguments[i]) {
                    config.promptId = pid
                }
            case "--chunk-ms":
                i += 1
                if i < arguments.count, let ms = Int(arguments[i]) {
                    config.chunkMs = ms
                }
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arg)")
            }
            i += 1
        }

        if config.inputFiles.isEmpty {
            logger.error("No input files specified. Use --input <path> to add audio files.")
            printUsage()
            return
        }

        if config.modelDir == nil {
            // Auto-download the requested <language>/<chunkMs>ms variant
            // (compiled .mlmodelc only) from HuggingFace.
            do {
                logger.info(
                    "No --model-dir supplied; downloading multilingual variant "
                        + "(\(config.language ?? "auto") @ \(config.chunkMs)ms) from HuggingFace..."
                )
                let dir = try await StreamingNemotronMultilingualAsrManager.downloadVariant(
                    languageCode: config.language ?? "auto",
                    chunkMs: config.chunkMs
                )
                config.modelDir = dir
                logger.info("Downloaded to \(dir.path)")
            } catch {
                logger.error("Auto-download failed: \(error.localizedDescription)")
                return
            }
        }

        let transcriber = NemotronMultilingualTranscribe(config: config)
        await transcriber.run()
    }

    private static func printUsage() {
        print(
            """
            Nemotron Speech Streaming Multilingual Transcription

            Usage: fluidaudio nemotron-multilingual-transcribe [options]

            Options:
                --input, -i <path>        Audio file to transcribe (.wav) - required, repeatable
                --model-dir, -m <path>    Local path to CoreML models. If omitted, the
                                          matching variant is auto-downloaded from HuggingFace.
                --language, -l <code>     Language hint (e.g. en-US, zh-CN, ja-JP, de-DE, auto).
                                          Also selects the per-language ship for auto-download.
                --chunk-ms <int>          Chunk-size tier for auto-download: 560 / 1120 /
                                          2240 (default, recommended) / 4480.
                --prompt-id <int>         Raw prompt id (overrides --language)
                --help, -h                Show this help

            Notes:
                - Auto-download pulls compiled .mlmodelc only from
                  FluidInference/Nemotron-3.5-ASR-Streaming-Multilingual-0.6b-CoreML,
                  cached under ~/Library/Application Support/FluidAudio/Models/.
                - Per-language hints (en/es/fr/it/pt/de/zh/ja) fetch the vocab-pruned
                  ship; "auto" or any other language fetches the full multilingual model.
                - When neither --language nor --prompt-id is provided, the model's
                  default prompt id ("auto") is used.

            Examples:
                # Auto-download German @ 2240ms and transcribe
                fluidaudio nemotron-multilingual-transcribe \\
                    --input audio.wav --language de-DE

                # Pin a chunk tier
                fluidaudio nemotron-multilingual-transcribe \\
                    --input audio.wav --language ja-JP --chunk-ms 1120

                # Use a local model directory instead of downloading
                fluidaudio nemotron-multilingual-transcribe \\
                    --input audio.wav --model-dir ~/my-models
            """
        )
    }

    /// Run transcription
    public func run() async {
        logger.info(String(repeating: "=", count: 70))
        logger.info("NEMOTRON SPEECH STREAMING MULTILINGUAL TRANSCRIPTION")
        logger.info(String(repeating: "=", count: 70))

        #if DEBUG
        logger.warning("WARNING: Running in DEBUG mode!")
        logger.warning(
            "For optimal performance, use: swift run -c release fluidaudio nemotron-multilingual-transcribe"
        )
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        #else
        logger.info("Running in RELEASE mode - optimal performance")
        #endif

        guard let modelDir = config.modelDir else {
            logger.error("Missing --model-dir")
            return
        }

        do {
            logger.info("Loading Nemotron multilingual models from \(modelDir.path)...")
            let manager = StreamingNemotronMultilingualAsrManager()
            try await manager.loadModels(from: modelDir)
            logger.info("Models loaded successfully")

            // Apply language / prompt-id selection (prompt-id wins)
            if let pid = config.promptId {
                await manager.setPromptId(pid)
                logger.info("Prompt id set to \(pid)")
            } else if let language = config.language {
                await manager.setLanguage(language)
                logger.info("Language hint: \(language)")
            } else {
                logger.info("Using default prompt id (auto)")
            }
            logger.info("")

            for (index, fileURL) in config.inputFiles.enumerated() {
                logger.info("[\(index + 1)/\(config.inputFiles.count)] Processing: \(fileURL.lastPathComponent)")

                guard FileManager.default.fileExists(atPath: fileURL.path) else {
                    logger.error("  File not found: \(fileURL.path)")
                    continue
                }

                do {
                    let audioFile = try AVAudioFile(forReading: fileURL)
                    let audioDuration = Double(audioFile.length) / audioFile.processingFormat.sampleRate

                    // Streaming read: feed the manager in 60-second blocks
                    // instead of allocating one giant PCM buffer for the
                    // whole file. Lifts the ~2 GB AVAudioPCMBuffer ceiling
                    // (>=20h files) and reduces peak memory pressure.
                    let blockSeconds: Double = 60
                    let blockFrames = AVAudioFrameCount(audioFile.processingFormat.sampleRate * blockSeconds)

                    let startTime = Date()
                    while audioFile.framePosition < audioFile.length {
                        let remaining = AVAudioFrameCount(audioFile.length - audioFile.framePosition)
                        let thisFrames = min(blockFrames, remaining)
                        guard
                            let block = AVAudioPCMBuffer(
                                pcmFormat: audioFile.processingFormat,
                                frameCapacity: thisFrames
                            )
                        else {
                            logger.error("  Failed to create audio buffer for block")
                            break
                        }
                        try audioFile.read(into: block, frameCount: thisFrames)
                        _ = try await manager.process(audioBuffer: block)
                    }
                    let transcript = try await manager.finish()
                    let processingTime = Date().timeIntervalSince(startTime)

                    let detected = await manager.detectedLanguage() ?? "(none)"

                    let rtf = audioDuration > 0 ? processingTime / audioDuration : 0.0
                    let rtfx = rtf > 0 ? 1.0 / rtf : 0.0

                    logger.info("  Duration:    \(String(format: "%.2f", audioDuration))s")
                    logger.info("  Processing:  \(String(format: "%.2f", processingTime))s")
                    logger.info("  RTFx:        \(String(format: "%.1f", rtfx))x")
                    logger.info("  Detected:    \(detected)")
                    logger.info("  Transcript:  \(transcript)")
                    logger.info("")

                    await manager.reset()

                } catch {
                    logger.error("  Error: \(error.localizedDescription)")
                    logger.info("")
                }
            }

            logger.info(String(repeating: "=", count: 70))
            logger.info("Transcription complete")

        } catch {
            logger.error("Fatal error: \(error.localizedDescription)")
        }
    }
}
#endif
