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
            logger.error(
                "Missing --model-dir. The multilingual model is not auto-downloaded; supply a local path containing the .mlmodelc/.mlpackage bundles plus metadata.json and tokenizer.json."
            )
            printUsage()
            return
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
                --model-dir, -m <path>    Path to multilingual CoreML models (required)
                --language, -l <code>     Language hint (e.g. en-US, zh-CN, ja-JP, auto)
                --prompt-id <int>         Raw prompt id (overrides --language)
                --help, -h                Show this help

            Notes:
                - This command is local-path-only: the multilingual model is not
                  hosted on HuggingFace yet.
                - When neither --language nor --prompt-id is provided, the model's
                  default prompt id ("auto") is used.

            Examples:
                # Transcribe with auto language detection
                fluidaudio nemotron-multilingual-transcribe \\
                    --input audio.wav --model-dir ~/my-models

                # Transcribe with explicit language hint
                fluidaudio nemotron-multilingual-transcribe \\
                    --input audio.wav --model-dir ~/my-models --language zh-CN
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
                        guard let block = AVAudioPCMBuffer(
                            pcmFormat: audioFile.processingFormat,
                            frameCapacity: thisFrames
                        ) else {
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
