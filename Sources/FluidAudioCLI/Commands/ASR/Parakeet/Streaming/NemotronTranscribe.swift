#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Nemotron Speech Streaming transcription for custom audio files
public class NemotronTranscribe {
    private let logger = AppLogger(category: "NemotronTranscribe")

    public struct Config {
        var inputFiles: [URL] = []
        var modelDir: URL?
        var chunkSize: NemotronChunkSize = .ms1120

        public init() {}
    }

    private let config: Config

    public init(config: Config = Config()) {
        self.config = config
    }

    /// Run CLI transcription
    public static func run(arguments: [String]) async {
        let logger = AppLogger(category: "NemotronTranscribe")

        var config = Config()

        // Parse arguments
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
            case "--chunk", "-c":
                i += 1
                if i < arguments.count, let ms = Int(arguments[i]) {
                    switch ms {
                    case 1120: config.chunkSize = .ms1120
                    case 560: config.chunkSize = .ms560
                    case 160: config.chunkSize = .ms160
                    case 80: config.chunkSize = .ms80
                    default:
                        logger.warning(
                            "Invalid chunk size: \(ms)ms. Valid options: 1120, 560, 160, or 80. Using default 1120ms.")
                    }
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

        let transcriber = NemotronTranscribe(config: config)
        await transcriber.run()
    }

    private static func printUsage() {
        print(
            """
            Nemotron Speech Streaming Transcription

            Usage: fluidaudio nemotron-transcribe [options]

            Options:
                --input, -i <path>        Audio file to transcribe (.wav) - required, can be used multiple times
                --model-dir, -m <path>    Path to Nemotron CoreML models (optional, auto-downloads if not provided)
                --chunk, -c <ms>          Chunk size: 1120, 560, 160, or 80 (default: 1120)
                --help, -h                Show this help

            Chunk Sizes:
                1120ms  Original chunk size (1.12s) - best accuracy & speed
                560ms   Half chunk size (0.56s) - lower latency
                160ms   Very low latency (0.16s)
                80ms    Ultra low latency (0.08s)

            Examples:
                # Transcribe a single file
                fluidaudio nemotron-transcribe --input audio.wav

                # Transcribe multiple files with 560ms chunks
                fluidaudio nemotron-transcribe -i file1.wav -i file2.wav --chunk 560

                # Ultra low latency with 160ms chunks
                fluidaudio nemotron-transcribe --input audio.wav --chunk 160

                # Use custom model directory
                fluidaudio nemotron-transcribe --input audio.wav --model-dir ~/my-models
            """
        )
    }

    /// Run transcription
    public func run() async {
        logger.info(String(repeating: "=", count: 70))
        logger.info("NEMOTRON SPEECH STREAMING TRANSCRIPTION (\(config.chunkSize.rawValue)ms chunks)")
        logger.info(String(repeating: "=", count: 70))

        #if DEBUG
        logger.warning("WARNING: Running in DEBUG mode!")
        logger.warning("For optimal performance, use: swift run -c release fluidaudio nemotron-transcribe")
        try? await Task.sleep(nanoseconds: 2_000_000_000)
        #else
        logger.info("Running in RELEASE mode - optimal performance")
        #endif

        do {
            // Download Nemotron models if needed
            let modelDir = try await getOrDownloadModels()

            // Load models
            logger.info("Loading Nemotron models...")
            let manager = StreamingNemotronAsrManager()
            try await manager.loadModels(from: modelDir)
            logger.info("Models loaded successfully")
            logger.info("")

            // Process each input file
            for (index, fileURL) in config.inputFiles.enumerated() {
                logger.info("[\(index + 1)/\(config.inputFiles.count)] Processing: \(fileURL.lastPathComponent)")

                guard FileManager.default.fileExists(atPath: fileURL.path) else {
                    logger.error("  File not found: \(fileURL.path)")
                    continue
                }

                do {
                    // Load audio file
                    let audioFile = try AVAudioFile(forReading: fileURL)
                    guard
                        let buffer = AVAudioPCMBuffer(
                            pcmFormat: audioFile.processingFormat,
                            frameCapacity: AVAudioFrameCount(audioFile.length)
                        )
                    else {
                        logger.error("  Failed to create audio buffer")
                        continue
                    }
                    try audioFile.read(into: buffer)

                    let audioDuration = Double(audioFile.length) / audioFile.processingFormat.sampleRate

                    // Transcribe
                    let startTime = Date()
                    _ = try await manager.process(audioBuffer: buffer)
                    let transcript = try await manager.finish()
                    let processingTime = Date().timeIntervalSince(startTime)

                    let rtf = audioDuration > 0 ? processingTime / audioDuration : 0.0
                    let rtfx = rtf > 0 ? 1.0 / rtf : 0.0

                    // Output results
                    logger.info("  Duration:    \(String(format: "%.2f", audioDuration))s")
                    logger.info("  Processing:  \(String(format: "%.2f", processingTime))s")
                    logger.info("  RTFx:        \(String(format: "%.1f", rtfx))x")
                    logger.info("  Transcript:  \(transcript)")
                    logger.info("")

                    // Reset for next file
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

    private func getOrDownloadModels() async throws -> URL {
        if let modelDir = config.modelDir {
            return modelDir
        }

        let repo = config.chunkSize.repo

        // Check default cache location
        // Note: downloadRepo appends repo.folderName internally, so we use the parent dir
        let modelsBaseDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/fluidaudio/models")
        let cacheDir = modelsBaseDir.appendingPathComponent(repo.folderName)

        // Check for int8 encoder (the only format loaded by StreamingNemotronAsrManager)
        let encoderInt8Path = cacheDir.appendingPathComponent("encoder/encoder_int8.mlmodelc")

        if FileManager.default.fileExists(atPath: encoderInt8Path.path) {
            logger.info("Using cached Nemotron models at \(cacheDir.path)")
            return cacheDir
        }

        // Download models (downloadRepo appends folderName internally)
        logger.info("Downloading Nemotron \(config.chunkSize.rawValue)ms models from HuggingFace...")
        try await DownloadUtils.downloadRepo(repo, to: modelsBaseDir)

        return cacheDir
    }
}
#endif
