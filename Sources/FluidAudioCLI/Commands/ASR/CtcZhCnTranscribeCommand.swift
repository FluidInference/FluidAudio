#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

enum CtcZhCnTranscribeCommand {
    private static let logger = AppLogger(category: "CtcZhCnTranscribe")

    static func run(arguments: [String]) async {
        // Parse arguments
        var audioPath: String?
        var useInt8 = true
        var verbose = false

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--fp32":
                useInt8 = false
            case "--int8":
                useInt8 = true
            case "--verbose", "-v":
                verbose = true
            case "--help", "-h":
                printUsage()
                return
            default:
                if audioPath == nil {
                    audioPath = arg
                }
            }
            i += 1
        }

        guard let audioPath = audioPath else {
            logger.error("Error: No audio file specified")
            printUsage()
            return
        }

        let audioURL = URL(fileURLWithPath: audioPath)
        guard FileManager.default.fileExists(atPath: audioURL.path) else {
            logger.error("Error: Audio file not found: \(audioPath)")
            return
        }

        do {
            logger.info("Loading CTC zh-CN models (encoder: \(useInt8 ? "int8" : "fp32"))...")

            let manager = try await CtcZhCnManager.load(
                useInt8Encoder: useInt8,
                progressHandler: verbose ? createProgressHandler() : nil
            )

            logger.info("Transcribing: \(audioPath)")

            let startTime = Date()
            let text = try await manager.transcribe(audioURL: audioURL)
            let elapsed = Date().timeIntervalSince(startTime)

            logger.info("Transcription completed in \(String(format: "%.2f", elapsed))s")
            logger.info("")
            logger.info("Result:")
            print(text)

        } catch {
            logger.error("Transcription failed: \(error.localizedDescription)")
            if verbose {
                logger.error("Error details: \(String(describing: error))")
            }
        }
    }

    private static func createProgressHandler() -> DownloadUtils.ProgressHandler {
        return { progress in
            let percentage = progress.fractionCompleted * 100.0
            switch progress.phase {
            case .listing:
                logger.info("Listing files from repository...")
            case .downloading(let completed, let total):
                logger.info(
                    "Downloading models: \(completed)/\(total) files (\(String(format: "%.1f", percentage))%)"
                )
            case .compiling(let modelName):
                logger.info("Compiling \(modelName)...")
            }
        }
    }

    private static func printUsage() {
        logger.info(
            """
            CTC zh-CN Transcribe - Mandarin Chinese speech recognition

            Usage: fluidaudiocli ctc-zh-cn-transcribe <audio_file> [options]

            Arguments:
                <audio_file>    Path to audio file (WAV, MP3, etc.)

            Options:
                --int8          Use int8 quantized encoder (default, faster)
                --fp32          Use fp32 encoder (higher precision)
                --verbose, -v   Show download progress and detailed logs
                --help, -h      Show this help message

            Examples:
                # Basic transcription
                fluidaudiocli ctc-zh-cn-transcribe audio.wav

                # Use fp32 encoder for higher precision
                fluidaudiocli ctc-zh-cn-transcribe audio.wav --fp32

            Model Info:
                - Language: Mandarin Chinese (Simplified, zh-CN)
                - Vocabulary: 7000 SentencePiece tokens
                - Max audio: 15 seconds (longer audio is truncated)
                - Int8 encoder: 0.55GB (recommended)
                - FP32 encoder: 1.1GB

            Performance (FLEURS 100 samples):
                - Int8 encoder: 10.54% CER
                - FP32 encoder: 10.45% CER

            Note: Models auto-download from HuggingFace on first use.
            """
        )
    }
}
#endif
