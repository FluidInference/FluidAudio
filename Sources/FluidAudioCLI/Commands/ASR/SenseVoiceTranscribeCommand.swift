#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `sensevoice-transcribe <audio> [--fp32] [--verbose]`
enum SenseVoiceTranscribeCommand {
    private static let logger = AppLogger(category: "SenseVoiceTranscribe")

    static func run(arguments: [String]) async {
        var audioPath: String?
        var precision: SenseVoiceEncoderPrecision = .fp16
        var verbose = false

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--int8":
                precision = .int8
            case "--fp32":
                precision = .fp32
            case "--verbose", "-v":
                verbose = true
            case "--help", "-h":
                printUsage()
                return
            default:
                if audioPath == nil { audioPath = arguments[i] }
            }
            i += 1
        }

        guard let audioPath else {
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
            logger.info("Loading SenseVoice models (encoder: \(precision.rawValue))...")
            let manager = try await SenseVoiceManager.load(precision: precision)

            let start = Date()
            let text = try await manager.transcribe(audioURL: audioURL)
            let elapsed = Date().timeIntervalSince(start)

            if verbose { logger.info("Transcribed in \(String(format: "%.2f", elapsed))s") }
            print(text)
        } catch {
            logger.error("Transcription failed: \(error)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio sensevoice-transcribe <audio-file> [options]

            Transcribe audio with SenseVoiceSmall (multilingual, non-autoregressive).

            Options:
              --int8        Use the int8 encoder (~half size, ANE, accuracy-neutral)
              --fp32        Use the fp32 encoder (for hardware without a Neural Engine)
              --verbose,-v  Print timing
              --help,-h     Show this help
            """
        )
    }
}
#endif
