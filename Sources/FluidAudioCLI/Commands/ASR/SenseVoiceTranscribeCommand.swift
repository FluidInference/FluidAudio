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
        var showScores = false
        var repeatCount: Int?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--int8":
                precision = .int8
            case "--fp32":
                precision = .fp32
            case "--verbose", "-v":
                verbose = true
            case "--scores":
                showScores = true
            case "--repeat":
                if i + 1 < arguments.count, let n = Int(arguments[i + 1]), n > 0 {
                    repeatCount = n
                    i += 1
                }
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

            let samples = try AudioConverter(sampleRate: Double(SenseVoiceConfig.sampleRate))
                .resampleAudioFile(audioURL)
            let start = Date()
            var result = try await manager.transcribeDetailed(audio: samples)
            let elapsed = Date().timeIntervalSince(start)
            if verbose { logger.info("Transcribed in \(String(format: "%.2f", elapsed))s") }
            if let repeatCount {
                // The first call above warms the model; time the next `repeatCount`.
                let start = Date()
                for _ in 0..<repeatCount { result = try await manager.transcribeDetailed(audio: samples) }
                let average = Date().timeIntervalSince(start) / Double(repeatCount)
                let audioSeconds = Double(samples.count) / Double(SenseVoiceConfig.sampleRate)
                print(
                    String(
                        format: "Average %.1f ms per call over %d calls (RTFx %.0f)",
                        average * 1000, repeatCount, audioSeconds / average))
            }
            print(result.text)
            if showScores {
                func line(_ name: String, _ scores: [SenseVoiceTagScore]) {
                    let top = scores.prefix(4).map { "\($0.tag) \(String(format: "%.3f", $0.probability))" }
                    print("\(name): \(top.joined(separator: ", "))")
                }
                print("tags: \(result.tags.joined(separator: " "))")
                line("language", result.scores.language)
                line("emotion", result.scores.emotion)
                line("event", result.scores.event)
            }
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
              --scores      Print language, emotion and audio-event probabilities
              --repeat N    Warm up once, then time N transcriptions
              --help,-h     Show this help
            """
        )
    }
}
#endif
