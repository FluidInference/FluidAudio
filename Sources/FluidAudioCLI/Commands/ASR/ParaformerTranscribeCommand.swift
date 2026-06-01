#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `paraformer-transcribe <audio> [--verbose]`
enum ParaformerTranscribeCommand {
    private static let logger = AppLogger(category: "ParaformerTranscribe")

    static func run(arguments: [String]) async {
        var audioPath: String?
        var verbose = false
        var precision: ParaformerPrecision = .fp16
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--int8": precision = .int8
            case "--verbose", "-v": verbose = true
            case "--help", "-h":
                print("Usage: fluidaudio paraformer-transcribe <audio-file> [--int8] [--verbose]")
                return
            default: if audioPath == nil { audioPath = arguments[i] }
            }
            i += 1
        }
        guard let audioPath else {
            logger.error("Error: No audio file specified")
            return
        }
        let url = URL(fileURLWithPath: audioPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.error("Error: Audio file not found: \(audioPath)")
            return
        }
        do {
            logger.info("Loading Paraformer-large (zh) models (\(precision.rawValue))...")
            let manager = try await ParaformerManager.load(precision: precision)
            let start = Date()
            let text = try await manager.transcribe(audioURL: url)
            if verbose { logger.info("Transcribed in \(String(format: "%.2f", Date().timeIntervalSince(start)))s") }
            print(text)
        } catch {
            logger.error("Transcription failed: \(error)")
        }
    }
}
#endif
