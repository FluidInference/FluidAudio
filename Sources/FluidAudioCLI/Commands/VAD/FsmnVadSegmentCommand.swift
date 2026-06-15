#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// `fsmn-vad-segment <audio>` — print detected speech segments [start_ms, end_ms].
enum FsmnVadSegmentCommand {
    private static let logger = AppLogger(category: "FsmnVadSegment")

    static func run(arguments: [String]) async {
        let paths = arguments.filter { !$0.hasPrefix("-") }
        guard let audioPath = paths.first else {
            print("Usage: fluidaudio fsmn-vad-segment <audio-file>")
            return
        }
        let url = URL(fileURLWithPath: audioPath)
        guard FileManager.default.fileExists(atPath: url.path) else {
            logger.error("Error: Audio file not found: \(audioPath)")
            return
        }
        do {
            logger.info("Loading FSMN-VAD models...")
            let vad = try await FsmnVadManager.load()
            let start = Date()
            let segments = try await vad.detect(audioURL: url)
            logger.info(
                "Detected \(segments.count) speech segment(s) in \(String(format: "%.2f", Date().timeIntervalSince(start)))s"
            )
            for s in segments {
                print("[\(s.startMs), \(s.endMs)]  (\(String(format: "%.2f", Double(s.endMs - s.startMs) / 1000.0)) s)")
            }
        } catch {
            logger.error("VAD failed: \(error)")
        }
    }
}
#endif
