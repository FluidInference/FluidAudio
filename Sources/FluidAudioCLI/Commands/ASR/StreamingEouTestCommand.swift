#if os(macOS)
import FluidAudio
import Foundation
import OSLog

/// Test command for streaming EOU model with cache-aware encoder
public enum StreamingEouTestCommand {

    private static let logger = AppLogger(category: "StreamingEouTest")

    public static func run(arguments: [String]) async {
        // Check for help flag
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var modelsPath: String?
        var audioFile: String?
        var chunkDurationMs: Double = 500  // Default 500ms chunks

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--models":
                if i + 1 < arguments.count {
                    modelsPath = arguments[i + 1]
                    i += 1
                }
            case "--audio":
                if i + 1 < arguments.count {
                    audioFile = arguments[i + 1]
                    i += 1
                }
            case "--chunk-duration":
                if i + 1 < arguments.count {
                    chunkDurationMs = Double(arguments[i + 1]) ?? 500
                    i += 1
                }
            default:
                // Try as positional audio file
                if !arguments[i].hasPrefix("-") && audioFile == nil {
                    audioFile = arguments[i]
                }
            }
            i += 1
        }

        guard let modelsPath = modelsPath else {
            print("Error: --models path is required")
            printUsage()
            return
        }

        guard let audioFile = audioFile else {
            print("Error: audio file path is required")
            printUsage()
            return
        }

        print("=== Streaming EOU Test ===")
        print("Models: \(modelsPath)")
        print("Audio: \(audioFile)")
        print("Chunk: \(Int(chunkDurationMs))ms")
        print()

        do {
            // Load streaming models
            print("Loading streaming models...")
            let modelsURL = URL(fileURLWithPath: modelsPath)
            let manager = StreamingEouAsrManager()
            try await manager.initializeFromLocalPath(modelsURL)
            print("Models loaded successfully")

            // Load audio file
            print("\nLoading audio file...")
            let audioURL = URL(fileURLWithPath: audioFile)
            let audioSamples = try await loadAudioFile(audioURL)
            let audioDuration = Double(audioSamples.count) / 16000.0
            print("Audio loaded: \(audioSamples.count) samples (\(String(format: "%.2f", audioDuration))s)")

            // Process in chunks
            let chunkSamples = Int(chunkDurationMs / 1000.0 * 16000.0)
            let numChunks = (audioSamples.count + chunkSamples - 1) / chunkSamples

            print("\nProcessing \(numChunks) chunks of \(chunkSamples) samples each...")
            print()

            var fullText = ""
            var totalProcessingTime: TimeInterval = 0
            let overallStart = Date()

            for chunkIdx in 0..<numChunks {
                let startSample = chunkIdx * chunkSamples
                let endSample = min(startSample + chunkSamples, audioSamples.count)
                let chunk = Array(audioSamples[startSample..<endSample])
                let isFinal = chunkIdx == numChunks - 1

                let chunkStart = Date()
                let result = try await manager.processChunk(chunk, isFinal: isFinal)
                let chunkTime = Date().timeIntervalSince(chunkStart)
                totalProcessingTime += chunkTime

                if !result.text.isEmpty {
                    fullText += result.text
                    print(
                        "Chunk \(chunkIdx + 1)/\(numChunks): \"\(result.text)\" "
                            + "(RTFx: \(String(format: "%.1f", result.rtfx))x, "
                            + "time: \(String(format: "%.0f", chunkTime * 1000))ms)"
                    )
                } else {
                    print(
                        "Chunk \(chunkIdx + 1)/\(numChunks): [no tokens] "
                            + "(RTFx: \(String(format: "%.1f", result.rtfx))x)"
                    )
                }

                if result.eouDetected {
                    print("  [EOU detected]")
                }
            }

            let overallTime = Date().timeIntervalSince(overallStart)
            let overallRtfx = audioDuration / overallTime

            print()
            print("=== Results ===")
            print("Full text: \"\(fullText.trimmingCharacters(in: .whitespaces))\"")
            print("Audio duration: \(String(format: "%.2f", audioDuration))s")
            print("Processing time: \(String(format: "%.2f", overallTime))s")
            print("Overall RTFx: \(String(format: "%.1f", overallRtfx))x")
            print("Avg chunk time: \(String(format: "%.0f", (totalProcessingTime / Double(numChunks)) * 1000))ms")

        } catch {
            print("Error: \(error)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio streaming-eou-test --models <path> --audio <file> [options]

            Options:
              --models <path>        Path to streaming model directory (required)
              --audio <file>         Path to audio file (required)
              --chunk-duration <ms>  Chunk duration in milliseconds (default: 500)
              --help, -h             Show this help message

            Example:
              fluidaudio streaming-eou-test \\
                --models ./parakeet_eou_streaming_coreml \\
                --audio test.wav \\
                --chunk-duration 500
            """)
    }

    private static func loadAudioFile(_ url: URL) async throws -> [Float] {
        // Use AudioConverter to load and convert audio
        let converter = AudioConverter()
        let audio = try converter.resampleAudioFile(url)
        return audio
    }
}
#endif
