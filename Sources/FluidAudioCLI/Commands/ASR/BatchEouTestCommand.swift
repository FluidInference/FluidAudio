#if os(macOS)
import FluidAudio
import Foundation
import OSLog

/// Test command for batch EOU model
public enum BatchEouTestCommand {

    private static let logger = AppLogger(category: "BatchEouTest")

    public static func run(arguments: [String]) async {
        // Check for help flag
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var modelsPath: String?
        var audioFile: String?
        var useMLPackage = false

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
            case "--use-mlpackage":
                useMLPackage = true
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

        print("=== Batch EOU Test ===")
        print("Models: \(modelsPath)")
        print("Audio: \(audioFile)")
        print()

        do {
            // Load batch models
            print("Loading batch models...")
            let modelsURL = URL(fileURLWithPath: modelsPath)
            let manager = BatchEouAsrManager()
            try await manager.initializeFromLocalPath(modelsURL, useMLPackage: useMLPackage)
            print("Models loaded successfully")

            // Load audio file
            print("\nLoading audio file...")
            let audioURL = URL(fileURLWithPath: audioFile)
            let audioSamples = try await loadAudioFile(audioURL)
            let audioDuration = Double(audioSamples.count) / 16000.0
            print("Audio loaded: \(audioSamples.count) samples (\(String(format: "%.2f", audioDuration))s)")
            
            // Print audio stats
            let minVal = audioSamples.min() ?? 0
            let maxVal = audioSamples.max() ?? 0
            let meanVal = audioSamples.reduce(0, +) / Float(audioSamples.count)
            print("Audio stats: min=\(minVal), max=\(maxVal), mean=\(meanVal)")

            // Check if audio exceeds max length
            if audioSamples.count > BatchEouAsrManager.maxAudioSamples {
                print(
                    "Warning: Audio exceeds max length (\(BatchEouAsrManager.maxAudioSamples) samples = 15s). Will fail."
                )
            }

            // Transcribe
            print("\nTranscribing...")
            let startTime = Date()
            let result = try await manager.transcribe(audioSamples)
            let totalTime = Date().timeIntervalSince(startTime)

            print()
            print("=== Results ===")
            print("Text: \"\(result.text)\"")
            print("Confidence: \(String(format: "%.3f", result.confidence))")
            print("Audio duration: \(String(format: "%.2f", result.audioDuration))s")
            print("Processing time: \(String(format: "%.2f", totalTime))s")
            print("RTFx: \(String(format: "%.1f", result.rtfx))x")
            print("EOU detected: \(result.eouDetected)")

            if let timings = result.tokenTimings {
                print("\nToken timings (\(timings.count) tokens):")
                for timing in timings.prefix(20) {
                    print(
                        "  [\(timing.frameIndex)] \"\(timing.token)\" (conf: \(String(format: "%.3f", timing.confidence)))"
                    )
                }
                if timings.count > 20 {
                    print("  ... and \(timings.count - 20) more tokens")
                }
            }

        } catch {
            print("Error: \(error)")
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio batch-eou-test --models <path> --audio <file>

            Options:
              --models <path>        Path to batch model directory (required)
              --audio <file>         Path to audio file (required)
              --use-mlpackage        Use .mlpackage extension instead of .mlmodelc
              --help, -h             Show this help message

            Example:
              fluidaudio batch-eou-test \\
                --models ./parakeet_eou_coreml \\
                --audio test.wav
            """)
    }

    private static func loadAudioFile(_ url: URL) async throws -> [Float] {
        let converter = AudioConverter()
        let audio = try converter.resampleAudioFile(url)
        return audio
    }
}
#endif
