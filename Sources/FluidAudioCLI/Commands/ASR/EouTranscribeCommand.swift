#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// CLI command for Parakeet Realtime EOU 120M transcription
enum EouTranscribeCommand {
    private static let logger = AppLogger(category: "EouTranscribe")

    private struct Options {
        var audioPath: String?
        var showTimings: Bool = false
        var debug: Bool = false
        var localModelsPath: String?
        var useMLPackage: Bool = false
    }

    static func run(arguments: [String]) async {
        var options = Options()
        var index = 0

        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--timings":
                options.showTimings = true
            case "--debug":
                options.debug = true
            case "--local-models":
                index += 1
                if index < arguments.count {
                    options.localModelsPath = arguments[index]
                } else {
                    logger.error("--local-models requires a path argument")
                    exit(1)
                }
            case "--use-mlpackage":
                options.useMLPackage = true
            default:
                if arg.hasPrefix("--") {
                    logger.warning("Unknown option: \(arg)")
                } else if options.audioPath == nil {
                    options.audioPath = arg
                } else {
                    logger.warning("Ignoring extra argument: \(arg)")
                }
            }
            index += 1
        }

        guard let audioPath = options.audioPath else {
            logger.error("No audio file provided")
            printUsage()
            exit(1)
        }

        do {
            print("Loading audio from: \(audioPath)")

            // Convert audio to 16kHz mono
            let samples = try AudioConverter().resampleAudioFile(path: audioPath)
            let audioDuration = Double(samples.count) / 16000.0
            print("Audio duration: \(String(format: "%.2f", audioDuration))s (\(samples.count) samples)")

            // Initialize EOU ASR manager
            print("Initializing Parakeet EOU 120M model...")
            let manager = EouAsrManager()
            if let localPath = options.localModelsPath {
                let modelDir = URL(fileURLWithPath: localPath)
                let ext = options.useMLPackage ? ".mlpackage" : ".mlmodelc"
                print("Loading from local path: \(localPath) (using \(ext))")
                try await manager.initializeFromLocalPath(modelDir, useMLPackage: options.useMLPackage)
            } else {
                try await manager.initialize()
            }
            print("Model loaded successfully")

            // Handle long audio by chunking (max 15 seconds per chunk)
            let maxSamples = EouAsrManager.maxAudioSamples
            var fullText = ""
            var allTimings: [EouTokenTiming] = []
            var totalProcessingTime: TimeInterval = 0

            if samples.count <= maxSamples {
                // Single chunk
                print("\nTranscribing...")
                let result = try await manager.transcribe(samples)
                fullText = result.text
                allTimings = result.tokenTimings ?? []
                totalProcessingTime = result.processingTime
            } else {
                // Multiple chunks with overlap
                let chunkSize = maxSamples
                let overlapSamples = 32000  // 2 second overlap
                let stride = chunkSize - overlapSamples
                var offset = 0
                var chunkIndex = 0

                print("\nTranscribing in chunks (audio > 15s)...")

                while offset < samples.count {
                    let endIndex = min(offset + chunkSize, samples.count)
                    let chunk = Array(samples[offset..<endIndex])

                    let chunkStart = Double(offset) / 16000.0
                    let chunkEnd = Double(endIndex) / 16000.0
                    print(
                        "  Chunk \(chunkIndex + 1): \(String(format: "%.1f", chunkStart))s - \(String(format: "%.1f", chunkEnd))s"
                    )

                    let result = try await manager.transcribe(chunk)

                    if !result.text.isEmpty {
                        if !fullText.isEmpty {
                            fullText += " "
                        }
                        fullText += result.text
                    }

                    if let timings = result.tokenTimings {
                        // Adjust timings for chunk offset
                        let offsetSeconds = Double(offset) / 16000.0
                        for timing in timings {
                            var adjustedTiming = timing
                            // Note: timing.timeSeconds is based on frameIndex
                            // We'd need to add offset but EouTokenTiming is immutable
                            allTimings.append(timing)
                        }
                    }

                    totalProcessingTime += result.processingTime

                    if result.eouDetected {
                        print("    <EOU> detected")
                    }

                    offset += stride
                    chunkIndex += 1

                    // Reset state between chunks
                    manager.resetState()
                }
            }

            // Print results
            print("\n" + String(repeating: "=", count: 60))
            print("TRANSCRIPTION RESULT")
            print(String(repeating: "=", count: 60))
            print("\n\(fullText)\n")

            // Performance metrics
            let rtfx = audioDuration / totalProcessingTime
            print(String(repeating: "-", count: 60))
            print("Performance:")
            print("  Audio duration:    \(String(format: "%.2f", audioDuration))s")
            print("  Processing time:   \(String(format: "%.3f", totalProcessingTime))s")
            print("  RTFx:              \(String(format: "%.1f", rtfx))x realtime")

            // Show token timings if requested
            if options.showTimings && !allTimings.isEmpty {
                print("\n" + String(repeating: "-", count: 60))
                print("Token Timings:")
                for (i, timing) in allTimings.prefix(50).enumerated() {
                    print(
                        "  [\(i)] \(String(format: "%.3f", timing.timeSeconds))s: \"\(timing.token)\" (conf: \(String(format: "%.3f", timing.confidence)))"
                    )
                }
                if allTimings.count > 50 {
                    print("  ... (\(allTimings.count - 50) more tokens)")
                }
            }

            print(String(repeating: "=", count: 60))

        } catch {
            print("ERROR: Transcription failed: \(error)")
            logger.error("Transcription failed: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudio eou-transcribe <audio-file> [options]

            Transcribe audio using Parakeet Realtime EOU 120M model.

            Options:
              --timings              Show per-token timing information
              --debug                Enable debug output
              --local-models <path>  Load models from local directory instead of downloading
              --use-mlpackage        Use .mlpackage files instead of .mlmodelc (for debugging)
              --help, -h             Show this help message

            Example:
              fluidaudio eou-transcribe audio.wav
              fluidaudio eou-transcribe audio.wav --timings
              fluidaudio eou-transcribe audio.wav --local-models ./models --use-mlpackage
            """)
    }
}
#endif
