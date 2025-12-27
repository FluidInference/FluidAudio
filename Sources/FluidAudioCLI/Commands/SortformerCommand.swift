#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Handler for the 'sortformer' command - Sortformer streaming diarization
enum SortformerCommand {
    private static let logger = AppLogger(category: "Sortformer")

    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            fputs("ERROR: No audio file specified\n", stderr)
            fflush(stderr)
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var preprocessorPath: String?
        var modelPath: String?
        var debugMode = false
        var outputFile: String?

        // Parse remaining arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--preprocessor":
                if i + 1 < arguments.count {
                    preprocessorPath = arguments[i + 1]
                    i += 1
                }
            case "--model":
                if i + 1 < arguments.count {
                    modelPath = arguments[i + 1]
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Default model paths (relative to working directory)
        let defaultPreprocessor = "Streaming-Sortformer-Conversion/coreml_models/SortformerPreprocessor.mlpackage"
        let defaultModel = "Streaming-Sortformer-Conversion/coreml_models/Sortformer.mlpackage"

        let preprocessorURL = URL(fileURLWithPath: preprocessorPath ?? defaultPreprocessor)
        let modelURL = URL(fileURLWithPath: modelPath ?? defaultModel)

        print("Sortformer Streaming Diarization")
        print("   Audio: \(audioFile)")
        print("   Preprocessor: \(preprocessorURL.path)")
        print("   Model: \(modelURL.path)")

        // Check models exist
        guard FileManager.default.fileExists(atPath: preprocessorURL.path) else {
            print("ERROR: Preprocessor model not found: \(preprocessorURL.path)")
            exit(1)
        }
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("ERROR: Main model not found: \(modelURL.path)")
            exit(1)
        }

        // Initialize Sortformer
        let config = SortformerConfig(debugMode: debugMode)
        let diarizer = SortformerDiarizer(config: config)

        do {
            print("Loading models...")
            let loadStart = Date()
            try await diarizer.initialize(
                preprocessorPath: preprocessorURL,
                mainModelPath: modelURL
            )
            let loadTime = Date().timeIntervalSince(loadStart)
            print("Models loaded in \(String(format: "%.2f", loadTime))s")
        } catch {
            print("ERROR: Failed to initialize Sortformer: \(error)")
            exit(1)
        }

        // Load audio
        do {
            print("Loading audio...")
            let audioSamples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Float(audioSamples.count) / 16000.0
            print("Loaded \(audioSamples.count) samples (\(String(format: "%.1f", duration))s)")

            // Process with progress
            print("Processing...")
            fflush(stdout)
            let startTime = Date()
            var lastProgressPrint = Date()
            let result = try diarizer.processComplete(audioSamples) { processed, total, chunks in
                let now = Date()
                if now.timeIntervalSince(lastProgressPrint) >= 2.0 {
                    let percent = Float(processed) / Float(total) * 100
                    let elapsed = now.timeIntervalSince(startTime)
                    let processedSeconds = Float(processed) / 16000.0
                    let currentRtfx = processedSeconds / Float(elapsed)
                    print(
                        "   Progress: \(String(format: "%.1f", percent))% | Chunks: \(chunks) | RTFx: \(String(format: "%.1f", currentRtfx))x"
                    )
                    fflush(stdout)
                    lastProgressPrint = now
                }
            }
            let processingTime = Date().timeIntervalSince(startTime)

            let rtfx = duration / Float(processingTime)
            print("Processing completed in \(String(format: "%.2f", processingTime))s")
            print("   Real-time factor (RTFx): \(String(format: "%.1f", rtfx))x")
            print("   Total frames: \(result.totalFrames)")
            print("   Frame duration: \(String(format: "%.3f", result.frameDurationSeconds))s")

            // Extract segments
            let segments = result.extractSegments(threshold: 0.5)
            print("   Found \(segments.count) segments")

            // Print segments
            print("\n--- Speaker Segments ---")
            for segment in segments {
                let start = String(format: "%.2f", segment.startTimeSeconds)
                let end = String(format: "%.2f", segment.endTimeSeconds)
                let dur = String(format: "%.2f", segment.durationSeconds)
                print("\(segment.speakerLabel): \(start)s - \(end)s (\(dur)s)")
            }

            // Print speaker probabilities summary
            print("\n--- Speaker Activity Summary ---")
            let numSpeakers = 4
            var speakerActivity = [Float](repeating: 0, count: numSpeakers)
            for frame in 0..<result.totalFrames {
                for spk in 0..<numSpeakers {
                    let prob = result.allProbabilities[frame * numSpeakers + spk]
                    if prob > 0.5 {
                        speakerActivity[spk] += result.frameDurationSeconds
                    }
                }
            }
            for spk in 0..<numSpeakers {
                let activeTime = String(format: "%.1f", speakerActivity[spk])
                let percent = String(format: "%.1f", (speakerActivity[spk] / duration) * 100)
                print("Speaker_\(spk): \(activeTime)s active (\(percent)%)")
            }

            // Save output if requested
            if let outputFile = outputFile {
                var output: [String: Any] = [
                    "audioFile": audioFile,
                    "durationSeconds": duration,
                    "processingTimeSeconds": processingTime,
                    "rtfx": rtfx,
                    "totalFrames": result.totalFrames,
                    "frameDurationSeconds": result.frameDurationSeconds,
                    "segmentCount": segments.count,
                ]

                var segmentDicts: [[String: Any]] = []
                for segment in segments {
                    segmentDicts.append([
                        "speaker": segment.speakerLabel,
                        "speakerIndex": segment.speakerIndex,
                        "startTimeSeconds": segment.startTimeSeconds,
                        "endTimeSeconds": segment.endTimeSeconds,
                        "durationSeconds": segment.durationSeconds,
                    ])
                }
                output["segments"] = segmentDicts

                let jsonData = try JSONSerialization.data(
                    withJSONObject: output,
                    options: [.prettyPrinted, .sortedKeys]
                )
                try jsonData.write(to: URL(fileURLWithPath: outputFile))
                print("Results saved to: \(outputFile)")
            }

        } catch {
            print("ERROR: Failed to process audio: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Sortformer Command Usage:
                fluidaudio sortformer <audio_file> [options]

            Options:
                --preprocessor <path>   Path to SortformerPreprocessor.mlpackage
                --model <path>          Path to Sortformer.mlpackage
                --debug                 Enable debug mode
                --output <file>         Save results to JSON file

            Examples:
                # Basic usage (uses default model paths)
                fluidaudio sortformer audio.wav

                # With custom model paths
                fluidaudio sortformer audio.wav --preprocessor ./models/SortformerPreprocessor.mlpackage --model ./models/Sortformer.mlpackage

                # Save results to file
                fluidaudio sortformer audio.wav --output results.json
            """
        )
    }
}
#endif
