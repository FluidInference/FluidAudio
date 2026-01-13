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
        var debugMode = false
        var outputFile: String?

        // VAD parameters
        var onset: Float?
        var offset: Float?
        var padOnset: Float?
        var padOffset: Float?
        var minDurationOn: Float?
        var minDurationOff: Float?
        var modelPath: String?

        // Parse remaining arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--debug":
                debugMode = true
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--onset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    onset = v
                    i += 1
                }
            case "--offset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    offset = v
                    i += 1
                }
            case "--pad-onset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    padOnset = v
                    i += 1
                }
            case "--pad-offset":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    padOffset = v
                    i += 1
                }
            case "--min-duration-on":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    minDurationOn = v
                    i += 1
                }
            case "--min-duration-off":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    minDurationOff = v
                    i += 1
                }
            case "--model-path":
                if i + 1 < arguments.count {
                    modelPath = arguments[i + 1]
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        print("Sortformer Streaming Diarization")
        print("   Audio: \(audioFile)")

        // Initialize Sortformer with default config (NVIDIA low latency: 1.04s)
        var config = SortformerConfig.default
        var postConfig = SortformerPostProcessingConfig.default
        config.debugMode = debugMode

        if let v = onset { postConfig.onsetThreshold = v }
        if let v = offset { postConfig.offsetThreshold = v }
        if let v = padOnset { postConfig.onsetPadSeconds = v }
        if let v = padOffset { postConfig.offsetPadSeconds = v }
        if let v = minDurationOn { postConfig.minDurationOn = v }
        if let v = minDurationOff { postConfig.minDurationOff = v }
        let diarizer = SortformerDiarizer(config: config, postProcessingConfig: postConfig)

        do {
            let loadStart = Date()
            let models: SortformerModels
            if let modelPath = modelPath {
                print("Loading models from local path: \(modelPath)")
                models = try await SortformerModels.load(
                    config: config, mainModelPath: URL(fileURLWithPath: modelPath))
            } else {
                print("Loading models from HuggingFace...")
                models = try await SortformerModels.loadFromHuggingFace(config: config, computeUnits: .cpuOnly)
            }
            print("Initializing...")
            diarizer.initialize(models: models)
            let loadTime = Date().timeIntervalSince(loadStart)
            print("Models loaded in \(String(format: "%.2f", loadTime))s")
        } catch {
            print("ERROR: Failed to initialize Sortformer: \(error)")
            exit(1)
        }

        // Load audio
        do {
            print("Loading audio...")

            let audioSamples = try AudioConverter(debug: config.debugMode).resampleAudioFile(
                path: audioFile)
            let duration = Float(audioSamples.count) / 16000.0
            print("Loaded \(audioSamples.count) samples (\(String(format: "%.1f", duration))s)")

            // Debug: Save and print first 10 samples for comparison
            if config.debugMode {
                print(
                    "[DEBUG] First 10 audio samples: \((0..<min(10, audioSamples.count)).map { String(format: "%.6f", audioSamples[$0]) }.joined(separator: ", "))"
                )
                let audioData = audioSamples.withUnsafeBytes { Data($0) }
                try? audioData.write(to: URL(fileURLWithPath: "swift_audio_16k.bin"))
                print("[DEBUG] Saved \(audioSamples.count) samples to swift_audio_16k.bin")
            }

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
            print("   Total frames: \(result.numFrames)")
            print("   Frame duration: \(String(format: "%.3f", result.config.frameDurationSeconds))s")

            // Extract segments
            let segments = result.segments.flatMap { $0 }
            print("   Found \(segments.count) segments")

            // Print segments
            print("\n--- Speaker Segments ---")
            for segment in segments {
                let start = String(format: "%.2f", segment.startTime)
                let end = String(format: "%.2f", segment.endTime)
                let dur = String(format: "%.2f", segment.duration)
                print("\(segment.speakerLabel): \(start)s - \(end)s (\(dur)s)")
            }

            // Print speaker probabilities summary
            print("\n--- Speaker Activity Summary ---")
            let numSpeakers = 4
            var speakerActivity = [Float](repeating: 0, count: numSpeakers)
            for frame in 0..<result.numFrames {
                for spk in 0..<numSpeakers {
                    let prob = result.framePredictions[frame * numSpeakers + spk]
                    if prob > 0.5 {
                        speakerActivity[spk] += result.config.frameDurationSeconds
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
                    "totalFrames": result.numFrames,
                    "frameDurationSeconds": result.config.frameDurationSeconds,
                    "segmentCount": segments.count,
                ]

                var segmentDicts: [[String: Any]] = []
                for segment in segments {
                    segmentDicts.append([
                        "speaker": segment.speakerLabel,
                        "speakerIndex": segment.speakerIndex,
                        "startTimeSeconds": segment.startTime,
                        "endTimeSeconds": segment.endTime,
                        "durationSeconds": segment.duration,
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
                --model-path <path>     Path to local CoreML model (.mlpackage or .mlmodelc)
                --debug                 Enable debug mode
                --output <file>         Save results to JSON file
                --onset <value>         Onset threshold for speech detection (default: 0.5)
                --offset <value>        Offset threshold for speech detection (default: 0.5)
                --pad-onset <value>     Padding before speech segments in seconds
                --pad-offset <value>    Padding after speech segments in seconds
                --min-duration-on <v>   Minimum speech segment duration in seconds
                --min-duration-off <v>  Minimum silence duration in seconds

            Examples:
                # Basic usage (downloads model from HuggingFace)
                fluidaudio sortformer audio.wav

                # With local model path
                fluidaudio sortformer audio.wav --model-path ./coreml_models/SortformerPipeline.mlpackage

                # Save results to file
                fluidaudio sortformer audio.wav --output results.json
            """
        )
    }
}
#endif
