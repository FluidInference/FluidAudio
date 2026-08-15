#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// Handler for the 'sortformer' command - Sortformer streaming diarization
enum SortformerCommand {
    private static let logger = AppLogger(category: "Sortformer")

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

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

        // SortformerConfig tuning fields
        var predScoreThreshold: Float?
        var silenceThreshold: Float?
        var scoresBoostLatest: Float?
        var strongBoostRate: Float?
        var weakBoostRate: Float?
        var minPosScoresRate: Float?
        var spkcacheSilFramesPerSpk: Int?
        var configName = "default"
        var palettized = false
        var offline = false
        var computeUnitsName: String?

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
            case "--config":
                if i + 1 < arguments.count {
                    configName = arguments[i + 1].lowercased()
                    i += 1
                }
            case "--offline":
                offline = true
            case "--palettized":
                palettized = true
            case "--compute-units":
                if i + 1 < arguments.count {
                    computeUnitsName = arguments[i + 1].lowercased()
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    predScoreThreshold = v
                    i += 1
                }
            case "--silence-threshold":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    silenceThreshold = v
                    i += 1
                }
            case "--scores-boost-latest":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    scoresBoostLatest = v
                    i += 1
                }
            case "--strong-boost-rate":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    strongBoostRate = v
                    i += 1
                }
            case "--weak-boost-rate":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    weakBoostRate = v
                    i += 1
                }
            case "--min-pos-scores-rate":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    minPosScoresRate = v
                    i += 1
                }
            case "--spkcache-sil-frames":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    spkcacheSilFramesPerSpk = v
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        if offline {
            var postConfig = DiarizerTimelineConfig.sortformerDefault
            if let v = onset { postConfig.onsetThreshold = v }
            if let v = offset { postConfig.offsetThreshold = v }
            if let v = padOnset { postConfig.onsetPadSeconds = v }
            if let v = padOffset { postConfig.offsetPadSeconds = v }
            if let v = minDurationOn { postConfig.minDurationOn = v }
            if let v = minDurationOff { postConfig.minDurationOff = v }
            await runOffline(
                audioFile: audioFile,
                modelPath: modelPath,
                palettized: palettized,
                computeUnits: parseComputeUnits(computeUnitsName),
                outputFile: outputFile,
                postConfig: postConfig)
            return
        }

        print("Sortformer Streaming Diarization")
        print("   Audio: \(audioFile)")

        // Select config (default = NVIDIA low latency ~1.04s). `--config efficient` = chunk_len=25 (~2s, higher throughput).
        var config: SortformerConfig
        switch configName {
        case "efficient":
            config = .efficientV2_1
        case "fast", "fastv2_1":
            config = .fastV2_1
        case "low", "balanced":
            config = .balancedV2_1
        case "high", "highcontext":
            config = .highContextV2_1
        default:
            config = .default
        }
        var postConfig = DiarizerTimelineConfig.sortformerDefault
        config.debugMode = debugMode
        if palettized { config.precision = .palettized }
        if let v = predScoreThreshold { config.predScoreThreshold = v }
        if let v = silenceThreshold { config.silenceThreshold = v }
        if let v = scoresBoostLatest { config.scoresBoostLatest = v }
        if let v = strongBoostRate { config.strongBoostRate = v }
        if let v = weakBoostRate { config.weakBoostRate = v }
        if let v = minPosScoresRate { config.minPosScoresRate = v }
        if let v = spkcacheSilFramesPerSpk { config.spkcacheSilFramesPerSpk = v }

        if let v = onset { postConfig.onsetThreshold = v }
        if let v = offset { postConfig.offsetThreshold = v }
        if let v = padOnset { postConfig.onsetPadSeconds = v }
        if let v = padOffset { postConfig.offsetPadSeconds = v }
        if let v = minDurationOn { postConfig.minDurationOn = v }
        if let v = minDurationOff { postConfig.minDurationOff = v }
        let diarizer = SortformerDiarizer(config: config, timelineConfig: postConfig)

        do {
            let loadStart = Date()
            let models: SortformerModels
            let computeUnits = Self.parseComputeUnits(computeUnitsName)
            if let modelPath = modelPath {
                print("Loading models from local path: \(modelPath)")
                models = try await SortformerModels.load(
                    config: config, mainModelPath: URL(fileURLWithPath: modelPath), computeUnits: computeUnits)
            } else {
                print("Loading models from HuggingFace...")
                models = try await SortformerModels.loadFromHuggingFace(
                    config: config, computeUnits: computeUnits ?? .cpuOnly)
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
                let debugPath = NSTemporaryDirectory() + "swift_audio_16k.bin"
                let audioData = audioSamples.withUnsafeBytes { Data($0) }
                try? audioData.write(to: URL(fileURLWithPath: debugPath))
                print("[DEBUG] Saved \(audioSamples.count) samples to \(debugPath)")
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
            print("   Total frames: \(result.numFinalizedFrames)")
            print("   Frame duration: \(String(format: "%.3f", result.config.frameDurationSeconds))s")

            // Extract segments
            let segments = result.speakers.values.flatMap { $0.finalizedSegments }
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
            let numSpeakers = result.config.numSpeakers
            var speakerActivity = [Float](repeating: 0, count: numSpeakers)
            let predictions = result.finalizedPredictions
            for frame in 0..<result.numFinalizedFrames {
                for spk in 0..<numSpeakers {
                    let idx = frame * numSpeakers + spk
                    if idx < predictions.count, predictions[idx] > 0.5 {
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
                    "totalFrames": result.numFinalizedFrames,
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

    /// Whole-file diarization via the fused offline Sortformer model (no streaming state).
    private static func runOffline(
        audioFile: String,
        modelPath: String?,
        palettized: Bool,
        computeUnits: MLComputeUnits?,
        outputFile: String?,
        postConfig: DiarizerTimelineConfig
    ) async {
        print("Sortformer Offline Diarization")
        print("   Audio: \(audioFile)")

        var offlineConfig = OfflineSortformerConfig.offlineV2_1
        if palettized { offlineConfig.precision = .palettized }
        let diarizer = OfflineSortformerDiarizer(config: offlineConfig, timelineConfig: postConfig)

        do {
            let loadStart = Date()
            if let modelPath = modelPath {
                print("Loading offline model from local path: \(modelPath)")
                try await diarizer.initialize(
                    modelPath: URL(fileURLWithPath: modelPath), computeUnits: computeUnits ?? .all)
            } else {
                print("Loading offline model from HuggingFace...")
                try await diarizer.initializeFromHuggingFace(computeUnits: computeUnits ?? .all)
            }
            print("Model loaded in \(String(format: "%.2f", Date().timeIntervalSince(loadStart)))s")
        } catch {
            print("ERROR: Failed to initialize offline Sortformer: \(error)")
            exit(1)
        }

        do {
            print("Loading audio...")
            let audioSamples = try AudioConverter().resampleAudioFile(path: audioFile)
            let duration = Float(audioSamples.count) / 16000.0
            print("Loaded \(audioSamples.count) samples (\(String(format: "%.1f", duration))s)")

            print("Processing...")
            fflush(stdout)
            let startTime = Date()
            let result = try diarizer.processComplete(audioSamples)
            let processingTime = Date().timeIntervalSince(startTime)
            let rtfx = duration / Float(processingTime)

            print("Processing completed in \(String(format: "%.2f", processingTime))s")
            print("   Real-time factor (RTFx): \(String(format: "%.1f", rtfx))x")
            print("   Total frames: \(result.numFinalizedFrames)")

            let segments = result.speakers.values.flatMap { $0.finalizedSegments }
            print("   Found \(segments.count) segments")
            print("\n--- Speaker Segments ---")
            for segment in segments.sorted() {
                let start = String(format: "%.2f", segment.startTime)
                let end = String(format: "%.2f", segment.endTime)
                let dur = String(format: "%.2f", segment.duration)
                print("\(segment.speakerLabel): \(start)s - \(end)s (\(dur)s)")
            }

            if let outputFile = outputFile {
                var segmentDicts: [[String: Any]] = []
                for segment in segments.sorted() {
                    segmentDicts.append([
                        "speaker": segment.speakerLabel,
                        "speakerIndex": segment.speakerIndex,
                        "startTimeSeconds": segment.startTime,
                        "endTimeSeconds": segment.endTime,
                        "durationSeconds": segment.duration,
                    ])
                }
                let output: [String: Any] = [
                    "audioFile": audioFile,
                    "mode": "offline",
                    "durationSeconds": duration,
                    "processingTimeSeconds": processingTime,
                    "rtfx": rtfx,
                    "totalFrames": result.numFinalizedFrames,
                    "segmentCount": segments.count,
                    "segments": segmentDicts,
                ]
                let jsonData = try JSONSerialization.data(
                    withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
                try jsonData.write(to: URL(fileURLWithPath: outputFile))
                print("Results saved to: \(outputFile)")
            }
        } catch {
            print("ERROR: Failed to process audio: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        let usage = """

            Sortformer Command Usage:
                fluidaudio sortformer <audio_file> [options]

            Options:
                --model-path <path>         Path to local CoreML model (.mlpackage or .mlmodelc)
                --offline                   Whole-file mode: fused offline model (no streaming state)
                --palettized                Use the 6-bit palettized model set (~2.5x smaller)
                --compute-units <units>     CoreML compute units: cpu | cpuAndGPU | ane | all.
                                            Default auto-selects; use cpu on pre-M1 devices
                                            (e.g. A14) where .all is slow or incompatible.
                --debug                     Enable debug mode
                --output <file>             Save results to JSON file
                --onset <value>             Onset threshold for speech detection (default: 0.5)
                --offset <value>            Offset threshold for speech detection (default: 0.5)
                --pad-onset <value>         Padding before speech segments in seconds
                --pad-offset <value>        Padding after speech segments in seconds
                --min-duration-on <v>       Minimum speech segment duration in seconds
                --min-duration-off <v>      Minimum silence duration in seconds
                --threshold <0-1>           Prediction score threshold (default: 0.25)
                --silence-threshold <0-1>   Silence detection threshold (default: 0.2)
                --scores-boost-latest <fl>  Boost factor for latest frames (default: 0.05)
                --strong-boost-rate <0-1>   Strong boost rate for top-k selection (default: 0.75)
                --weak-boost-rate <fl>      Weak boost rate (default: 1.5)
                --min-pos-scores-rate <0-1> Minimum positive scores rate (default: 0.5)
                --spkcache-sil-frames <n>   Silence frames per speaker in cache (default: 3)

            Examples:
                # Basic usage (downloads model from HuggingFace)
                fluidaudio sortformer audio.wav

                # With local model path
                fluidaudio sortformer audio.wav --model-path ./coreml_models/SortformerPipeline.mlpackage

                # Offline whole-file mode (fused model, fastest when the full audio is available)
                fluidaudio sortformer audio.wav --offline

                # Tune streaming parameters
                fluidaudio sortformer audio.wav --threshold 0.3 --silence-threshold 0.15

                # Save results to file
                fluidaudio sortformer audio.wav --output results.json

                # Smaller, CPU-only load for older devices (e.g. A14)
                fluidaudio sortformer audio.wav --palettized --compute-units cpu
            """
        fputs(usage, stderr)
        fflush(stderr)
    }

    /// Map a `--compute-units` CLI value to `MLComputeUnits`. Returns `nil` for unknown/absent
    /// input so callers fall back to their own per-path default.
    private static func parseComputeUnits(_ name: String?) -> MLComputeUnits? {
        switch name {
        case "cpu", "cpuonly":
            return .cpuOnly
        case "cpuandgpu", "cpugpu":
            return .cpuAndGPU
        case "cpuandneuralengine", "cpuane", "ane":
            return .cpuAndNeuralEngine
        case "all":
            return .all
        default:
            return nil
        }
    }
}
#endif
