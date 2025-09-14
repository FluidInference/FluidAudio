#if os(macOS)
import AVFoundation
import FluidAudio

/// Handler for the 'process' command - processes a single audio file
enum ProcessCommand {
    private static let logger = AppLogger(category: "Process")
    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var threshold: Float = 0.7
        var debugMode = false
        var outputFile: String?

        // Parse remaining arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.8
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

        logger.info("🎵 Processing audio file: \(audioFile)")
        logger.info("   Clustering threshold: \(threshold)")

        let config = DiarizerConfig(
            clusteringThreshold: threshold,
            debugMode: debugMode
        )

        let manager = DiarizerManager(config: config)

        do {
            let models = try await DiarizerModels.downloadIfNeeded()
            manager.initialize(models: models)
            logger.info("Models initialized")
        } catch {
            logger.error("Failed to initialize models: \(error)")
            exit(1)
        }

        // Load and process audio file
        do {
            let audioSamples = try AudioConverter().resampleAudioFile(path: audioFile)
            logger.info("Loaded audio: \(audioSamples.count) samples")

            let startTime = Date()
            let result = try manager.performCompleteDiarization(
                audioSamples, sampleRate: 16000)
            let processingTime = Date().timeIntervalSince(startTime)

            let duration = Float(audioSamples.count) / 16000.0
            let rtfx = duration / Float(processingTime)

            logger.info("Diarization completed in \(String(format: "%.1f", processingTime))s")
            logger.info("   Real-time factor (RTFx): \(String(format: "%.2f", rtfx))x")
            logger.info("   Found \(result.segments.count) segments")
            logger.info("   Detected \(result.speakerDatabase?.count ?? 0) speakers (total), mapped: TBD")

            // Create output
            let output = ProcessingResult(
                audioFile: audioFile,
                durationSeconds: duration,
                processingTimeSeconds: processingTime,
                realTimeFactor: rtfx,
                segments: result.segments,
                speakerCount: result.speakerDatabase?.count ?? 0,
                config: config
            )

            // Output results
            if let outputFile = outputFile {
                try await ResultsFormatter.saveResults(output, to: outputFile)
                logger.info("💾 Results saved to: \(outputFile)")
            } else {
                await ResultsFormatter.printResults(output)
            }

        } catch {
            logger.error("Failed to process audio file: \(error)")
            exit(1)
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Process Command Usage:
                fluidaudio process <audio_file> [options]

            Options:
                --threshold <float>    Clustering threshold (default: 0.8)
                --debug               Enable debug mode
                --output <file>       Save results to file instead of stdout

            Example:
                fluidaudio process audio.wav --threshold 0.5 --output results.json
            """
        )
    }
}
#endif
