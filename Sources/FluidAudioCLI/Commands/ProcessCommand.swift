#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

nonisolated(unsafe) var standardError = FileHandle.standardError

/// Handler for the 'process' command - processes a single audio file
enum ProcessCommand {
    private static let logger = AppLogger(category: "Process")
    static func run(arguments: [String]) async {
        guard !arguments.isEmpty else {
            fputs("ERROR: No audio file specified\n", stderr)
            fflush(stderr)
            logger.error("No audio file specified")
            printUsage()
            exit(1)
        }

        let audioFile = arguments[0]
        var mode = "streaming"  // Default to streaming
        var threshold: Float = 0.7045655  // PyAnnote community-1 default
        var chunkDuration: Float = 10.0
        var chunkOverlap: Float = 0.0
        var debugMode = false
        var outputFile: String?
        var rttmFile: String?
        var embeddingExportPath: String?

        // Parse remaining arguments
        var i = 1
        while i < arguments.count {
            switch arguments[i] {
            case "--mode":
                if i + 1 < arguments.count {
                    mode = arguments[i + 1]
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.8
                    i += 1
                }
            case "--chunk-seconds":
                if i + 1 < arguments.count {
                    chunkDuration = Float(arguments[i + 1]) ?? 10.0
                    i += 1
                }
            case "--overlap-seconds":
                if i + 1 < arguments.count {
                    chunkOverlap = Float(arguments[i + 1]) ?? 0.0
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--rttm":
                if i + 1 < arguments.count {
                    rttmFile = arguments[i + 1]
                    i += 1
                }
            case "--export-embeddings":
                if i + 1 < arguments.count {
                    embeddingExportPath = arguments[i + 1]
                    i += 1
                }
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Validate mode
        guard mode == "streaming" || mode == "offline" else {
            fputs("ERROR: Invalid mode: \(mode)\n", stderr)
            fflush(stderr)
            logger.error("Invalid mode: \(mode). Must be 'streaming' or 'offline'")
            printUsage()
            exit(1)
        }

        logger.info("🎵 Processing audio file (\(mode.uppercased()) MODE): \(audioFile)")
        logger.info("   Clustering threshold: \(threshold)")

        if mode == "streaming" {
            // Streaming mode - use DiarizerManager
            let config = DiarizerConfig(
                clusteringThreshold: threshold,
                debugMode: debugMode,
                chunkDuration: chunkDuration,
                chunkOverlap: chunkOverlap
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
                logger.info("   Detected \(result.speakerDatabase?.count ?? 0) speakers")

                // Create output
                let output = ProcessingResult(
                    audioFile: audioFile,
                    durationSeconds: duration,
                    processingTimeSeconds: processingTime,
                    realTimeFactor: rtfx,
                    segments: result.segments,
                    speakerCount: result.speakerDatabase?.count ?? 0,
                    config: config,
                    metrics: nil,
                    timings: result.timings
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
        } else {
            // Offline mode - use OfflineDiarizerManager
            do {
                let modelDir = OfflineDiarizerModels.defaultModelsDirectory()
                let offlineConfig = OfflineDiarizerConfig(
                    clusteringThreshold: Double(threshold),
                    embeddingExportPath: embeddingExportPath
                )
                let manager = OfflineDiarizerManager(config: offlineConfig)

                let models = try await OfflineDiarizerModels.load(from: modelDir)
                manager.initialize(models: models)

                logger.info("Offline manager initialized")

                // Load and process audio file without materializing the full sample buffer.
                let audioURL = URL(fileURLWithPath: audioFile)
                let factory = StreamingAudioSourceFactory()
                let targetSampleRate = offlineConfig.segmentation.sampleRate
                let diskSourceResult = try factory.makeDiskBackedSource(
                    from: audioURL,
                    targetSampleRate: targetSampleRate
                )
                let diskSource = diskSourceResult.source
                defer { diskSource.cleanup() }
                let loadDurationText = String(format: "%.2f", diskSourceResult.loadDuration)
                logger.info(
                    "Prepared disk-backed audio source: \(diskSource.sampleCount) samples (\(loadDurationText)s)")

                let startTime = Date()
                let result = try await manager.process(
                    audioSource: diskSource,
                    audioLoadingSeconds: diskSourceResult.loadDuration
                )
                let processingTime = Date().timeIntervalSince(startTime)

                let durationSeconds = Double(diskSource.sampleCount) / Double(targetSampleRate)
                let rtfx = durationSeconds / processingTime

                logger.info("Diarization completed in \(String(format: "%.1f", processingTime))s")
                logger.info("   Real-time factor (RTFx): \(String(format: "%.2f", rtfx))x")
                logger.info("   Found \(result.segments.count) segments")

                let speakerCount = Set(result.segments.map { $0.speakerId }).count
                logger.info("   Detected \(speakerCount) speakers")

                var metrics: DiarizationMetrics?
                if let rttmFile = rttmFile {
                    do {
                        let groundTruth = try RTTMParser.loadSegments(from: rttmFile)
                        metrics = DiarizationMetricsCalculator.offlineMetrics(
                            predicted: result.segments,
                            groundTruth: groundTruth,
                            frameSize: 0.01,
                            audioDurationSeconds: durationSeconds,
                            logger: logger
                        )
                    } catch {
                        logger.error("Failed to compute offline metrics: \(error.localizedDescription)")
                    }
                }

                // Create simplified output for offline mode
                let output = ProcessingResult(
                    audioFile: audioFile,
                    durationSeconds: Float(durationSeconds),
                    processingTimeSeconds: processingTime,
                    realTimeFactor: Float(rtfx),
                    segments: result.segments,
                    speakerCount: speakerCount,
                    config: nil,
                    metrics: metrics,
                    timings: result.timings
                )

                // Output results
                if let outputFile = outputFile {
                    try await ResultsFormatter.saveResults(output, to: outputFile)
                    logger.info("💾 Results saved to: \(outputFile)")
                } else {
                    await ResultsFormatter.printResults(output)
                }

            } catch {
                fputs("ERROR: Failed to process audio file (offline mode): \(error)\n", stderr)
                fflush(stderr)
                logger.error("Failed to process audio file (offline mode): \(error)")
                exit(1)
            }
        }
    }

    private static func printUsage() {
        logger.info(
            """

            Process Command Usage:
                fluidaudio process <audio_file> [options]

            Options:
                --mode <streaming|offline>  Diarization mode (default: streaming)
                --threshold <float>          Clustering threshold (default: 0.7045655, pyannote community-1)
                --debug                      Enable debug mode
                --output <file>              Save results to file instead of stdout
                --rttm <file>                Compute offline DER/JER metrics against RTTM annotations
                --export-embeddings <file>   Export embeddings to JSON for debugging (offline mode only)


            Examples:
                # Streaming mode (default)
                fluidaudio process audio.wav --output results.json

                # Offline mode with VBx clustering (default threshold 0.7045655)
                fluidaudio process audio.wav --mode offline --output results.json

                # Offline mode with embedding export for debugging
                fluidaudio process audio.wav --mode offline --export-embeddings embeddings.json
            """
        )
    }
}
#endif
