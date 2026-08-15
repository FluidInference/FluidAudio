#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

nonisolated(unsafe) var standardError = FileHandle.standardError

/// Handler for the 'process' command - processes a single audio file
enum ProcessCommand {
    private static let logger = AppLogger(category: "Process")

    static func run(arguments: [String]) async {
        if arguments.contains("--help") {
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
        let remaining = Array(arguments.dropFirst())
        let parsed = parseArguments(remaining)

        guard parsed.mode == "streaming" || parsed.mode == "offline" else {
            fputs("ERROR: Invalid mode: \(parsed.mode)\n", stderr)
            fflush(stderr)
            logger.error("Invalid mode: \(parsed.mode). Must be 'streaming' or 'offline'")
            printUsage()
            exit(1)
        }

        logger.info("🎵 Processing audio file (\(parsed.mode.uppercased()) MODE): \(audioFile)")

        if parsed.mode == "streaming" {
            if parsed.rttmFile != nil {
                logger.warning("--rttm is only supported in offline mode, ignoring")
            }
            await runStreaming(audioFile: audioFile, args: parsed)
        } else {
            await runOffline(audioFile: audioFile, args: parsed)
        }
    }

    // MARK: - Streaming Mode

    private static func runStreaming(audioFile: String, args: ParsedArgs) async {
        let config = DiarizerConfig(
            clusteringThreshold: args.thresholdS,
            minSpeechDuration: args.minSpeechDuration,
            minEmbeddingUpdateDuration: args.minEmbeddingUpdateDuration,
            minSilenceGap: args.minSilenceGap,
            numClusters: args.numClusters,
            minActiveFramesCount: args.minActiveFramesCount,
            debugMode: args.debug,
            chunkDuration: args.chunkDuration,
            chunkOverlap: args.chunkOverlap
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

            let output = ProcessingResult(
                audioFile: audioFile,
                durationSeconds: duration,
                processingTimeSeconds: processingTime,
                realTimeFactor: rtfx,
                segments: result.segments,
                speakerCount: result.speakerDatabase?.count ?? 0,
                config: config,
                timings: result.timings
            )

            if let outputFile = args.outputFile {
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

    // MARK: - Offline Mode

    private static func runOffline(audioFile: String, args: ParsedArgs) async {
        do {
            var offlineConfig = OfflineDiarizerConfig(
                clusteringThreshold: args.thresholdD,
                Fa: args.fa,
                Fb: args.fb,
                windowDuration: args.windowDuration,
                sampleRate: args.sampleRate,
                segmentationStepRatio: args.stepRatio,
                embeddingBatchSize: args.embeddingBatchSize,
                embeddingExcludeOverlap: !args.includeOverlap,
                embeddingSkipStrategy: args.skipStrategy,
                minSegmentDuration: args.minSegmentDuration,
                minGapDuration: args.minGapDuration,
                exclusiveSegments: !args.overlappingSegments,
                speechOnsetThreshold: args.speechOnsetThreshold,
                speechOffsetThreshold: args.speechOffsetThreshold,
                segmentationMinDurationOn: args.minDurationOn,
                segmentationMinDurationOff: args.minDurationOff,
                maxVBxIterations: args.maxVBxIterations,
                convergenceTolerance: args.convergenceTolerance,
                embeddingExportPath: args.embeddingExportPath
            )

            // Apply speaker count constraints
            if let exactly = args.numSpeakers {
                offlineConfig = offlineConfig.withSpeakers(exactly: exactly)
            } else if args.minSpeakers != nil || args.maxSpeakers != nil {
                offlineConfig = offlineConfig.withSpeakers(
                    min: args.minSpeakers, max: args.maxSpeakers
                )
            }

            let modelDir = OfflineDiarizerModels.defaultModelsDirectory()
            let manager = OfflineDiarizerManager(config: offlineConfig)

            let configuration = MLModelConfigurationUtils.defaultConfiguration(
                computeUnits: args.computeUnits)
            let models = try await OfflineDiarizerModels.load(
                from: modelDir, configuration: configuration)
            manager.initialize(models: models)

            logger.info("Offline manager initialized")

            // Load and process audio file without materializing the full sample buffer.
            let audioURL = URL(fileURLWithPath: audioFile)
            let factory = AudioSourceFactory()
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
            ) { chunksProcessed, totalChunks in
                let printInterval = max(1, totalChunks / 4)
                if chunksProcessed % printInterval == 0 || chunksProcessed == totalChunks {
                    let percent = Int(Double(chunksProcessed) / Double(totalChunks) * 100)
                    logger.info("   Progress: \(percent)% (\(chunksProcessed)/\(totalChunks) chunks)")
                }
            }
            let processingTime = Date().timeIntervalSince(startTime)

            let durationSeconds = Double(diskSource.sampleCount) / Double(targetSampleRate)
            let rtfx = durationSeconds / processingTime

            logger.info("Diarization completed in \(String(format: "%.1f", processingTime))s")
            logger.info("   Real-time factor (RTFx): \(String(format: "%.2f", rtfx))x")
            logger.info("   Found \(result.segments.count) segments")

            let speakerCount = Set(result.segments.map { $0.speakerId }).count
            logger.info("   Detected \(speakerCount) speakers")

            var metrics: DiarizationMetrics?
            if let rttmFile = args.rttmFile {
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

            if let outputFile = args.outputFile {
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

    // MARK: - Argument Parsing

    private struct ParsedArgs {
        var mode = "streaming"
        var outputFile: String?
        var debug = false
        var rttmFile: String?
        var embeddingExportPath: String?

        // Streaming-mode params
        var thresholdS: Float = 0.7045655  // matches pyannote speaker-diarization-3.1 config.yaml
        var chunkDuration: Float = 10.0
        var chunkOverlap: Float = 0.0
        var minSpeechDuration: Float = 1.0
        var minSilenceGap: Float = 0.5
        var minActiveFramesCount: Float = 10.0
        var numClusters: Int = -1
        var minEmbeddingUpdateDuration: Float = 2.0

        // Offline-mode params
        var thresholdD: Double = 0.6
        var fa: Double = 0.07
        var fb: Double = 0.8
        var windowDuration: Double = 10.0
        var sampleRate: Int = 16_000
        var stepRatio: Double = 0.2
        var embeddingBatchSize: Int = 32
        var includeOverlap = false
        var skipStrategy: OfflineDiarizerConfig.EmbeddingSkipStrategy = .none
        var minSegmentDuration: Double = 1.0
        var minGapDuration: Double = 0.1
        var overlappingSegments = false
        var speechOnsetThreshold: Float = 0.5
        var speechOffsetThreshold: Float = 0.5
        var minDurationOn: Double = 0.0
        var minDurationOff: Double = 0.0
        var maxVBxIterations: Int = 20
        var convergenceTolerance: Double = 1e-4
        var minSpeakers: Int?
        var maxSpeakers: Int?
        var numSpeakers: Int?
        var computeUnits: MLComputeUnits = .all
    }

    private static func parseArguments(_ args: [String]) -> ParsedArgs {
        var parsed = ParsedArgs()
        var i = 0

        while i < args.count {
            switch args[i] {
            // Common
            case "--mode":
                if i + 1 < args.count {
                    parsed.mode = args[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < args.count {
                    parsed.outputFile = args[i + 1]
                    i += 1
                }
            case "--debug":
                parsed.debug = true
            case "--rttm":
                if i + 1 < args.count {
                    parsed.rttmFile = args[i + 1]
                    i += 1
                }

            // Streaming
            case "--threshold":
                if i + 1 < args.count {
                    if let val = Float(args[i + 1]) {
                        parsed.thresholdS = val
                        parsed.thresholdD = Double(val)
                    } else {
                        logger.warning("Invalid --threshold value: '\(args[i + 1])', using defaults")
                    }
                    i += 1
                }
            case "--chunk-seconds":
                if i + 1 < args.count {
                    parsed.chunkDuration = Float(args[i + 1]) ?? 10.0
                    i += 1
                }
            case "--overlap-seconds":
                if i + 1 < args.count {
                    parsed.chunkOverlap = Float(args[i + 1]) ?? 0.0
                    i += 1
                }
            case "--min-speech-duration":
                if i + 1 < args.count {
                    parsed.minSpeechDuration = Float(args[i + 1]) ?? 1.0
                    i += 1
                }
            case "--min-silence-gap":
                if i + 1 < args.count {
                    parsed.minSilenceGap = Float(args[i + 1]) ?? 0.5
                    i += 1
                }
            case "--min-active-frames":
                if i + 1 < args.count {
                    parsed.minActiveFramesCount = Float(args[i + 1]) ?? 10.0
                    i += 1
                }
            case "--num-clusters":
                if i + 1 < args.count {
                    parsed.numClusters = Int(args[i + 1]) ?? -1
                    i += 1
                }
            case "--min-embed-update":
                if i + 1 < args.count {
                    parsed.minEmbeddingUpdateDuration = Float(args[i + 1]) ?? 2.0
                    i += 1
                }

            // Offline
            case "--fa":
                if i + 1 < args.count {
                    parsed.fa = Double(args[i + 1]) ?? 0.07
                    i += 1
                }
            case "--fb":
                if i + 1 < args.count {
                    parsed.fb = Double(args[i + 1]) ?? 0.8
                    i += 1
                }
            case "--window-duration":
                if i + 1 < args.count {
                    parsed.windowDuration = Double(args[i + 1]) ?? 10.0
                    i += 1
                }
            case "--sample-rate":
                if i + 1 < args.count {
                    parsed.sampleRate = Int(args[i + 1]) ?? 16_000
                    i += 1
                }
            case "--step-ratio":
                if i + 1 < args.count {
                    parsed.stepRatio = Double(args[i + 1]) ?? 0.2
                    i += 1
                }
            case "--batch-size":
                if i + 1 < args.count {
                    parsed.embeddingBatchSize = Int(args[i + 1]) ?? 32
                    i += 1
                }
            case "--include-overlap":
                parsed.includeOverlap = true
            case "--skip-similarity":
                if i + 1 < args.count {
                    let t = Float(args[i + 1]) ?? 0.95
                    parsed.skipStrategy = .maskSimilarity(threshold: t)
                    i += 1
                }
            case "--min-segment-duration":
                if i + 1 < args.count {
                    parsed.minSegmentDuration = Double(args[i + 1]) ?? 1.0
                    i += 1
                }
            case "--min-gap-duration":
                if i + 1 < args.count {
                    parsed.minGapDuration = Double(args[i + 1]) ?? 0.1
                    i += 1
                }
            case "--overlapping-segments":
                parsed.overlappingSegments = true
            case "--onset-threshold":
                if i + 1 < args.count {
                    parsed.speechOnsetThreshold = Float(args[i + 1]) ?? 0.5
                    i += 1
                }
            case "--offset-threshold":
                if i + 1 < args.count {
                    parsed.speechOffsetThreshold = Float(args[i + 1]) ?? 0.5
                    i += 1
                }
            case "--min-duration-on":
                if i + 1 < args.count {
                    parsed.minDurationOn = Double(args[i + 1]) ?? 0.0
                    i += 1
                }
            case "--min-duration-off":
                if i + 1 < args.count {
                    parsed.minDurationOff = Double(args[i + 1]) ?? 0.0
                    i += 1
                }
            case "--max-vbx-iterations":
                if i + 1 < args.count {
                    parsed.maxVBxIterations = Int(args[i + 1]) ?? 20
                    i += 1
                }
            case "--convergence-tolerance":
                if i + 1 < args.count {
                    parsed.convergenceTolerance = Double(args[i + 1]) ?? 1e-4
                    i += 1
                }
            case "--min-speakers":
                if i + 1 < args.count {
                    parsed.minSpeakers = Int(args[i + 1])
                    i += 1
                }
            case "--max-speakers":
                if i + 1 < args.count {
                    parsed.maxSpeakers = Int(args[i + 1])
                    i += 1
                }
            case "--num-speakers":
                if i + 1 < args.count {
                    parsed.numSpeakers = Int(args[i + 1])
                    i += 1
                }
            case "--export-embeddings":
                if i + 1 < args.count {
                    parsed.embeddingExportPath = args[i + 1]
                    i += 1
                }
            case "--compute-units":
                if i + 1 < args.count {
                    switch args[i + 1].lowercased() {
                    case "all": parsed.computeUnits = .all
                    case "cpu-only": parsed.computeUnits = .cpuOnly
                    case "ane": parsed.computeUnits = .cpuAndNeuralEngine
                    default:
                        logger.warning("Invalid --compute-units value '\(args[i + 1])', using 'all'")
                    }
                    i += 1
                }

            default:
                logger.warning("Unknown option: \(args[i])")
            }
            i += 1
        }

        return parsed
    }

    // MARK: - Usage

    private static func printUsage() {
        let usage = """
            Process Command Usage:
                fluidaudio process <audio_file> [options]

            COMMON OPTIONS:
                --mode <streaming|offline>  Diarization mode (default: streaming)
                --output <file>             Save results to JSON file
                --debug                     Enable debug logging
                --help                      Show this help message

            STREAMING MODE OPTIONS (pyannote segmentation + WeSpeaker embeddings):
                --threshold <0.5-0.9>          Clustering threshold, lower = more speakers (default: 0.7045655)
                --chunk-seconds <sec>          Audio chunk size in seconds (default: 10.0)
                --overlap-seconds <sec>        Overlap between consecutive chunks (default: 0.0)
                --min-speech-duration <sec>     Drop segments shorter than this (default: 1.0)
                --min-silence-gap <sec>         Split same-speaker segments if gap exceeds this (default: 0.5)
                --min-active-frames <n>         Minimum active frames for valid speech (default: 10.0)
                --num-clusters <n>              Expected speakers, -1 = auto (default: -1)
                --min-embed-update <sec>        Min segment duration to update embeddings (default: 2.0)

            OFFLINE MODE OPTIONS (VBx clustering, all optional):
                --rttm <file>                   Compute DER/JER against RTTM annotations
                --threshold <0-√2>              Euclidean clustering threshold (default: 0.6)
                --fa <float>                    VBx warm-start precision (default: 0.07)
                --fb <float>                    VBx warm-start recall (default: 0.8)
                --window-duration <sec>         Segmentation window size (default: 10.0)
                --sample-rate <hz>              Target audio sample rate (default: 16000)
                --step-ratio <0-1>              Segmentation step ratio, lower = more overlap (default: 0.2)
                --batch-size <1-32>             Embedding batch size (default: 32)
                --include-overlap               Include overlap regions in embedding extraction
                --skip-similarity <0-1>         Skip embedding if mask similarity ≥ threshold (default: none)
                --min-segment-duration <sec>    Minimum segment duration (default: 1.0)
                --min-gap-duration <sec>        Merge segments separated by less than this gap (default: 0.1)
                --overlapping-segments          Allow speakers to overlap in output
                --onset-threshold <0-1>         VAD onset probability threshold (default: 0.5)
                --offset-threshold <0-1>        VAD offset probability threshold (default: 0.5)
                --min-duration-on <sec>         Min speech duration to trigger VAD on (default: 0.0)
                --min-duration-off <sec>        Min silence to trigger VAD off (default: 0.0)
                --max-vbx-iterations <n>        VBx max iterations (default: 20)
                --convergence-tolerance <float> VBx convergence tolerance (default: 1e-4)
                --min-speakers <n>              Minimum number of speakers
                --max-speakers <n>              Maximum number of speakers
                --num-speakers <n>              Exact speaker count (overrides min/max)
                --export-embeddings <file>      Export embeddings to JSON for debugging
                --compute-units <all|cpu-only|ane>  ML compute units for inference; use cpu-only/ane
                                                to guarantee off-GPU (default: all)

            Examples:
                # Streaming mode (default)
                fluidaudio process audio.wav --output results.json

                # Streaming with custom chunking
                fluidaudio process audio.wav --chunk-seconds 5 --overlap-seconds 2 --threshold 0.8

                # Offline mode with default settings
                fluidaudio process audio.wav --mode offline --output results.json

                # Offline with VBx tuning and speaker constraints
                fluidaudio process audio.wav --mode offline --fa 0.1 --fb 0.9 --num-speakers 3

                # Offline with embedding export and RTTM evaluation
                fluidaudio process audio.wav --mode offline --export-embeddings emb.json --rttm ann.rttm

            """
        fputs(usage, stderr)
        fflush(stderr)
    }
}
#endif
