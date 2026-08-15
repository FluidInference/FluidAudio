#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Streaming diarization benchmark for evaluating real-time performance
/// Uses first-occurrence speaker mapping for true streaming evaluation
enum StreamDiarizationBenchmark {
    private static let logger = AppLogger(category: "DiarizationBench")

    struct BenchmarkResult {
        let meetingName: String
        let der: Float
        let missRate: Float
        let falseAlarmRate: Float
        let speakerErrorRate: Float
        let jer: Float
        let rtfx: Float
        let processingTime: Double
        let chunksProcessed: Int
        let detectedSpeakers: Int
        let groundTruthSpeakers: Int
        let speakerFragmentation: Float
        let latency90th: Double
        let latency99th: Double
        // Timing breakdown
        let modelDownloadTime: Double
        let modelCompileTime: Double
        let audioLoadTime: Double
        let segmentationTime: Double
        let embeddingTime: Double
        let clusteringTime: Double
        let totalInferenceTime: Double
    }

    private struct ParsedArgs {
        var mode = "streaming"
        var dataset = "ami-sdm"
        var singleFile: String?
        var maxFiles: Int?
        var chunkSeconds: Double = 10.0
        var overlapSeconds: Double = 0.0
        var thresholdS: Float = 0.7045655  // matches pyannote config.yaml
        var assignmentThreshold: Float = 0.84
        var updateThreshold: Float = 0.56
        var outputFile: String?
        var csvFile: String?
        var verbose = false
        var debugMode = false
        var autoDownload = false
        var iterations = 1

        // Streaming DiarizerConfig
        var minSpeechDuration: Float = 1.0
        var minSilenceGap: Float = 0.5
        var minActiveFramesCount: Float = 10.0
        var minEmbeddingUpdateDuration: Float = 2.0
        var numClusters: Int = -1

        // Offline DiarizerConfig
        var thresholdD: Double = 0.6
        var fa: Double = 0.07
        var fb: Double = 0.8
        var windowDuration: Double = 10.0
        var sampleRate: Int = 16000
        var stepRatio: Double = 0.2
        var batchSize: Int = 32
        var excludeOverlap = true
        var skipSimilarity: Float?
        var minSegmentDuration: Double = 1.0
        var minGapDuration: Double = 0.1
        var exclusiveSegments = true
        var onsetThreshold: Float = 0.5
        var offsetThreshold: Float = 0.5
        var segMinDurationOn: Double = 0.0
        var segMinDurationOff: Double = 0.0
        var maxVBxIterations: Int = 20
        var convergenceTolerance: Double = 1e-4
        var minSpeakers: Int?
        var maxSpeakers: Int?
        var numSpeakers: Int?
    }

    static func printUsage() {
        let usage = """
            Diarization Benchmark Command

            Evaluates speaker diarization in either streaming (online) or offline (VBx) mode.

            Usage: fluidaudio diarization-benchmark [options]

            Common Options:
                --mode <streaming|offline>  Diarization mode (default: streaming)
                --dataset <name>            Dataset to benchmark (default: ami-sdm)
                --single-file <name>        Process a specific meeting (e.g., ES2004a)
                --max-files <n>             Maximum number of files to process
                --output <file>             Output JSON file for results
                --csv <file>                Output CSV file for summary
                --verbose                   Enable verbose output
                --debug                     Enable debug output
                --auto-download             Auto-download dataset if missing
                --iterations <n>            Number of iterations per file (default: 1)
                --help                      Show this help message

            Streaming Mode Options:
                --chunk-seconds <sec>       Chunk duration (default: 10.0)
                --overlap-seconds <sec>     Overlap between chunks (default: 0.0)
                --threshold <0.5-0.9>       Clustering threshold, lower = more speakers (default: 0.7045655)
                --assignment-threshold      Threshold for assigning to speakers (default: 0.84)
                --update-threshold          Threshold for updating embeddings (default: 0.56)
                --min-speech-duration <sec>  Drop segments shorter than this (default: 1.0)
                --min-silence-gap <sec>     Split same-speaker if gap exceeds this (default: 0.5)
                --min-active-frames <n>     Minimum active frames for valid speech (default: 10.0)
                --num-clusters <n>          Expected speakers, -1 = auto (default: -1)
                --min-embed-update <sec>    Min segment duration to update embeddings (default: 2.0)

            Offline Mode Options:
                --threshold <0-√2>          Clustering threshold (default: 0.6)
                --fa <float>                VBx warm-start precision (default: 0.07)
                --fb <float>                VBx warm-start recall (default: 0.8)
                --window-duration <sec>     Segmentation window size (default: 10.0)
                --sample-rate <hz>          Target audio sample rate (default: 16000)
                --step-ratio <0-1>          Segmentation step ratio (default: 0.2)
                --batch-size <1-32>         Embedding batch size (default: 32)
                --include-overlap           Include overlap in embedding extraction
                --skip-similarity <0-1>     Skip embedding if mask similarity >= threshold
                --min-segment-duration <sec> Minimum segment duration (default: 1.0)
                --min-gap-duration <sec>    Merge segments separated by less than this gap (default: 0.1)
                --overlapping-segments      Allow speakers to overlap in output
                --onset-threshold <0-1>     VAD onset probability (default: 0.5)
                --offset-threshold <0-1>    VAD offset probability (default: 0.5)
                --min-duration-on <sec>     Min speech to trigger VAD on (default: 0.0)
                --min-duration-off <sec>    Min silence to trigger VAD off (default: 0.0)
                --max-vbx-iterations <n>    VBx max iterations (default: 20)
                --convergence-tolerance <f> VBx convergence tolerance (default: 1e-4)
                --min-speakers <n>          Minimum number of speakers
                --max-speakers <n>          Maximum number of speakers
                --num-speakers <n>          Exact speaker count (overrides min/max)

            Modes:
                streaming   Online diarization with chunk-based processing (first-occurrence speaker mapping)
                offline     Batch diarization with VBx clustering (optimal speaker mapping with Hungarian algorithm)

            Streaming Modes (via chunk/overlap settings):
                Real-time:  --chunk-seconds 3 --overlap-seconds 2   (~15-30x RTFx)
                Balanced:   --chunk-seconds 10 --overlap-seconds 5  (~70x RTFx)
                Batch:      --chunk-seconds 10 --overlap-seconds 0  (~140x RTFx)

            Performance Targets:
                DER < 30%  (competitive with research systems)
                RTFx > 1x  (real-time capable, streaming mode)

            Examples:
                # Offline VBx clustering (research-grade accuracy)
                fluidaudio diarization-benchmark --mode offline --single-file ES2004a

                # Streaming mode with real-time settings
                fluidaudio diarization-benchmark --mode streaming --single-file ES2004a \\
                    --chunk-seconds 3 --overlap-seconds 2

                # Full AMI benchmark in offline mode
                fluidaudio diarization-benchmark --mode offline --dataset ami-sdm --csv results.csv

                # Quick test on 5 files (offline)
                fluidaudio diarization-benchmark --mode offline --max-files 5 --verbose
            """
        fputs(usage, stderr)
        fflush(stderr)
    }

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        var args = ParsedArgs()

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--mode":
                if i + 1 < arguments.count {
                    args.mode = arguments[i + 1]
                    i += 1
                }
            case "--dataset":
                if i + 1 < arguments.count {
                    args.dataset = arguments[i + 1]
                    i += 1
                }
            case "--single-file":
                if i + 1 < arguments.count {
                    args.singleFile = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    args.maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--chunk-seconds":
                if i + 1 < arguments.count {
                    args.chunkSeconds = Double(arguments[i + 1]) ?? 10.0
                    i += 1
                }
            case "--overlap-seconds":
                if i + 1 < arguments.count {
                    args.overlapSeconds = Double(arguments[i + 1]) ?? 0.0
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    if let val = Float(arguments[i + 1]) {
                        args.thresholdS = val
                        args.thresholdD = Double(val)
                    } else {
                        logger.warning("Invalid --threshold value '\(arguments[i + 1])', using defaults")
                    }
                    i += 1
                }
            case "--assignment-threshold":
                if i + 1 < arguments.count {
                    args.assignmentThreshold = Float(arguments[i + 1]) ?? 0.84
                    i += 1
                }
            case "--update-threshold":
                if i + 1 < arguments.count {
                    args.updateThreshold = Float(arguments[i + 1]) ?? 0.56
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    args.outputFile = arguments[i + 1]
                    i += 1
                }
            case "--csv":
                if i + 1 < arguments.count {
                    args.csvFile = arguments[i + 1]
                    i += 1
                }
            case "--verbose":
                args.verbose = true
            case "--debug":
                args.debugMode = true
            case "--auto-download":
                args.autoDownload = true
            case "--iterations":
                if i + 1 < arguments.count {
                    args.iterations = Int(arguments[i + 1]) ?? 1
                    i += 1
                }

            // Streaming-only flags
            case "--min-speech-duration":
                if i + 1 < arguments.count {
                    args.minSpeechDuration = Float(arguments[i + 1]) ?? 1.0
                    i += 1
                }
            case "--min-silence-gap":
                if i + 1 < arguments.count {
                    args.minSilenceGap = Float(arguments[i + 1]) ?? 0.5
                    i += 1
                }
            case "--min-active-frames":
                if i + 1 < arguments.count {
                    args.minActiveFramesCount = Float(arguments[i + 1]) ?? 10.0
                    i += 1
                }
            case "--num-clusters":
                if i + 1 < arguments.count {
                    args.numClusters = Int(arguments[i + 1]) ?? -1
                    i += 1
                }
            case "--min-embed-update":
                if i + 1 < arguments.count {
                    args.minEmbeddingUpdateDuration = Float(arguments[i + 1]) ?? 2.0
                    i += 1
                }

            // Offline-only flags
            case "--fa":
                if i + 1 < arguments.count {
                    args.fa = Double(arguments[i + 1]) ?? 0.07
                    i += 1
                }
            case "--fb":
                if i + 1 < arguments.count {
                    args.fb = Double(arguments[i + 1]) ?? 0.8
                    i += 1
                }
            case "--window-duration":
                if i + 1 < arguments.count {
                    args.windowDuration = Double(arguments[i + 1]) ?? 10.0
                    i += 1
                }
            case "--sample-rate":
                if i + 1 < arguments.count {
                    args.sampleRate = Int(arguments[i + 1]) ?? 16000
                    i += 1
                }
            case "--step-ratio":
                if i + 1 < arguments.count {
                    args.stepRatio = Double(arguments[i + 1]) ?? 0.2
                    i += 1
                }
            case "--batch-size":
                if i + 1 < arguments.count {
                    args.batchSize = Int(arguments[i + 1]) ?? 32
                    i += 1
                }
            case "--include-overlap":
                args.excludeOverlap = false
            case "--skip-similarity":
                if i + 1 < arguments.count {
                    args.skipSimilarity = Float(arguments[i + 1])
                    i += 1
                }
            case "--min-segment-duration":
                if i + 1 < arguments.count {
                    args.minSegmentDuration = Double(arguments[i + 1]) ?? 1.0
                    i += 1
                }
            case "--min-gap-duration":
                if i + 1 < arguments.count {
                    args.minGapDuration = Double(arguments[i + 1]) ?? 0.1
                    i += 1
                }
            case "--overlapping-segments":
                args.exclusiveSegments = false
            case "--onset-threshold":
                if i + 1 < arguments.count {
                    args.onsetThreshold = Float(arguments[i + 1]) ?? 0.5
                    i += 1
                }
            case "--offset-threshold":
                if i + 1 < arguments.count {
                    args.offsetThreshold = Float(arguments[i + 1]) ?? 0.5
                    i += 1
                }
            case "--min-duration-on":
                if i + 1 < arguments.count {
                    args.segMinDurationOn = Double(arguments[i + 1]) ?? 0.0
                    i += 1
                }
            case "--min-duration-off":
                if i + 1 < arguments.count {
                    args.segMinDurationOff = Double(arguments[i + 1]) ?? 0.0
                    i += 1
                }
            case "--max-vbx-iterations":
                if i + 1 < arguments.count {
                    args.maxVBxIterations = Int(arguments[i + 1]) ?? 20
                    i += 1
                }
            case "--convergence-tolerance":
                if i + 1 < arguments.count {
                    args.convergenceTolerance = Double(arguments[i + 1]) ?? 1e-4
                    i += 1
                }
            case "--min-speakers":
                if i + 1 < arguments.count {
                    args.minSpeakers = Int(arguments[i + 1])
                    i += 1
                }
            case "--max-speakers":
                if i + 1 < arguments.count {
                    args.maxSpeakers = Int(arguments[i + 1])
                    i += 1
                }
            case "--num-speakers":
                if i + 1 < arguments.count {
                    args.numSpeakers = Int(arguments[i + 1])
                    i += 1
                }
            default:
                logger.warning("Unknown argument: \(arguments[i])")
            }
            i += 1
        }

        // Validate mode
        guard args.mode == "streaming" || args.mode == "offline" else {
            logger.error("Invalid mode: \(args.mode). Must be 'streaming' or 'offline'")
            printUsage()
            return
        }

        logger.info("🚀 Starting Diarization Benchmark (\(args.mode.uppercased()) MODE)")
        logger.info("   Dataset: \(args.dataset)")
        logger.info("   Clustering threshold: \(args.thresholdS)")

        if args.mode == "streaming" {
            // Validate streaming settings
            let hopSize = max(args.chunkSeconds - args.overlapSeconds, 1.0)
            let overlapRatio = args.overlapSeconds / args.chunkSeconds

            logger.info("   Chunk size: \(args.chunkSeconds)s")
            logger.info("   Overlap: \(args.overlapSeconds)s (\(String(format: "%.0f", overlapRatio * 100))%)")
            logger.info("   Hop size: \(hopSize)s")
            logger.info("   Assignment threshold: \(args.assignmentThreshold)")
            logger.info("   Update threshold: \(args.updateThreshold)")

            // Determine streaming mode
            let streamingMode: String
            if args.overlapSeconds == 0 {
                streamingMode = "Batch (no overlap)"
            } else if overlapRatio >= 0.6 {
                streamingMode = "Real-time (high overlap)"
            } else {
                streamingMode = "Balanced"
            }
            logger.info("   Streaming mode: \(streamingMode)")
        } else {
            logger.info("   Using VBx clustering with optimal speaker mapping")
        }

        logger.info("")

        // Download dataset if needed
        if args.autoDownload {
            logger.info("📥 Downloading AMI dataset if needed...")
            // Download both audio and annotations
            await DatasetDownloader.downloadAMIDataset(
                variant: args.dataset == "ami-ihm" ? .ihm : .sdm,
                force: false,
                singleFile: args.singleFile
            )
            await DatasetDownloader.downloadAMIAnnotations(force: false)
        }

        // Get list of files to process
        let filesToProcess: [String]
        if let meeting = args.singleFile {
            filesToProcess = [meeting]
        } else {
            filesToProcess = getAMIFiles(dataset: args.dataset, maxFiles: args.maxFiles)
        }

        if filesToProcess.isEmpty {
            logger.error("❌ No files found to process")
            return
        }

        logger.info("📂 Processing \(filesToProcess.count) file(s)\n")

        // Initialize models once and track timing
        logger.info("🔧 Initializing models...")
        let modelStartTime = Date()
        let models: DiarizerModels
        var offlineManager: OfflineDiarizerManager?

        do {
            models = try await DiarizerModels.downloadIfNeeded()

            // For offline mode, also initialize the offline manager
            if args.mode == "offline" {
                let modelDir = OfflineDiarizerModels.defaultModelsDirectory()
                let embeddingSkipStrategy: OfflineDiarizerConfig.EmbeddingSkipStrategy =
                    if let simThreshold = args.skipSimilarity {
                        .maskSimilarity(threshold: simThreshold)
                    } else {
                        .none
                    }
                var offlineConfig = OfflineDiarizerConfig(
                    clusteringThreshold: args.thresholdD,
                    Fa: args.fa,
                    Fb: args.fb,
                    windowDuration: args.windowDuration,
                    sampleRate: args.sampleRate,
                    segmentationStepRatio: args.stepRatio,
                    embeddingBatchSize: args.batchSize,
                    embeddingExcludeOverlap: args.excludeOverlap,
                    embeddingSkipStrategy: embeddingSkipStrategy,
                    minSegmentDuration: args.minSegmentDuration,
                    minGapDuration: args.minGapDuration,
                    exclusiveSegments: args.exclusiveSegments,
                    speechOnsetThreshold: args.onsetThreshold,
                    speechOffsetThreshold: args.offsetThreshold,
                    segmentationMinDurationOn: args.segMinDurationOn,
                    segmentationMinDurationOff: args.segMinDurationOff,
                    maxVBxIterations: args.maxVBxIterations,
                    convergenceTolerance: args.convergenceTolerance
                )
                if let exact = args.numSpeakers {
                    offlineConfig = offlineConfig.withSpeakers(exactly: exact)
                } else if args.minSpeakers != nil || args.maxSpeakers != nil {
                    offlineConfig = offlineConfig.withSpeakers(min: args.minSpeakers, max: args.maxSpeakers)
                }
                offlineManager = OfflineDiarizerManager(config: offlineConfig)
                let offlineModels = try await OfflineDiarizerModels.load(from: modelDir)
                offlineManager?.initialize(models: offlineModels)
                logger.info("✅ Offline manager initialized")
            }
        } catch {
            logger.error("❌ Failed to initialize models: \(error)")
            return
        }
        let modelInitTime = Date().timeIntervalSince(modelStartTime)
        logger.info("✅ Models ready (took \(String(format: "%.2f", modelInitTime))s)\n")

        // Process each file
        var allResults: [BenchmarkResult] = []

        for (fileIndex, meetingName) in filesToProcess.enumerated() {
            logger.info(String(repeating: "=", count: 60))
            logger.info("[\(fileIndex + 1)/\(filesToProcess.count)] Processing: \(meetingName)")
            logger.info(String(repeating: "=", count: 60))

            var iterationResults: [BenchmarkResult] = []

            for iteration in 1...args.iterations {
                if args.iterations > 1 {
                    logger.info("  Iteration \(iteration)/\(args.iterations)")
                }

                let result: BenchmarkResult?
                if args.mode == "streaming" {
                    result = await processStreamingMeeting(
                        meetingName: meetingName,
                        models: models,
                        modelInitTime: modelInitTime,
                        chunkSeconds: args.chunkSeconds,
                        overlapSeconds: args.overlapSeconds,
                        threshold: args.thresholdS,
                        assignmentThreshold: args.assignmentThreshold,
                        updateThreshold: args.updateThreshold,
                        minSpeechDuration: args.minSpeechDuration,
                        minSilenceGap: args.minSilenceGap,
                        minActiveFramesCount: args.minActiveFramesCount,
                        minEmbeddingUpdateDuration: args.minEmbeddingUpdateDuration,
                        numClusters: args.numClusters,
                        verbose: args.verbose,
                        debugMode: args.debugMode
                    )
                } else {
                    result = await processOfflineMeeting(
                        meetingName: meetingName,
                        controller: offlineManager!,
                        modelInitTime: modelInitTime,
                        verbose: args.verbose,
                        debugMode: args.debugMode
                    )
                }

                if let result = result {
                    iterationResults.append(result)

                    // Print summary for this iteration
                    logger.info("📊 Results for \(meetingName) (iteration \(iteration)):")
                    logger.info("  DER: \(String(format: "%.1f", result.der))%")
                    logger.info("  JER: \(String(format: "%.1f", result.jer))%")
                    logger.info("  RTFx: \(String(format: "%.1f", result.rtfx))x")
                    logger.info("  Speakers: \(result.detectedSpeakers) detected / \(result.groundTruthSpeakers) truth")

                    // Print timing breakdown
                    logger.info("⏱️ Diarization Pipeline Timing Breakdown:")
                    logger.info("  Time spent in each stage of streaming diarization:\n")
                    logger.info("  Stage               Time (s)    %     Description")
                    logger.info("  " + String(repeating: "-", count: 60))
                    let totalTime = result.processingTime
                    logger.info(
                        String(
                            format: "  Model Download      %.3f      %.1f   Fetching diarization models",
                            result.modelDownloadTime, result.modelDownloadTime / totalTime * 100))
                    logger.info(
                        String(
                            format: "  Model Compile       %.3f      %.1f   CoreML compilation",
                            result.modelCompileTime, result.modelCompileTime / totalTime * 100))
                    logger.info(
                        String(
                            format: "  Audio Load          %.3f      %.1f   Loading audio file", result.audioLoadTime,
                            result.audioLoadTime / totalTime * 100))
                    logger.info(
                        String(
                            format: "  Segmentation        %.3f      %.1f   Detecting speech regions",
                            result.segmentationTime, result.segmentationTime / totalTime * 100))
                    logger.info(
                        String(
                            format: "  Embedding           %.3f      %.1f   Extracting speaker voices",
                            result.embeddingTime, result.embeddingTime / totalTime * 100))
                    logger.info(
                        String(
                            format: "  Clustering          %.3f      %.1f   Grouping same speakers",
                            result.clusteringTime, result.clusteringTime / totalTime * 100))
                    logger.info("  " + String(repeating: "-", count: 60))
                    logger.info(String(format: "  Total               %.3f    100.0   Full pipeline", totalTime))
                }
            }

            // Average results if multiple iterations
            if !iterationResults.isEmpty {
                let avgResult = averageResults(iterationResults)
                allResults.append(avgResult)

                if args.iterations > 1 {
                    logger.info("📊 Average over \(args.iterations) iterations:")
                    logger.info(
                        "  DER: \(String(format: "%.1f", avgResult.der))% ± \(String(format: "%.1f", standardDeviation(iterationResults.map { $0.der })))%"
                    )
                    logger.info(
                        "  RTFx: \(String(format: "%.1f", avgResult.rtfx))x ± \(String(format: "%.1f", standardDeviation(iterationResults.map { $0.rtfx })))x"
                    )
                }
            }
        }

        // Fail loudly if no meeting produced a result (e.g. missing ground truth
        // annotations) instead of exiting cleanly with empty metrics (issue #752).
        guard !allResults.isEmpty else {
            logger.error("❌ Benchmark produced no results — see errors above")
            exit(1)
        }

        // Print final summary
        printFinalSummary(results: allResults)

        // Save results
        if let outputPath = args.outputFile {
            saveJSONResults(results: allResults, to: outputPath)
        }

        if let csvPath = args.csvFile {
            saveCSVResults(results: allResults, to: csvPath)
        }
    }

    private static func processStreamingMeeting(
        meetingName: String,
        models: DiarizerModels,
        modelInitTime: Double,
        chunkSeconds: Double,
        overlapSeconds: Double,
        threshold: Float,
        assignmentThreshold: Float,
        updateThreshold: Float,
        minSpeechDuration: Float,
        minSilenceGap: Float,
        minActiveFramesCount: Float,
        minEmbeddingUpdateDuration: Float,
        numClusters: Int,
        verbose: Bool,
        debugMode: Bool
    ) async -> BenchmarkResult? {

        // Load audio
        let audioPath = getAudioPath(for: meetingName)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            logger.error("❌ Audio file not found: \(audioPath)")
            return nil
        }

        do {
            // Track audio loading time
            let audioLoadStart = Date()
            let audioData = try await loadAudioFile(at: audioPath)
            let audioLoadTime = Date().timeIntervalSince(audioLoadStart)
            let totalDuration = Double(audioData.count) / 16000.0

            if verbose {
                logger.info("  Audio duration: \(String(format: "%.1f", totalDuration))s")
                logger.info("  Audio load time: \(String(format: "%.3f", audioLoadTime))s")
            }

            // Initialize diarizer with streaming manager
            let config = DiarizerConfig(
                clusteringThreshold: threshold,
                minSpeechDuration: minSpeechDuration,
                minEmbeddingUpdateDuration: minEmbeddingUpdateDuration,
                minSilenceGap: minSilenceGap,
                numClusters: numClusters,
                minActiveFramesCount: minActiveFramesCount,
                debugMode: debugMode,
                chunkDuration: Float(chunkSeconds),
                chunkOverlap: Float(overlapSeconds)
            )

            let diarizerManager = DiarizerManager(config: config)
            diarizerManager.initialize(models: models)

            // Configure streaming manager
            diarizerManager.speakerManager.speakerThreshold = assignmentThreshold
            diarizerManager.speakerManager.embeddingThreshold = updateThreshold

            // Process in chunks
            let samplesPerChunk = Int(chunkSeconds * 16000)
            let hopSamples = Int((chunkSeconds - overlapSeconds) * 16000)
            var position = 0
            var chunkIndex = 0

            let startTime = Date()
            var chunkLatencies: [Double] = []
            var allSegments: [TimedSpeakerSegment] = []
            var speakerAppearances: [String: [Int]] = [:]  // Track which chunks each speaker appears in

            // Aggregate timing data across chunks
            var totalSegmentationTime: Double = 0
            var totalEmbeddingTime: Double = 0
            var totalClusteringTime: Double = 0

            while position < audioData.count {
                let chunkStart = Date()
                let chunkEnd = min(position + samplesPerChunk, audioData.count)
                let chunk = Array(audioData[position..<chunkEnd])

                // Pad if necessary
                var paddedChunk = chunk
                if paddedChunk.count < samplesPerChunk {
                    paddedChunk.append(contentsOf: [Float](repeating: 0, count: samplesPerChunk - paddedChunk.count))
                }

                let chunkStartTime = Double(position) / 16000.0

                // Process chunk and track timing
                let inferenceStart = Date()
                let chunkResult = try diarizerManager.performCompleteDiarization(
                    paddedChunk, atTime: chunkStartTime)
                let inferenceTime = Date().timeIntervalSince(inferenceStart)

                // Track chunk processing latency
                let chunkLatency = Date().timeIntervalSince(chunkStart)
                chunkLatencies.append(chunkLatency)

                // Estimate timing breakdown (approximate based on typical ratios)
                // In streaming mode, operations are incremental per chunk
                let estimatedSegTime = inferenceTime * 0.3  // ~30% for segmentation
                let estimatedEmbTime = inferenceTime * 0.5  // ~50% for embedding
                let estimatedClustTime = inferenceTime * 0.2  // ~20% for clustering

                totalSegmentationTime += estimatedSegTime
                totalEmbeddingTime += estimatedEmbTime
                totalClusteringTime += estimatedClustTime

                // Collect segments with adjusted times
                for segment in chunkResult.segments {
                    let adjustedSegment = TimedSpeakerSegment(
                        speakerId: segment.speakerId,
                        embedding: segment.embedding,
                        startTimeSeconds: segment.startTimeSeconds,
                        endTimeSeconds: segment.endTimeSeconds,
                        qualityScore: segment.qualityScore
                    )
                    allSegments.append(adjustedSegment)

                    // Track speaker appearances for fragmentation analysis
                    if speakerAppearances[segment.speakerId] == nil {
                        speakerAppearances[segment.speakerId] = []
                    }
                    speakerAppearances[segment.speakerId]?.append(chunkIndex)
                }

                // Verbose progress
                if verbose && chunkIndex % 20 == 0 {
                    let progress = Double(position) / Double(audioData.count) * 100
                    let elapsed = Date().timeIntervalSince(startTime)
                    let processedDuration = Double(position) / 16000.0
                    let rtfx = processedDuration / elapsed

                    let currentSpeakerCount = diarizerManager.speakerManager.speakerCount
                    logger.info(
                        String(
                            format: "    [Chunk %3d] %.1f%% | RTFx: %.1fx | Speakers: %d | Latency: %.3fs",
                            chunkIndex, progress, rtfx,
                            currentSpeakerCount,
                            chunkLatency))
                }

                position += hopSamples
                chunkIndex += 1
            }

            let totalElapsed = Date().timeIntervalSince(startTime)
            let finalRTFx = totalDuration / totalElapsed

            // Load ground truth (throws if annotations are missing — never scores
            // against placeholder data, see issue #752)
            let groundTruth = try AMIParser.loadAMIGroundTruth(
                for: meetingName,
                duration: Float(totalDuration)
            )

            guard !groundTruth.isEmpty else {
                logger.warning("⚠️ No ground truth found for \(meetingName)")
                return nil
            }

            // Calculate metrics with first-occurrence mapping for true streaming evaluation
            let metrics = calculateStreamingMetrics(
                predicted: allSegments,
                groundTruth: groundTruth,
                totalDuration: Float(totalDuration)
            )

            // Calculate speaker fragmentation (how many separate ID clusters per true speaker)
            let fragmentation = calculateFragmentation(
                speakerAppearances: speakerAppearances,
                totalChunks: chunkIndex
            )

            // Calculate latency percentiles
            let sortedLatencies = chunkLatencies.sorted()
            let p90Index = Int(Double(sortedLatencies.count) * 0.9)
            let p99Index = Int(Double(sortedLatencies.count) * 0.99)
            let latency90th = sortedLatencies[min(p90Index, sortedLatencies.count - 1)]
            let latency99th = sortedLatencies[min(p99Index, sortedLatencies.count - 1)]

            // Calculate total inference time
            let totalInferenceTime = totalSegmentationTime + totalEmbeddingTime + totalClusteringTime

            let finalSpeakerCount = diarizerManager.speakerManager.speakerCount

            return BenchmarkResult(
                meetingName: meetingName,
                der: metrics.der,
                missRate: metrics.missRate,
                falseAlarmRate: metrics.falseAlarmRate,
                speakerErrorRate: metrics.speakerErrorRate,
                jer: metrics.jer,
                rtfx: Float(finalRTFx),
                processingTime: totalElapsed,
                chunksProcessed: chunkIndex,
                detectedSpeakers: finalSpeakerCount,
                groundTruthSpeakers: AMIParser.getGroundTruthSpeakerCount(for: meetingName),
                speakerFragmentation: fragmentation,
                latency90th: latency90th,
                latency99th: latency99th,
                // Timing breakdown
                modelDownloadTime: modelInitTime * 0.7,  // Estimate ~70% for download
                modelCompileTime: modelInitTime * 0.3,  // Estimate ~30% for compile
                audioLoadTime: audioLoadTime,
                segmentationTime: totalSegmentationTime,
                embeddingTime: totalEmbeddingTime,
                clusteringTime: totalClusteringTime,
                totalInferenceTime: totalInferenceTime
            )

        } catch {
            logger.error("❌ Error processing \(meetingName): \(error)")
            return nil
        }
    }

    private static func processOfflineMeeting(
        meetingName: String,
        controller: OfflineDiarizerManager,
        modelInitTime: Double,
        verbose: Bool,
        debugMode: Bool
    ) async -> BenchmarkResult? {

        // Load audio
        let audioPath = getAudioPath(for: meetingName)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            logger.error("❌ Audio file not found: \(audioPath)")
            return nil
        }

        do {
            // Track audio loading time
            let audioLoadStart = Date()
            let audioData = try await loadAudioFile(at: audioPath)
            let audioLoadTime = Date().timeIntervalSince(audioLoadStart)
            let totalDuration = Double(audioData.count) / 16000.0

            if verbose {
                logger.info("  Audio duration: \(String(format: "%.1f", totalDuration))s")
                logger.info("  Audio load time: \(String(format: "%.3f", audioLoadTime))s")
            }

            // Process with offline controller
            let startTime = Date()
            let result = try await controller.process(audio: audioData)
            let totalElapsed = Date().timeIntervalSince(startTime)
            let finalRTFx = totalDuration / totalElapsed

            if verbose {
                logger.info("  Processing time: \(String(format: "%.3f", totalElapsed))s")
                logger.info("  RTFx: \(String(format: "%.1f", finalRTFx))x")
            }

            // Load ground truth (throws if annotations are missing — never scores
            // against placeholder data, see issue #752)
            let groundTruth = try AMIParser.loadAMIGroundTruth(
                for: meetingName,
                duration: Float(totalDuration)
            )

            guard !groundTruth.isEmpty else {
                logger.warning("⚠️ No ground truth found for \(meetingName)")
                return nil
            }

            // Calculate metrics with Hungarian algorithm (optimal mapping for offline)
            let metrics = DiarizationMetricsCalculator.offlineMetrics(
                predicted: result.segments,
                groundTruth: groundTruth,
                frameSize: 0.01,
                audioDurationSeconds: totalDuration,
                logger: logger
            )

            // Extract timing breakdown if available
            let segmentationTime = result.timings?.segmentationSeconds ?? 0
            let embeddingTime = result.timings?.embeddingExtractionSeconds ?? 0
            let clusteringTime = result.timings?.speakerClusteringSeconds ?? 0
            let totalInferenceTime = segmentationTime + embeddingTime + clusteringTime

            // Count detected speakers
            let detectedSpeakers = Set(result.segments.map { $0.speakerId }).count

            return BenchmarkResult(
                meetingName: meetingName,
                der: metrics.der,
                missRate: metrics.missRate,
                falseAlarmRate: metrics.falseAlarmRate,
                speakerErrorRate: metrics.speakerErrorRate,
                jer: metrics.jer,
                rtfx: Float(finalRTFx),
                processingTime: totalElapsed,
                chunksProcessed: 1,  // Offline processes entire file at once
                detectedSpeakers: detectedSpeakers,
                groundTruthSpeakers: AMIParser.getGroundTruthSpeakerCount(for: meetingName),
                speakerFragmentation: 1.0,  // No fragmentation in offline mode
                latency90th: totalElapsed,
                latency99th: totalElapsed,
                // Timing breakdown
                modelDownloadTime: modelInitTime * 0.7,
                modelCompileTime: modelInitTime * 0.3,
                audioLoadTime: audioLoadTime,
                segmentationTime: segmentationTime,
                embeddingTime: embeddingTime,
                clusteringTime: clusteringTime,
                totalInferenceTime: totalInferenceTime
            )

        } catch {
            logger.error("❌ Error processing \(meetingName): \(error)")
            return nil
        }
    }

    /// Calculate DER metrics with first-occurrence mapping for streaming evaluation
    private static func calculateStreamingMetrics(
        predicted: [TimedSpeakerSegment],
        groundTruth: [TimedSpeakerSegment],
        totalDuration: Float
    ) -> (der: Float, missRate: Float, falseAlarmRate: Float, speakerErrorRate: Float, jer: Float) {

        let frameSize: Float = 0.01
        let totalFrames = Int(totalDuration / frameSize)

        // Build a first-occurrence mapping based on chronological appearance
        // This matches what test-speaker-manager does for consistent results
        var firstOccurrenceMap: [String: String] = [:]
        var usedGroundTruthSpeakers = Set<String>()

        // Sort segments by start time to process chronologically
        let sortedPredicted = predicted.sorted { $0.startTimeSeconds < $1.startTimeSeconds }
        let sortedGroundTruth = groundTruth.sorted { $0.startTimeSeconds < $1.startTimeSeconds }

        // Map each predicted speaker to ground truth based on first significant overlap
        // Ensure each ground truth speaker is only assigned once (1-to-1 mapping)
        for predSegment in sortedPredicted {
            // Skip if already mapped
            if firstOccurrenceMap[predSegment.speakerId] != nil {
                continue
            }

            // Find overlapping ground truth segments
            var overlapsByGtSpeaker: [String: Float] = [:]

            for gtSegment in sortedGroundTruth {
                // Skip if this GT speaker is already assigned to another predicted speaker
                if usedGroundTruthSpeakers.contains(gtSegment.speakerId) {
                    continue
                }

                let overlap =
                    min(predSegment.endTimeSeconds, gtSegment.endTimeSeconds)
                    - max(predSegment.startTimeSeconds, gtSegment.startTimeSeconds)

                if overlap > 0 {
                    overlapsByGtSpeaker[gtSegment.speakerId, default: 0] += overlap
                }
            }

            // Find the GT speaker with most overlap
            if let (bestMatch, bestOverlap) = overlapsByGtSpeaker.max(by: { $0.value < $1.value }),
                bestOverlap > 0.5
            {  // Require at least 0.5s total overlap
                firstOccurrenceMap[predSegment.speakerId] = bestMatch
                usedGroundTruthSpeakers.insert(bestMatch)
            }
        }

        logger.debug("🔄 STREAMING MAPPING (first-occurrence): \(firstOccurrenceMap)")

        // Calculate frame-based metrics
        var missedFrames = 0
        var falseAlarmFrames = 0
        var speakerErrorFrames = 0

        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            // Find active speakers at this time
            var gtSpeaker: String?
            for segment in groundTruth {
                if frameTime >= segment.startTimeSeconds && frameTime < segment.endTimeSeconds {
                    gtSpeaker = segment.speakerId
                    break
                }
            }

            var predSpeaker: String?
            for segment in predicted {
                if frameTime >= segment.startTimeSeconds && frameTime < segment.endTimeSeconds {
                    predSpeaker = segment.speakerId
                    break
                }
            }

            switch (gtSpeaker, predSpeaker) {
            case (nil, nil):
                continue  // Both silent - correct
            case (nil, _):
                falseAlarmFrames += 1  // System speaking when should be silent
            case (_, nil):
                missedFrames += 1  // System silent when should be speaking
            case (let gt?, let pred?):
                // Use streaming mapping if available, otherwise treat as error
                let mappedPred = firstOccurrenceMap[pred]
                if mappedPred != gt {
                    speakerErrorFrames += 1
                }
            }
        }

        // Calculate JER (Jaccard Error Rate) with streaming mapping
        // JER uses Jaccard similarity: intersection over union of speaker sets
        var totalJaccardScore: Float = 0
        var activeFrames = 0

        for frame in 0..<totalFrames {
            let frameTime = Float(frame) * frameSize

            var gtSpeakers = Set<String>()
            for segment in groundTruth {
                if frameTime >= segment.startTimeSeconds && frameTime < segment.endTimeSeconds {
                    gtSpeakers.insert(segment.speakerId)
                }
            }

            var predSpeakers = Set<String>()
            for segment in predicted {
                if frameTime >= segment.startTimeSeconds && frameTime < segment.endTimeSeconds {
                    if let mapped = firstOccurrenceMap[segment.speakerId] {
                        predSpeakers.insert(mapped)
                    }
                }
            }

            // Only calculate Jaccard for frames where at least one system detects speech
            if !gtSpeakers.isEmpty || !predSpeakers.isEmpty {
                activeFrames += 1

                // Calculate Jaccard index for this frame
                let intersection = gtSpeakers.intersection(predSpeakers)
                let union = gtSpeakers.union(predSpeakers)

                let frameJaccard = union.isEmpty ? 0 : Float(intersection.count) / Float(union.count)
                totalJaccardScore += frameJaccard
            }
        }

        let averageJaccard = activeFrames > 0 ? totalJaccardScore / Float(activeFrames) : 0
        let jer = (1.0 - averageJaccard) * 100.0

        // Debug JER calculation
        if true {  // Enable debug output
            logger.debug(
                "🔍 JER Debug: Active frames: \(activeFrames)/\(totalFrames), Avg Jaccard: \(String(format: "%.3f", averageJaccard))"
            )

            // Count frame types for analysis
            var perfectFrames = 0
            var partialFrames = 0
            var missedFrames = 0
            var falseAlarmFrames = 0

            for frame in 0..<totalFrames {
                let frameTime = Float(frame) * frameSize
                var gtSpeakers = Set<String>()
                for segment in groundTruth {
                    if frameTime >= segment.startTimeSeconds && frameTime < segment.endTimeSeconds {
                        gtSpeakers.insert(segment.speakerId)
                    }
                }
                var predSpeakers = Set<String>()
                for segment in predicted {
                    if frameTime >= segment.startTimeSeconds && frameTime < segment.endTimeSeconds {
                        if let mapped = firstOccurrenceMap[segment.speakerId] {
                            predSpeakers.insert(mapped)
                        }
                    }
                }

                if gtSpeakers == predSpeakers && !gtSpeakers.isEmpty {
                    perfectFrames += 1
                } else if !gtSpeakers.isEmpty && predSpeakers.isEmpty {
                    missedFrames += 1
                } else if gtSpeakers.isEmpty && !predSpeakers.isEmpty {
                    falseAlarmFrames += 1
                } else if !gtSpeakers.intersection(predSpeakers).isEmpty {
                    partialFrames += 1
                }
            }

            logger.debug(
                "   Perfect match frames: \(perfectFrames) (\(String(format: "%.1f", Float(perfectFrames)/Float(totalFrames)*100))%)"
            )
            logger.debug(
                "   Partial match frames: \(partialFrames) (\(String(format: "%.1f", Float(partialFrames)/Float(totalFrames)*100))%)"
            )
            logger.debug(
                "   Missed speech frames: \(missedFrames) (\(String(format: "%.1f", Float(missedFrames)/Float(totalFrames)*100))%)"
            )
            logger.debug(
                "   False alarm frames: \(falseAlarmFrames) (\(String(format: "%.1f", Float(falseAlarmFrames)/Float(totalFrames)*100))%)"
            )
        }

        // Calculate rates
        let missRate = (Float(missedFrames) / Float(totalFrames)) * 100.0
        let falseAlarmRate = (Float(falseAlarmFrames) / Float(totalFrames)) * 100.0
        let speakerErrorRate = (Float(speakerErrorFrames) / Float(totalFrames)) * 100.0
        let der = missRate + falseAlarmRate + speakerErrorRate

        logger.info(
            "📊 STREAMING METRICS: DER=\(String(format: "%.1f", der))% (Miss=\(String(format: "%.1f", missRate))%, FA=\(String(format: "%.1f", falseAlarmRate))%, SE=\(String(format: "%.1f", speakerErrorRate))%)"
        )

        return (
            der: der, missRate: missRate, falseAlarmRate: falseAlarmRate, speakerErrorRate: speakerErrorRate, jer: jer
        )
    }

    private static func calculateFragmentation(
        speakerAppearances: [String: [Int]],
        totalChunks: Int
    ) -> Float {
        // Calculate how fragmented speaker IDs are
        // 1.0 = perfect (each speaker has one continuous segment)
        // >1.0 = fragmented (speakers appear in non-consecutive chunks)

        guard !speakerAppearances.isEmpty else { return 1.0 }

        var totalFragments = 0

        for (_, chunks) in speakerAppearances {
            guard !chunks.isEmpty else { continue }

            let sortedChunks = chunks.sorted()
            var fragments = 1

            for i in 1..<sortedChunks.count {
                // If chunks are not consecutive, it's a new fragment
                if sortedChunks[i] - sortedChunks[i - 1] > 1 {
                    fragments += 1
                }
            }

            totalFragments += fragments
        }

        // Ideal is 1 fragment per speaker
        let idealFragments = speakerAppearances.count
        return Float(totalFragments) / Float(max(idealFragments, 1))
    }

    private static func getAMIFiles(dataset: String, maxFiles: Int?) -> [String] {
        // Get list of AMI meeting names
        // Official AMI SDM test set (NeMo/pyannote eval convention).
        // Single source of truth lives on DatasetDownloader.
        let allMeetings = DatasetDownloader.officialAMITestSet

        // Filter existing files
        var availableMeetings: [String] = []
        for meeting in allMeetings {
            let path = getAudioPath(for: meeting)
            if FileManager.default.fileExists(atPath: path) {
                availableMeetings.append(meeting)
            }
        }

        // Limit if requested
        if let max = maxFiles {
            return Array(availableMeetings.prefix(max))
        }

        return availableMeetings
    }

    private static func getAudioPath(for meeting: String) -> String {
        let homeDir = FileManager.default.homeDirectoryForCurrentUser
        return homeDir.appendingPathComponent(
            "FluidAudioDatasets/ami_official/sdm/\(meeting).Mix-Headset.wav"
        ).path
    }

    private static func loadAudioFile(at path: String) async throws -> [Float] {
        let converter = AudioConverter()
        return try converter.resampleAudioFile(path: path)
    }

    private static func averageResults(_ results: [BenchmarkResult]) -> BenchmarkResult {
        guard !results.isEmpty else {
            fatalError("Cannot average empty results")
        }

        let count = Float(results.count)

        return BenchmarkResult(
            meetingName: results[0].meetingName,
            der: results.map { $0.der }.reduce(0, +) / count,
            missRate: results.map { $0.missRate }.reduce(0, +) / count,
            falseAlarmRate: results.map { $0.falseAlarmRate }.reduce(0, +) / count,
            speakerErrorRate: results.map { $0.speakerErrorRate }.reduce(0, +) / count,
            jer: results.map { $0.jer }.reduce(0, +) / count,
            rtfx: results.map { $0.rtfx }.reduce(0, +) / count,
            processingTime: Double(results.map { Float($0.processingTime) }.reduce(0, +)) / Double(count),
            chunksProcessed: Int(Float(results.map { $0.chunksProcessed }.reduce(0, +)) / count),
            detectedSpeakers: Int(Float(results.map { $0.detectedSpeakers }.reduce(0, +)) / count),
            groundTruthSpeakers: results[0].groundTruthSpeakers,
            speakerFragmentation: results.map { $0.speakerFragmentation }.reduce(0, +) / count,
            latency90th: Double(results.map { Float($0.latency90th) }.reduce(0, +)) / Double(count),
            latency99th: Double(results.map { Float($0.latency99th) }.reduce(0, +)) / Double(count),
            // Timing averages
            modelDownloadTime: Double(results.map { Float($0.modelDownloadTime) }.reduce(0, +)) / Double(count),
            modelCompileTime: Double(results.map { Float($0.modelCompileTime) }.reduce(0, +)) / Double(count),
            audioLoadTime: Double(results.map { Float($0.audioLoadTime) }.reduce(0, +)) / Double(count),
            segmentationTime: Double(results.map { Float($0.segmentationTime) }.reduce(0, +)) / Double(count),
            embeddingTime: Double(results.map { Float($0.embeddingTime) }.reduce(0, +)) / Double(count),
            clusteringTime: Double(results.map { Float($0.clusteringTime) }.reduce(0, +)) / Double(count),
            totalInferenceTime: Double(results.map { Float($0.totalInferenceTime) }.reduce(0, +)) / Double(count)
        )
    }

    private static func standardDeviation(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 0 }

        let mean = values.reduce(0, +) / Float(values.count)
        let squaredDiffs = values.map { pow($0 - mean, 2) }
        let variance = squaredDiffs.reduce(0, +) / Float(values.count - 1)

        return sqrt(variance)
    }

    private static func printFinalSummary(results: [BenchmarkResult]) {
        guard !results.isEmpty else { return }

        logger.info("" + String(repeating: "=", count: 80))
        logger.info("DIARIZATION BENCHMARK SUMMARY")
        logger.info(String(repeating: "=", count: 80))

        // Print detailed results table sorted by DER
        logger.info("📋 Results Sorted by DER (Best → Worst):")
        logger.info(String(repeating: "-", count: 90))
        // Simple header without String(format:)
        logger.info("Meeting        DER %    JER %    Miss %     FA %     SE %   Speakers     RTFx")
        logger.info(String(repeating: "-", count: 90))

        for result in results.sorted(by: { $0.der < $1.der }) {
            let speakerInfo = "\(result.detectedSpeakers)/\(result.groundTruthSpeakers)"
            // Format meeting name to fixed width
            let meetingCol = result.meetingName.padding(toLength: 12, withPad: " ", startingAt: 0)
            let speakerCol = speakerInfo.padding(toLength: 10, withPad: " ", startingAt: 0)
            logger.info(
                String(
                    format: "%@ %8.1f %8.1f %8.1f %8.1f %8.1f %@ %8.1f",
                    meetingCol,
                    result.der,
                    result.jer,
                    result.missRate,
                    result.falseAlarmRate,
                    result.speakerErrorRate,
                    speakerCol,
                    result.rtfx))
        }
        logger.info(String(repeating: "-", count: 90))

        // Calculate aggregates and add summary row
        let avgDER = results.map { $0.der }.reduce(0, +) / Float(results.count)
        let avgJER = results.map { $0.jer }.reduce(0, +) / Float(results.count)
        let avgMiss = results.map { $0.missRate }.reduce(0, +) / Float(results.count)
        let avgFA = results.map { $0.falseAlarmRate }.reduce(0, +) / Float(results.count)
        let avgSE = results.map { $0.speakerErrorRate }.reduce(0, +) / Float(results.count)
        let avgRTFx = results.map { $0.rtfx }.reduce(0, +) / Float(results.count)

        // Print average row
        logger.info(
            String(
                format: "AVERAGE      %8.1f %8.1f %8.1f %8.1f %8.1f         - %8.1f",
                avgDER, avgJER, avgMiss, avgFA, avgSE, avgRTFx))
        logger.info(String(repeating: "=", count: 90))

        // Check against targets
        logger.info("✅ Target Check:")
        if avgDER < 30 {
            logger.info("  ✅ DER < 30% (achieved: \(String(format: "%.1f", avgDER))%)")
        } else {
            logger.info("  ❌ DER < 30% (achieved: \(String(format: "%.1f", avgDER))%)")
        }

        if avgRTFx > 1 {
            logger.info("  ✅ RTFx > 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        } else {
            logger.info("  ❌ RTFx > 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        }
    }

    private static func saveJSONResults(results: [BenchmarkResult], to path: String) {
        let jsonData = results.map { result in
            [
                "meeting": result.meetingName,
                "der": result.der,
                "missRate": result.missRate,
                "falseAlarmRate": result.falseAlarmRate,
                "speakerErrorRate": result.speakerErrorRate,
                "jer": result.jer,
                "rtfx": result.rtfx,
                "processingTime": result.processingTime,
                "chunksProcessed": result.chunksProcessed,
                "detectedSpeakers": result.detectedSpeakers,
                "groundTruthSpeakers": result.groundTruthSpeakers,
                "speakerFragmentation": result.speakerFragmentation,
                "latency90th": result.latency90th,
                "latency99th": result.latency99th,
                // Add timing breakdown
                "timings": [
                    "modelDownloadSeconds": result.modelDownloadTime,
                    "modelCompilationSeconds": result.modelCompileTime,
                    "audioLoadingSeconds": result.audioLoadTime,
                    "segmentationSeconds": result.segmentationTime,
                    "embeddingExtractionSeconds": result.embeddingTime,
                    "speakerClusteringSeconds": result.clusteringTime,
                    "totalInferenceSeconds": result.totalInferenceTime,
                    "totalProcessingSeconds": result.processingTime,
                ],
            ]
        }

        do {
            let data = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
            try data.write(to: URL(fileURLWithPath: path))
            logger.info("💾 JSON results saved to: \(path)")
        } catch {
            logger.error("❌ Failed to save JSON: \(error)")
        }
    }

    private static func saveCSVResults(results: [BenchmarkResult], to path: String) {
        var csv =
            "Meeting,DER,MissRate,FalseAlarm,SpeakerError,JER,RTFx,ProcessingTime,Chunks,DetectedSpeakers,TrueSpeakers,Fragmentation,Latency90th,Latency99th\n"

        for result in results {
            csv += "\(result.meetingName),"
            csv += "\(String(format: "%.2f", result.der)),"
            csv += "\(String(format: "%.2f", result.missRate)),"
            csv += "\(String(format: "%.2f", result.falseAlarmRate)),"
            csv += "\(String(format: "%.2f", result.speakerErrorRate)),"
            csv += "\(String(format: "%.2f", result.jer)),"
            csv += "\(String(format: "%.2f", result.rtfx)),"
            csv += "\(String(format: "%.2f", result.processingTime)),"
            csv += "\(result.chunksProcessed),"
            csv += "\(result.detectedSpeakers),"
            csv += "\(result.groundTruthSpeakers),"
            csv += "\(String(format: "%.3f", result.speakerFragmentation)),"
            csv += "\(String(format: "%.4f", result.latency90th)),"
            csv += "\(String(format: "%.4f", result.latency99th))\n"
        }

        // Add summary row
        if !results.isEmpty {
            let count = Float(results.count)
            csv += "AVERAGE,"
            csv += "\(String(format: "%.2f", results.map { $0.der }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.2f", results.map { $0.missRate }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.2f", results.map { $0.falseAlarmRate }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.2f", results.map { $0.speakerErrorRate }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.2f", results.map { $0.jer }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.2f", results.map { $0.rtfx }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.2f", results.map { Float($0.processingTime) }.reduce(0, +) / count)),"
            csv += "\(Int(results.map { $0.chunksProcessed }.reduce(0, +) / results.count)),"
            csv += "\(String(format: "%.1f", results.map { Float($0.detectedSpeakers) }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.1f", results.map { Float($0.groundTruthSpeakers) }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.3f", results.map { $0.speakerFragmentation }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.4f", results.map { Float($0.latency90th) }.reduce(0, +) / count)),"
            csv += "\(String(format: "%.4f", results.map { Float($0.latency99th) }.reduce(0, +) / count))\n"
        }

        do {
            try csv.write(to: URL(fileURLWithPath: path), atomically: true, encoding: .utf8)
            logger.info("💾 CSV results saved to: \(path)")
        } catch {
            logger.error("❌ Failed to save CSV: \(error)")
        }
    }
}
#endif
