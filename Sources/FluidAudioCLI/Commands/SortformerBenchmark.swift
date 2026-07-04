#if os(macOS)
import FluidAudio
import Foundation

/// Sortformer streaming diarization benchmark for evaluating real-time performance
enum SortformerBenchmark {
    private static let logger = AppLogger(category: "SortformerBench")
    private static let derFrameStepSeconds: Double = 0.01

    typealias Dataset = DiarizationBenchmarkUtils.Dataset
    typealias BenchmarkResult = DiarizationBenchmarkUtils.BenchmarkResult

    static func printUsage() {
        let usage = """
            Sortformer Benchmark Command

            Evaluates Sortformer streaming speaker diarization on various corpora.

            Usage: fluidaudio sortformer-benchmark [options]

            Options:
                --dataset <name>            Dataset to use: ami, voxconverse, callhome (default: ami)
                --single-file <name>        Process a specific meeting (e.g., ES2004a)
                --max-files <n>             Maximum number of files to process
                --threshold <value>         Speaker activity threshold (default: 0.5)
                --silence-threshold <0-1>   Silence detection threshold (default: 0.2)
                --scores-boost-latest <fl>  Boost factor for latest frames (default: 0.05)
                --strong-boost-rate <0-1>   Strong boost rate (default: 0.75)
                --weak-boost-rate <fl>      Weak boost rate (default: 1.5)
                --min-pos-scores-rate <0-1> Minimum positive scores rate (default: 0.5)
                --spkcache-sil-frames <n>   Silence frames per speaker in cache (default: 3)
                --model <path>              Path to Sortformer.mlpackage
                --nvidia-low-latency        Use NVIDIA 1.04s latency config
                --nvidia-high-latency       Use NVIDIA 30.4s latency config
                --gradient-descent          Use Gradient Descent config (default)
                --hf                        Use HuggingFace/cache-backed model loading
                --local                     Use local mlpackage loading
                --output <file>             Output JSON file for results
                --progress <file>           Progress file for resuming (default: .sortformer_progress.json)
                --resume                    Resume from previous progress file
                --verbose                   Enable verbose output
                --debug                     Enable debug mode
                --auto-download             Auto-download AMI dataset if missing
                --help                      Show this help message

            Performance Targets:
                DER ~11%   (NVIDIA benchmark on DI-HARD III)
                RTFx > 1x  (real-time capable)

            Examples:
                # Quick test on one file
                fluidaudio sortformer-benchmark --single-file ES2004a

                # Full AMI benchmark
                fluidaudio sortformer-benchmark --auto-download --output results.json

                # Test with custom model paths
                fluidaudio sortformer-benchmark --single-file ES2004a \\
                    --preprocessor ./models/SortformerPreprocessor.mlpackage \\
                    --model ./models/Sortformer.mlpackage
            """
        fputs(usage, stderr)
        fflush(stderr)
    }

    static func run(arguments: [String]) async {
        // Parse arguments
        var singleFile: String?
        var maxFiles: Int?
        var threshold: Float = 0.5
        var collarSeconds: Double = 0
        var onsetThreshold: Float?
        var offsetThreshold: Float?
        var modelPath: String?
        var outputFile: String?
        var verbose = false
        var debugMode = false
        var autoDownload = false
        var useNvidiaLowLatency = false
        var useNvidiaHighLatency = false
        var useHuggingFace = true
        var useLocalModels = false
        var offline = false
        var palettized = false
        var progressFile: String = ".sortformer_progress.json"
        var resumeFromProgress = false
        var dataset: Dataset = .ami

        // SortformerConfig tuning fields
        var silenceThreshold: Float?
        var scoresBoostLatest: Float?
        var strongBoostRate: Float?
        var weakBoostRate: Float?
        var minPosScoresRate: Float?
        var spkcacheSilFramesPerSpk: Int?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    if let d = Dataset(rawValue: arguments[i + 1].lowercased()) {
                        dataset = d
                    } else {
                        print("Unknown dataset: \(arguments[i + 1]). Using ami.")
                    }
                    i += 1
                }
            case "--single-file":
                if i + 1 < arguments.count {
                    singleFile = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.5
                    i += 1
                }
            case "--collar":
                if i + 1 < arguments.count {
                    collarSeconds = Double(arguments[i + 1]) ?? 0
                    i += 1
                }
            case "--onset":
                if i + 1 < arguments.count {
                    onsetThreshold = Float(arguments[i + 1])
                    i += 1
                }
            case "--offset":
                if i + 1 < arguments.count {
                    offsetThreshold = Float(arguments[i + 1])
                    i += 1
                }
            case "--model":
                if i + 1 < arguments.count {
                    modelPath = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--progress":
                if i + 1 < arguments.count {
                    progressFile = arguments[i + 1]
                    i += 1
                }
            case "--resume":
                resumeFromProgress = true
            case "--verbose":
                verbose = true
            case "--debug":
                debugMode = true
            case "--auto-download":
                autoDownload = true
            case "--nvidia-high-latency":
                useNvidiaHighLatency = true
            case "--nvidia-low-latency":
                useNvidiaLowLatency = true
            case "--gradient-descent":
                // gradient-descent uses the default SortformerConfig which is already selected
                break
            case "--silence-threshold":
                if i + 1 < arguments.count {
                    silenceThreshold = Float(arguments[i + 1])
                    i += 1
                }
            case "--scores-boost-latest":
                if i + 1 < arguments.count {
                    scoresBoostLatest = Float(arguments[i + 1])
                    i += 1
                }
            case "--strong-boost-rate":
                if i + 1 < arguments.count {
                    strongBoostRate = Float(arguments[i + 1])
                    i += 1
                }
            case "--weak-boost-rate":
                if i + 1 < arguments.count {
                    weakBoostRate = Float(arguments[i + 1])
                    i += 1
                }
            case "--min-pos-scores-rate":
                if i + 1 < arguments.count {
                    minPosScoresRate = Float(arguments[i + 1])
                    i += 1
                }
            case "--spkcache-sil-frames":
                if i + 1 < arguments.count {
                    spkcacheSilFramesPerSpk = Int(arguments[i + 1])
                    i += 1
                }
            case "--hf":
                useHuggingFace = true
            case "--local":
                useLocalModels = true
            case "--offline":
                offline = true
            case "--palettized":
                palettized = true
            case "--help":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arguments[i])")
            }
            i += 1
        }

        // Benchmarks should prefer the cache-backed Hugging Face loader by default.
        // `--local` explicitly opts out to local mlpackage loading.
        if useLocalModels {
            useHuggingFace = false
        }

        print("Starting Sortformer Benchmark")
        fflush(stdout)
        print("   Dataset: \(dataset.rawValue)")
        print("   Threshold: \(threshold)")
        let configName =
            useNvidiaLowLatency
            ? "NVIDIA 1.04s" : (useNvidiaHighLatency ? "NVIDIA 30.4s" : "Gradient Descent")
        print("   Config: \(configName)")

        let modeDesc =
            useHuggingFace
            ? "HuggingFace/cache-backed models"
            : "Local mlpackage"
        print("   Mode: \(modeDesc)")
        print("   Preprocessing: Native Swift mel spectrogram")

        // Default model paths based on config
        // Different configs need different models with matching input dimensions
        let modelDir: String
        if useNvidiaHighLatency {
            modelDir = "Streaming-Sortformer-Conversion/nvidia-high"
        } else if useNvidiaLowLatency {
            modelDir = "Streaming-Sortformer-Conversion/nvidia-low"
        } else {
            modelDir = "Streaming-Sortformer-Conversion/gradient-descent"
        }

        let defaultPipeline = "\(modelDir)/Sortformer.mlpackage"
        let pipelineURL = URL(fileURLWithPath: modelPath ?? defaultPipeline)

        print("   Pipeline: \(pipelineURL.path)")

        // Download dataset if needed
        if autoDownload && dataset == .ami {
            print("Downloading AMI dataset if needed...")
            await DatasetDownloader.downloadAMIDataset(
                variant: .sdm,
                force: false,
                singleFile: singleFile
            )
            await DatasetDownloader.downloadAMIAnnotations(force: false)
        }

        // Get list of files to process
        let filesToProcess: [String]
        if let meeting = singleFile {
            filesToProcess = [meeting]
        } else {
            filesToProcess = DiarizationBenchmarkUtils.getFiles(for: dataset, maxFiles: maxFiles)
        }

        if filesToProcess.isEmpty {
            print("No files found to process")
            fflush(stdout)
            return
        }

        print("Processing \(filesToProcess.count) file(s)")
        print("   Progress file: \(progressFile)")
        fflush(stdout)

        // Load previous progress if resuming
        var completedResults: [BenchmarkResult] = []
        var completedMeetings: Set<String> = []
        if resumeFromProgress {
            if let loaded = DiarizationBenchmarkUtils.loadProgress(from: progressFile) {
                completedResults = loaded
                completedMeetings = Set(loaded.map { $0.meetingName })
                print("Resuming: loaded \(completedResults.count) previous results")
                for result in completedResults {
                    print("   \(result.meetingName): \(String(format: "%.1f", result.der))% DER")
                }
            } else {
                print("No previous progress found, starting fresh")
            }
        }
        print("")
        fflush(stdout)

        // Offline (whole-file fused model) path — no streaming state, one fused call per window.
        if offline {
            await runOfflineBenchmark(
                filesToProcess: filesToProcess,
                completedResults: completedResults,
                completedMeetings: completedMeetings,
                dataset: dataset,
                palettized: palettized,
                modelPath: modelPath,
                useHuggingFace: useHuggingFace,
                threshold: threshold,
                collarSeconds: collarSeconds,
                verbose: verbose,
                progressFile: progressFile,
                outputFile: outputFile)
            return
        }

        // Initialize Sortformer
        print("Loading Sortformer models...")
        fflush(stdout)
        let modelLoadStart = Date()
        var config: SortformerConfig
        if useNvidiaHighLatency {
            config = SortformerConfig.highContextV2_1
        } else if useNvidiaLowLatency {
            config = SortformerConfig.balancedV2_1
        } else {
            config = SortformerConfig.default  // gradient-descent uses the default
        }
        config.debugMode = debugMode
        config.predScoreThreshold = threshold
        if let v = silenceThreshold { config.silenceThreshold = v }
        if let v = scoresBoostLatest { config.scoresBoostLatest = v }
        if let v = strongBoostRate { config.strongBoostRate = v }
        if let v = weakBoostRate { config.weakBoostRate = v }
        if let v = minPosScoresRate { config.minPosScoresRate = v }
        if let v = spkcacheSilFramesPerSpk { config.spkcacheSilFramesPerSpk = v }
        // Allow overriding the timeline binarization thresholds (sortformerDefault = 0.5/0.5).
        let diarizer: SortformerDiarizer
        if onsetThreshold != nil || offsetThreshold != nil {
            let timeline = DiarizerTimelineConfig(
                numSpeakers: config.numSpeakers,
                frameDurationSeconds: Float(config.frameDurationSeconds),
                onsetThreshold: onsetThreshold ?? 0.5,
                offsetThreshold: offsetThreshold ?? onsetThreshold ?? 0.5
            )
            diarizer = SortformerDiarizer(config: config, timelineConfig: timeline)
        } else {
            diarizer = SortformerDiarizer(config: config)
        }

        do {
            if useHuggingFace {
                let models = try await SortformerModels.loadFromHuggingFace(config: config)
                diarizer.initialize(models: models)
            } else {
                guard FileManager.default.fileExists(atPath: pipelineURL.path) else {
                    print("ERROR: Local pipeline model not found: \(pipelineURL.path)")
                    return
                }
                try await diarizer.initialize(
                    mainModelPath: pipelineURL
                )
            }
        } catch {
            print("Failed to initialize Sortformer: \(error)")
            return
        }

        let modelLoadTime = Date().timeIntervalSince(modelLoadStart)
        print("Models loaded in \(String(format: "%.2f", modelLoadTime))s\n")
        fflush(stdout)

        // Process each file
        var allResults: [BenchmarkResult] = completedResults

        for (fileIndex, meetingName) in filesToProcess.enumerated() {
            // Skip already completed files
            if completedMeetings.contains(meetingName) {
                print("[\(fileIndex + 1)/\(filesToProcess.count)] Skipping (already done): \(meetingName)")
                fflush(stdout)
                continue
            }

            print(String(repeating: "=", count: 60))
            print("[\(fileIndex + 1)/\(filesToProcess.count)] Processing: \(meetingName)")
            print(String(repeating: "=", count: 60))
            fflush(stdout)

            let result = await processMeeting(
                meetingName: meetingName,
                dataset: dataset,
                diarizer: diarizer,
                modelLoadTime: modelLoadTime,
                threshold: threshold,
                collarSeconds: collarSeconds,
                verbose: verbose
            )

            if let result = result {
                allResults.append(result)

                // Print summary
                print("Results for \(meetingName):")
                print("   DER: \(String(format: "%.1f", result.der))%")
                print("   RTFx: \(String(format: "%.1f", result.rtfx))x")
                print("   Speakers: \(result.detectedSpeakers) detected / \(result.groundTruthSpeakers) truth")

                // Save progress after each file
                DiarizationBenchmarkUtils.saveProgress(results: allResults, to: progressFile)
                print("Progress saved (\(allResults.count) files complete)")
            }
            fflush(stdout)

            // Reset diarizer state for next file
            diarizer.reset()
        }

        // Print final summary
        DiarizationBenchmarkUtils.printFinalSummary(
            results: allResults,
            title: "SORTFORMER BENCHMARK SUMMARY",
            derTargets: [15, 20]
        )

        // Save results
        if let outputPath = outputFile {
            DiarizationBenchmarkUtils.saveJSONResults(results: allResults, to: outputPath)
        }
    }

    /// Whole-file offline benchmark over the dataset using the fused offline Sortformer model.
    private static func runOfflineBenchmark(
        filesToProcess: [String],
        completedResults: [BenchmarkResult],
        completedMeetings: Set<String>,
        dataset: Dataset,
        palettized: Bool,
        modelPath: String?,
        useHuggingFace: Bool,
        threshold: Float,
        collarSeconds: Double,
        verbose: Bool,
        progressFile: String,
        outputFile: String?
    ) async {
        print("Loading offline Sortformer model...")
        fflush(stdout)
        let modelLoadStart = Date()

        var offlineConfig = OfflineSortformerConfig.offlineV2_1
        if palettized { offlineConfig.precision = .palettized }
        let diarizer = OfflineSortformerDiarizer(config: offlineConfig)

        do {
            if let modelPath = modelPath, !useHuggingFace {
                try await diarizer.initialize(modelPath: URL(fileURLWithPath: modelPath))
            } else {
                try await diarizer.initializeFromHuggingFace()
            }
        } catch {
            print("Failed to initialize offline Sortformer: \(error)")
            return
        }
        let modelLoadTime = Date().timeIntervalSince(modelLoadStart)
        print(
            "Model loaded in \(String(format: "%.2f", modelLoadTime))s (precision: \(offlineConfig.precision.rawValue))\n"
        )
        fflush(stdout)

        var allResults: [BenchmarkResult] = completedResults
        for (fileIndex, meetingName) in filesToProcess.enumerated() {
            if completedMeetings.contains(meetingName) {
                print("[\(fileIndex + 1)/\(filesToProcess.count)] Skipping (already done): \(meetingName)")
                continue
            }
            print(String(repeating: "=", count: 60))
            print("[\(fileIndex + 1)/\(filesToProcess.count)] Processing (offline): \(meetingName)")
            print(String(repeating: "=", count: 60))
            fflush(stdout)

            let result = await processMeeting(
                meetingName: meetingName,
                dataset: dataset,
                diarizer: nil,
                offlineDiarizer: diarizer,
                modelLoadTime: modelLoadTime,
                threshold: threshold,
                collarSeconds: collarSeconds,
                verbose: verbose)

            if let result = result {
                allResults.append(result)
                print("Results for \(meetingName):")
                print("   DER: \(String(format: "%.1f", result.der))%")
                print("   RTFx: \(String(format: "%.1f", result.rtfx))x")
                print("   Speakers: \(result.detectedSpeakers) detected / \(result.groundTruthSpeakers) truth")
                DiarizationBenchmarkUtils.saveProgress(results: allResults, to: progressFile)
            }
            fflush(stdout)
        }

        DiarizationBenchmarkUtils.printFinalSummary(
            results: allResults,
            title: "SORTFORMER OFFLINE BENCHMARK SUMMARY",
            derTargets: [15, 20])

        if let outputPath = outputFile {
            DiarizationBenchmarkUtils.saveJSONResults(results: allResults, to: outputPath)
        }
    }

    private static func processMeeting(
        meetingName: String,
        dataset: Dataset,
        diarizer: SortformerDiarizer?,
        offlineDiarizer: OfflineSortformerDiarizer? = nil,
        modelLoadTime: Double,
        threshold: Float,
        collarSeconds: Double,
        verbose: Bool
    ) async -> BenchmarkResult? {

        let audioPath = DiarizationBenchmarkUtils.getAudioPath(for: meetingName, dataset: dataset)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("Audio file not found: \(audioPath)")
            fflush(stdout)
            return nil
        }

        do {
            // Load audio
            let audioLoadStart = Date()
            let audioSamples = try AudioConverter().resampleAudioFile(path: audioPath)
            let audioLoadTime = Date().timeIntervalSince(audioLoadStart)
            let duration = Float(audioSamples.count) / 16000.0

            print("   Audio samples: \(audioSamples.count), duration: \(String(format: "%.1f", duration))s")
            fflush(stdout)
            if verbose {
                print("   Audio load time: \(String(format: "%.3f", audioLoadTime))s")
                fflush(stdout)
            }

            // Process with progress reporting
            let startTime = Date()
            var lastProgressPrint = Date()
            let result: DiarizerTimeline
            if let offlineDiarizer = offlineDiarizer {
                // Offline: one fused call per window, no streaming progress callback.
                result = try offlineDiarizer.processComplete(audioSamples)
            } else if let diarizer = diarizer {
                result = try diarizer.processComplete(audioSamples) { processed, total, chunks in
                    // Print progress every 2 seconds
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
            } else {
                print("No diarizer provided")
                return nil
            }
            let processingTime = Date().timeIntervalSince(startTime)

            let rtfx = duration / Float(processingTime)
            if verbose {
                print("   Processing time: \(String(format: "%.2f", processingTime))s")
                print("   RTFx: \(String(format: "%.1f", rtfx))x")
                print("   Total frames: \(result.numFinalizedFrames)")
            }

            // Extract segments
            var segments: [[DiarizerSegment]] = Array(repeating: [], count: result.config.numSpeakers)
            for (index, speaker) in result.speakers {
                segments[index] = speaker.finalizedSegments
            }

            // Print probability statistics
            let preds = result.finalizedPredictions
            let count = preds.count
            let minVal = preds.min() ?? 0
            let maxVal = preds.max() ?? 0
            let meanVal = count > 0 ? preds.reduce(0, +) / Float(count) : 0
            let above05 = preds.filter { $0 > 0.5 }.count

            print(
                "   Prob stats: min=\(String(format: "%.3f", minVal)), max=\(String(format: "%.3f", maxVal)), mean=\(String(format: "%.3f", meanVal))"
            )
            print(
                "   Activity: \(above05)/\(count) frames (\(String(format: "%.1f", Float(above05) / Float(count) * 100))%) above 0.5"
            )
            print("   Extracted \(segments.count) segments")
            fflush(stdout)

            // Load ground truth from RTTM file (matches Python's approach)
            var groundTruth = loadRTTMGroundTruth(for: meetingName, dataset: dataset)

            // Fall back to AMI word-aligned annotations if no RTTM available (AMI only)
            if groundTruth.isEmpty && dataset == .ami {
                print("   [RTTM] No RTTM file, falling back to AMI word-aligned annotations")
                groundTruth = try AMIParser.loadWordAlignedGroundTruth(
                    for: meetingName,
                    duration: duration
                )
            }

            guard !groundTruth.isEmpty else {
                print("No ground truth found for \(meetingName)")
                return nil
            }

            let referenceSegments = groundTruth.map {
                DERSpeakerSegment(
                    speaker: $0.speakerId,
                    start: Double($0.startTimeSeconds),
                    end: Double($0.endTimeSeconds)
                )
            }
            let hypothesisSegments = segmentsToDERSegments(segments)
            let derResult = DiarizationDER.compute(
                ref: referenceSegments,
                hyp: hypothesisSegments,
                frameStep: derFrameStepSeconds,
                collar: collarSeconds
            )
            let totalRefSpeech = max(derResult.totalRefSpeech, .leastNonzeroMagnitude)
            let derPercent = Float(derResult.der * 100)
            let missPercent = Float(derResult.miss / totalRefSpeech * 100)
            let faPercent = Float(derResult.falseAlarm / totalRefSpeech * 100)
            let sePercent = Float(derResult.confusion / totalRefSpeech * 100)

            // Count detected speakers
            let detectedSpeakers = segments.reduce(into: Set<Int>()) {
                $0.formUnion($1.map(\.speakerIndex))
            }.count

            // Get ground truth speaker count
            let groundTruthSpeakers: Int
            switch dataset {
            case .ami:
                groundTruthSpeakers = AMIParser.getGroundTruthSpeakerCount(for: meetingName)
            case .voxconverse, .callhome:
                // Count unique speakers from ground truth
                groundTruthSpeakers = Set(groundTruth.map { $0.speakerId }).count
            }

            return BenchmarkResult(
                meetingName: meetingName,
                der: derPercent,
                missRate: missPercent,
                falseAlarmRate: faPercent,
                speakerErrorRate: sePercent,
                rtfx: rtfx,
                processingTime: processingTime,
                totalFrames: result.numFinalizedFrames,
                detectedSpeakers: detectedSpeakers,
                groundTruthSpeakers: groundTruthSpeakers,
                modelLoadTime: modelLoadTime,
                audioLoadTime: audioLoadTime
            )

        } catch {
            print("Error processing \(meetingName): \(error)")
            return nil
        }
    }

    // MARK: - RTTM Ground Truth Loading (matches Python's approach)

    /// Load ground truth from RTTM file like Python does
    /// Format: SPEAKER <meeting_id> 1 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    private static func loadRTTMGroundTruth(for meetingName: String, dataset: Dataset) -> [TimedSpeakerSegment] {
        guard let rttmURL = DiarizationBenchmarkUtils.getRTTMURL(for: meetingName, dataset: dataset) else {
            print("   [RTTM] No RTTM URL for \(meetingName)")
            return []
        }
        let rttmPath = rttmURL.path

        guard FileManager.default.fileExists(atPath: rttmPath) else {
            print("   [RTTM] File not found: \(rttmPath)")
            return []
        }

        guard let content = try? String(contentsOfFile: rttmPath, encoding: .utf8) else {
            print("   [RTTM] Failed to read file: \(rttmPath)")
            return []
        }

        var segments: [TimedSpeakerSegment] = []
        let lines = content.components(separatedBy: .newlines)

        for line in lines {
            // Split and filter out empty strings (handles multiple spaces)
            let parts = line.trimmingCharacters(in: .whitespaces)
                .components(separatedBy: .whitespaces)
                .filter { !$0.isEmpty }
            // RTTM format: SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
            guard parts.count >= 8,
                parts[0] == "SPEAKER",
                let startTime = Float(parts[3]),
                let duration = Float(parts[4])
            else {
                continue
            }

            let speakerId = parts[7]
            let endTime = startTime + duration

            segments.append(
                TimedSpeakerSegment(
                    speakerId: speakerId,
                    embedding: [],  // Not needed for DER calculation
                    startTimeSeconds: startTime,
                    endTimeSeconds: endTime,
                    qualityScore: 1.0
                ))
        }

        // Debug: show unique speakers
        let speakers = Set(segments.map { $0.speakerId })
        print("   [RTTM] Loaded \(segments.count) segments from \(rttmPath), speakers: \(speakers.sorted())")
        return segments
    }

    private static func segmentsToDERSegments(
        _ segments: [[DiarizerSegment]]
    ) -> [DERSpeakerSegment] {
        segments.flatMap { speakerSegments in
            speakerSegments.map { segment in
                DERSpeakerSegment(
                    speaker: segment.speakerLabel,
                    start: Double(segment.startTime),
                    end: Double(segment.endTime)
                )
            }
        }
    }
}
#endif
