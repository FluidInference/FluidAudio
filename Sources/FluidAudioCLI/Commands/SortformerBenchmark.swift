#if os(macOS)
import FluidAudio
import Foundation

/// Sortformer streaming diarization benchmark for evaluating real-time performance
enum SortformerBenchmark {
    private static let logger = AppLogger(category: "SortformerBench")

    typealias Dataset = DiarizationBenchmarkUtils.Dataset
    typealias BenchmarkResult = DiarizationBenchmarkUtils.BenchmarkResult

    static func printUsage() {
        print(
            """
            Sortformer Benchmark Command

            Evaluates Sortformer streaming speaker diarization on various corpora.

            Usage: fluidaudio sortformer-benchmark [options]

            Options:
                --dataset <name>         Dataset to use: ami, voxconverse, callhome (default: ami)
                --single-file <name>     Process a specific meeting (e.g., ES2004a)
                --max-files <n>          Maximum number of files to process
                --threshold <value>      Speaker activity threshold (default: 0.5)
                --model <path>           Path to Sortformer.mlpackage
                --nvidia-low-latency     Use NVIDIA 1.04s latency config (20.57% DER target)
                --nvidia-high-latency            Use NVIDIA 30.4s latency config (20.57% DER target)
                --gradient-descent       Use Gradient Descent config (downloads from HuggingFace by default)
                --hf                     Download models from HuggingFace (clears cache first)
                --local                  Use local models instead of HuggingFace (for --gradient-descent)
                --output <file>          Output JSON file for results
                --progress <file>        Progress file for resuming (default: .sortformer_progress.json)
                --resume                 Resume from previous progress file
                --verbose                Enable verbose output
                --debug                  Enable debug mode
                --auto-download          Auto-download AMI dataset if missing
                --help                   Show this help message

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
            """)
    }

    static func run(arguments: [String]) async {
        // Parse arguments
        var singleFile: String?
        var maxFiles: Int?
        var threshold: Float = 0.5
        var modelPath: String?
        var outputFile: String?
        var verbose = false
        var debugMode = false
        var autoDownload = false
        var useNvidiaLowLatency = false
        var useNvidiaHighLatency = false
        var useGradientDescent = false
        var useHuggingFace = false
        var useLocalModels = false
        var progressFile: String = ".sortformer_progress.json"
        var resumeFromProgress = false
        var dataset: Dataset = .ami

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
                useGradientDescent = true
            case "--hf":
                useHuggingFace = true
            case "--local":
                useLocalModels = true
            case "--help":
                printUsage()
                return
            default:
                logger.warning("Unknown argument: \(arguments[i])")
            }
            i += 1
        }

        // Gradient descent uses HuggingFace by default unless --local is specified
        if useGradientDescent && !useLocalModels {
            useHuggingFace = true
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
            ? "HuggingFace models" : "Combined Pipeline"
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

        // Check models exist
        guard useHuggingFace || FileManager.default.fileExists(atPath: pipelineURL.path) else {
            print("ERROR: Pipeline model not found: \(pipelineURL.path)")
            return
        }

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
            config = SortformerConfig.default
        }
        config.debugMode = debugMode
        config.predScoreThreshold = threshold
        let diarizer = SortformerDiarizer(config: config)

        do {
            if useHuggingFace {
                let models = try await SortformerModels.loadFromHuggingFace(config: config)
                diarizer.initialize(models: models)
            } else {
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

    private static func processMeeting(
        meetingName: String,
        dataset: Dataset,
        diarizer: SortformerDiarizer,
        modelLoadTime: Double,
        threshold: Float,
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
            let result = try diarizer.processComplete(audioSamples) { processed, total, chunks in
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

            // Fall back to AMI XML annotations if no RTTM available (AMI only)
            if groundTruth.isEmpty && dataset == .ami {
                print("   [RTTM] No RTTM file, falling back to AMI annotations")
                groundTruth = await AMIParser.loadAMIGroundTruth(
                    for: meetingName,
                    duration: duration
                )
            }

            guard !groundTruth.isEmpty else {
                print("No ground truth found for \(meetingName)")
                return nil
            }

            // Get filtered predictions for simple DER calculation (matches Python/NeMo)
            let filteredPredictions = result.finalizedPredictions

            // Calculate DER using simple frame-level approach (matches NeMo evaluation)
            // Frame shift is 0.08s (80ms) to match NeMo's subsampling_factor * window_stride
            let simpleMetrics = calculateSimpleDER(
                predictions: filteredPredictions,
                numFrames: result.numFinalizedFrames,
                numSpeakers: result.config.numSpeakers,
                groundTruth: groundTruth,
                threshold: threshold,
                frameShift: 0.08  // 80ms frames like NeMo
            )

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
                der: simpleMetrics.der,
                missRate: simpleMetrics.miss,
                falseAlarmRate: simpleMetrics.fa,
                speakerErrorRate: simpleMetrics.se,
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

    // MARK: - Simple Frame-Level DER (matches Python's calculation)

    /// Calculate DER using simple frame-level binary comparison like Python
    /// This matches the NeMo evaluation approach without collar or complex segment overlap
    private static func calculateSimpleDER(
        predictions: [Float],
        numFrames: Int,
        numSpeakers: Int,
        groundTruth: [TimedSpeakerSegment],
        threshold: Float,
        frameShift: Float  // 0.08 for 80ms frames
    ) -> (der: Float, miss: Float, fa: Float, se: Float) {
        // Create reference binary matrix [numFrames, numSpeakers]
        var refBinary = [[Float]](repeating: [Float](repeating: 0.0, count: numSpeakers), count: numFrames)

        // Map ground truth speakers to indices
        let speakerLabels = Array(Set(groundTruth.map { $0.speakerId })).sorted()
        var speakerMap = [String: Int]()
        for (idx, label) in speakerLabels.enumerated() {
            if idx < numSpeakers {
                speakerMap[label] = idx
            }
        }

        // Fill reference binary from ground truth segments
        for segment in groundTruth {
            guard let spkIdx = speakerMap[segment.speakerId] else { continue }
            let startFrame = max(0, min(Int(segment.startTimeSeconds / frameShift), numFrames))
            let endFrame = max(0, min(Int(segment.endTimeSeconds / frameShift), numFrames))
            for frame in startFrame..<endFrame {
                refBinary[frame][spkIdx] = 1.0
            }
        }

        // Create prediction binary matrix
        var predBinary = [[Float]](repeating: [Float](repeating: 0.0, count: numSpeakers), count: numFrames)
        for frame in 0..<numFrames {
            for spk in 0..<numSpeakers {
                let idx = frame * numSpeakers + spk
                if idx < predictions.count {
                    predBinary[frame][spk] = predictions[idx] > threshold ? 1.0 : 0.0
                }
            }
        }

        // Try all permutations to find best DER
        let permutations = generatePermutations(numSpeakers)
        var bestDER: Float = .infinity
        var bestMiss: Float = 0
        var bestFA: Float = 0
        var bestSE: Float = 0

        for perm in permutations {
            var missFrames: Float = 0
            var faFrames: Float = 0
            var seFrames: Float = 0
            var totalRefSpeech: Float = 0

            for frame in 0..<numFrames {
                let refSpeech = refBinary[frame].contains(where: { $0 > 0 })
                var predSpeechPermuted = false
                for spk in 0..<numSpeakers {
                    if predBinary[frame][perm[spk]] > 0 {
                        predSpeechPermuted = true
                        break
                    }
                }

                if refSpeech {
                    totalRefSpeech += 1
                }

                if refSpeech && !predSpeechPermuted {
                    missFrames += 1
                } else if !refSpeech && predSpeechPermuted {
                    faFrames += 1
                } else if refSpeech && predSpeechPermuted {
                    // Calculate speaker error
                    var refSpks = Set<Int>()
                    var predSpks = Set<Int>()
                    for spk in 0..<numSpeakers {
                        if refBinary[frame][spk] > 0 {
                            refSpks.insert(spk)
                        }
                        if predBinary[frame][perm[spk]] > 0 {
                            predSpks.insert(spk)
                        }
                    }
                    let symDiff = refSpks.symmetricDifference(predSpks)
                    seFrames += Float(symDiff.count) / 2.0
                }
            }

            if totalRefSpeech > 0 {
                let der = (missFrames + faFrames + seFrames) / totalRefSpeech * 100
                if der < bestDER {
                    bestDER = der
                    bestMiss = missFrames / totalRefSpeech * 100
                    bestFA = faFrames / totalRefSpeech * 100
                    bestSE = seFrames / totalRefSpeech * 100
                }
            }
        }

        return (bestDER, bestMiss, bestFA, bestSE)
    }

    /// Generate all permutations of 0..<n
    private static func generatePermutations(_ n: Int) -> [[Int]] {
        if n == 0 { return [[]] }
        if n == 1 { return [[0]] }

        var result: [[Int]] = []
        var arr = Array(0..<n)

        func permute(_ start: Int) {
            if start == n {
                result.append(arr)
                return
            }
            for i in start..<n {
                arr.swapAt(start, i)
                permute(start + 1)
                arr.swapAt(start, i)
            }
        }

        permute(0)
        return result
    }
}
#endif
