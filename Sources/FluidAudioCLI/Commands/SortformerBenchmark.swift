#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Sortformer streaming diarization benchmark for evaluating real-time performance
enum SortformerBenchmark {
    private static let logger = AppLogger(category: "SortformerBench")

    struct BenchmarkResult {
        let meetingName: String
        let der: Float
        let missRate: Float
        let falseAlarmRate: Float
        let speakerErrorRate: Float
        let rtfx: Float
        let processingTime: Double
        let totalFrames: Int
        let detectedSpeakers: Int
        let groundTruthSpeakers: Int
        let modelLoadTime: Double
        let audioLoadTime: Double
    }

    static func printUsage() {
        print(
            """
            Sortformer Benchmark Command

            Evaluates Sortformer streaming speaker diarization on the AMI corpus.

            Usage: fluidaudio sortformer-benchmark [options]

            Options:
                --single-file <name>     Process a specific meeting (e.g., ES2004a)
                --max-files <n>          Maximum number of files to process
                --threshold <value>      Speaker activity threshold (default: 0.5)
                --preprocessor <path>    Path to SortformerPreprocessor.mlpackage
                --model <path>           Path to Sortformer.mlpackage
                --nvidia-config          Use NVIDIA 1.04s latency config (20.57% DER target)
                --low-latency            Use low-latency config (matches Python test)
                --simple-state           Use simple state update (matches Python test logic)
                --separate-models        Use separate PreEncoder+Head models (matches Python)
                --native-preprocessing   Use native Swift mel spectrogram (matches NeMo full-audio)
                --output <file>          Output JSON file for results
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
        var preprocessorPath: String?
        var modelPath: String?
        var outputFile: String?
        var verbose = false
        var debugMode = false
        var autoDownload = false
        var useNvidiaConfig = false
        var useLowLatency = false
        var useSimpleState = false
        var useSeparateModels = false
        var useNativePreprocessing = false

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
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
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--verbose":
                verbose = true
            case "--debug":
                debugMode = true
            case "--auto-download":
                autoDownload = true
            case "--nvidia-config":
                useNvidiaConfig = true
            case "--low-latency":
                useLowLatency = true
            case "--simple-state":
                useSimpleState = true
            case "--separate-models":
                useSeparateModels = true
            case "--native-preprocessing":
                useNativePreprocessing = true
            case "--help":
                printUsage()
                return
            default:
                if !arguments[i].starts(with: "--") {
                    logger.warning("Unknown argument: \(arguments[i])")
                }
            }
            i += 1
        }

        print("🚀 Starting Sortformer Benchmark")
        fflush(stdout)
        print("   Threshold: \(threshold)")
        let configName = useNvidiaConfig ? "NVIDIA 1.04s" : (useLowLatency ? "Low-latency" : "Default")
        print("   Config: \(configName)")
        print("   State Update: \(useSimpleState ? "Simple (Python test)" : "Full NeMo")")
        print("   Mode: \(useSeparateModels ? "Separate PreEncoder+Head" : "Combined Pipeline")")
        print("   Preprocessing: \(useNativePreprocessing ? "Native Swift mel spectrogram" : "CoreML chunked")")

        // Default model paths based on config
        // NVIDIA config uses different models with larger input dimensions
        let modelDir =
            useNvidiaConfig
            ? "Streaming-Sortformer-Conversion/coreml_models_nvidia"
            : "Streaming-Sortformer-Conversion/coreml_models"
        let defaultPreprocessor = "\(modelDir)/Pipeline_Preprocessor.mlpackage"
        let defaultPreEncoder = "\(modelDir)/Pipeline_PreEncoder.mlpackage"
        let defaultHead = "\(modelDir)/Pipeline_Head_Fixed.mlpackage"
        let defaultPipeline = "\(modelDir)/SortformerPipeline.mlpackage"

        let preprocessorURL = URL(fileURLWithPath: preprocessorPath ?? defaultPreprocessor)
        let preEncoderURL = URL(fileURLWithPath: defaultPreEncoder)
        let headURL = URL(fileURLWithPath: defaultHead)
        let pipelineURL = URL(fileURLWithPath: modelPath ?? defaultPipeline)

        print("   Preprocessor: \(preprocessorURL.path)")
        if useSeparateModels {
            print("   PreEncoder: \(preEncoderURL.path)")
            print("   Head: \(headURL.path)")
        } else {
            print("   Pipeline: \(pipelineURL.path)")
        }

        // Check models exist
        guard FileManager.default.fileExists(atPath: preprocessorURL.path) else {
            print("ERROR: Preprocessor model not found: \(preprocessorURL.path)")
            return
        }
        if useSeparateModels {
            guard FileManager.default.fileExists(atPath: preEncoderURL.path) else {
                print("ERROR: PreEncoder model not found: \(preEncoderURL.path)")
                return
            }
            guard FileManager.default.fileExists(atPath: headURL.path) else {
                print("ERROR: Head model not found: \(headURL.path)")
                return
            }
        } else {
            guard FileManager.default.fileExists(atPath: pipelineURL.path) else {
                print("ERROR: Pipeline model not found: \(pipelineURL.path)")
                return
            }
        }

        // Download dataset if needed
        if autoDownload {
            print("📥 Downloading AMI dataset if needed...")
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
            filesToProcess = getAMIFiles(maxFiles: maxFiles)
        }

        if filesToProcess.isEmpty {
            print("❌ No files found to process")
            fflush(stdout)
            return
        }

        print("📂 Processing \(filesToProcess.count) file(s)\n")
        fflush(stdout)

        // Initialize Sortformer
        print("🔧 Loading Sortformer models...")
        fflush(stdout)
        let modelLoadStart = Date()
        var config: SortformerConfig
        if useNvidiaConfig {
            config = SortformerConfig.nvidia
        } else if useLowLatency {
            config = SortformerConfig.lowLatency
        } else {
            config = SortformerConfig.default
        }
        config.debugMode = debugMode
        config.useSimpleStateUpdate = useSimpleState
        config.useNativePreprocessing = useNativePreprocessing
        let diarizer = SortformerDiarizer(config: config)

        do {
            if useSeparateModels {
                try await diarizer.initializeSeparate(
                    preprocessorPath: preprocessorURL,
                    preEncoderPath: preEncoderURL,
                    headPath: headURL
                )
            } else {
                try await diarizer.initialize(
                    preprocessorPath: preprocessorURL,
                    mainModelPath: pipelineURL
                )
            }
        } catch {
            print("❌ Failed to initialize Sortformer: \(error)")
            return
        }

        let modelLoadTime = Date().timeIntervalSince(modelLoadStart)
        print("✅ Models loaded in \(String(format: "%.2f", modelLoadTime))s\n")
        fflush(stdout)

        // Process each file
        var allResults: [BenchmarkResult] = []

        for (fileIndex, meetingName) in filesToProcess.enumerated() {
            print(String(repeating: "=", count: 60))
            print("[\(fileIndex + 1)/\(filesToProcess.count)] Processing: \(meetingName)")
            print(String(repeating: "=", count: 60))
            fflush(stdout)

            let result = await processMeeting(
                meetingName: meetingName,
                diarizer: diarizer,
                modelLoadTime: modelLoadTime,
                threshold: threshold,
                verbose: verbose
            )

            if let result = result {
                allResults.append(result)

                // Print summary
                print("📊 Results for \(meetingName):")
                print("   DER: \(String(format: "%.1f", result.der))%")
                print("   RTFx: \(String(format: "%.1f", result.rtfx))x")
                print("   Speakers: \(result.detectedSpeakers) detected / \(result.groundTruthSpeakers) truth")
            }

            // Reset diarizer state for next file
            diarizer.reset()
        }

        // Print final summary
        printFinalSummary(results: allResults)

        // Save results
        if let outputPath = outputFile {
            saveJSONResults(results: allResults, to: outputPath)
        }
    }

    private static func processMeeting(
        meetingName: String,
        diarizer: SortformerDiarizer,
        modelLoadTime: Double,
        threshold: Float,
        verbose: Bool
    ) async -> BenchmarkResult? {

        let audioPath = getAudioPath(for: meetingName)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("❌ Audio file not found: \(audioPath)")
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
                print("   Total frames: \(result.totalFrames)")
            }

            // Extract segments
            let segments = result.extractSegments(threshold: threshold)

            // Print probability statistics
            let stats = result.probabilityStats()
            print(
                "   Prob stats: min=\(String(format: "%.3f", stats.min)), max=\(String(format: "%.3f", stats.max)), mean=\(String(format: "%.3f", stats.mean))"
            )
            print(
                "   Activity: \(stats.above05)/\(stats.total) frames (\(String(format: "%.1f", Float(stats.above05) / Float(stats.total) * 100))%) above 0.5"
            )
            print("   Extracted \(segments.count) segments")
            fflush(stdout)

            // Load ground truth from RTTM file (matches Python's approach)
            var groundTruth = loadRTTMGroundTruth(for: meetingName)

            // Fall back to AMI XML annotations if no RTTM available
            if groundTruth.isEmpty {
                print("   [RTTM] No RTTM file, falling back to AMI annotations")
                groundTruth = await AMIParser.loadAMIGroundTruth(
                    for: meetingName,
                    duration: duration
                )
            }

            guard !groundTruth.isEmpty else {
                print("⚠️ No ground truth found for \(meetingName)")
                return nil
            }

            // Get filtered predictions for simple DER calculation (matches Python/NeMo)
            let filteredPredictions = result.applyMedianFilter(kernel: 7, numSpeakers: 4)

            // Calculate DER using simple frame-level approach (matches NeMo evaluation)
            // Frame shift is 0.08s (80ms) to match NeMo's subsampling_factor * window_stride
            let simpleMetrics = calculateSimpleDER(
                predictions: filteredPredictions,
                numFrames: result.totalFrames,
                numSpeakers: 4,
                groundTruth: groundTruth,
                threshold: threshold,
                frameShift: 0.08  // 80ms frames like NeMo
            )

            // Count detected speakers
            let detectedSpeakers = Set(segments.map { $0.speakerIndex }).count

            return BenchmarkResult(
                meetingName: meetingName,
                der: simpleMetrics.der,
                missRate: simpleMetrics.miss,
                falseAlarmRate: simpleMetrics.fa,
                speakerErrorRate: simpleMetrics.se,
                rtfx: rtfx,
                processingTime: processingTime,
                totalFrames: result.totalFrames,
                detectedSpeakers: detectedSpeakers,
                groundTruthSpeakers: AMIParser.getGroundTruthSpeakerCount(for: meetingName),
                modelLoadTime: modelLoadTime,
                audioLoadTime: audioLoadTime
            )

        } catch {
            print("❌ Error processing \(meetingName): \(error)")
            return nil
        }
    }

    private static func getAMIFiles(maxFiles: Int?) -> [String] {
        let allMeetings = [
            "ES2002a", "ES2002b", "ES2002c", "ES2002d",
            "ES2003a", "ES2003b", "ES2003c", "ES2003d",
            "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "ES2005a", "ES2005b", "ES2005c", "ES2005d",
            "ES2006a", "ES2006b", "ES2006c", "ES2006d",
            "ES2007a", "ES2007b", "ES2007c", "ES2007d",
            "ES2008a", "ES2008b", "ES2008c", "ES2008d",
        ]

        var availableMeetings: [String] = []
        for meeting in allMeetings {
            let path = getAudioPath(for: meeting)
            if FileManager.default.fileExists(atPath: path) {
                availableMeetings.append(meeting)
            }
        }

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

    private static func printFinalSummary(results: [BenchmarkResult]) {
        guard !results.isEmpty else { return }

        print("\n" + String(repeating: "=", count: 80))
        print("SORTFORMER BENCHMARK SUMMARY")
        print(String(repeating: "=", count: 80))

        print("📋 Results Sorted by DER:")
        print(String(repeating: "-", count: 70))
        print("Meeting        DER %    Miss %     FA %     SE %   Speakers     RTFx")
        print(String(repeating: "-", count: 70))

        for result in results.sorted(by: { $0.der < $1.der }) {
            let speakerInfo = "\(result.detectedSpeakers)/\(result.groundTruthSpeakers)"
            let meetingCol = result.meetingName.padding(toLength: 12, withPad: " ", startingAt: 0)
            let speakerCol = speakerInfo.padding(toLength: 10, withPad: " ", startingAt: 0)
            print(
                String(
                    format: "%@ %8.1f %8.1f %8.1f %8.1f %@ %8.1f",
                    meetingCol,
                    result.der,
                    result.missRate,
                    result.falseAlarmRate,
                    result.speakerErrorRate,
                    speakerCol,
                    result.rtfx))
        }
        print(String(repeating: "-", count: 70))

        let count = Float(results.count)
        let avgDER = results.map { $0.der }.reduce(0, +) / count
        let avgMiss = results.map { $0.missRate }.reduce(0, +) / count
        let avgFA = results.map { $0.falseAlarmRate }.reduce(0, +) / count
        let avgSE = results.map { $0.speakerErrorRate }.reduce(0, +) / count
        let avgRTFx = results.map { $0.rtfx }.reduce(0, +) / count

        print(
            String(
                format: "AVERAGE      %8.1f %8.1f %8.1f %8.1f         - %8.1f",
                avgDER, avgMiss, avgFA, avgSE, avgRTFx))
        print(String(repeating: "=", count: 70))

        print("\n✅ Target Check:")
        if avgDER < 15 {
            print("   ✅ DER < 15% (achieved: \(String(format: "%.1f", avgDER))%)")
        } else if avgDER < 20 {
            print("   🟡 DER < 20% (achieved: \(String(format: "%.1f", avgDER))%)")
        } else {
            print("   ❌ DER > 20% (achieved: \(String(format: "%.1f", avgDER))%)")
        }

        if avgRTFx > 1 {
            print("   ✅ RTFx > 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        } else {
            print("   ❌ RTFx < 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
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
                "rtfx": result.rtfx,
                "processingTime": result.processingTime,
                "totalFrames": result.totalFrames,
                "detectedSpeakers": result.detectedSpeakers,
                "groundTruthSpeakers": result.groundTruthSpeakers,
                "modelLoadTime": result.modelLoadTime,
                "audioLoadTime": result.audioLoadTime,
            ] as [String: Any]
        }

        do {
            let data = try JSONSerialization.data(withJSONObject: jsonData, options: .prettyPrinted)
            try data.write(to: URL(fileURLWithPath: path))
            print("💾 JSON results saved to: \(path)")
        } catch {
            print("❌ Failed to save JSON: \(error)")
        }
    }

    // MARK: - RTTM Ground Truth Loading (matches Python's approach)

    /// Load ground truth from RTTM file like Python does
    /// Format: SPEAKER <meeting_id> 1 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    private static func loadRTTMGroundTruth(for meetingName: String) -> [TimedSpeakerSegment] {
        // Look for RTTM file in Streaming-Sortformer-Conversion directory
        let rttmPath = "Streaming-Sortformer-Conversion/\(meetingName).rttm"

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
