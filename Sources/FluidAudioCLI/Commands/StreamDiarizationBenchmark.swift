import AVFoundation
import FluidAudio
import Foundation

/// Independent streaming diarization benchmark for evaluating real-time performance
/// This benchmark does NOT use Hungarian algorithm for speaker remapping
@available(macOS 13.0, *)
enum StreamDiarizationBenchmark {

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

    static func printUsage() {
        print(
            """
            Stream Diarization Benchmark Command

            Evaluates streaming speaker diarization WITHOUT retroactive speaker remapping.
            This measures true real-time performance as seen in production systems.

            Usage: fluidaudio stream-diarization-benchmark [options]

            Options:
                --dataset <name>         Dataset to benchmark (default: ami-sdm)
                --single-file <name>     Process a specific meeting (e.g., ES2004a)
                --max-files <n>          Maximum number of files to process
                --chunk-seconds <sec>    Chunk duration for streaming (default: 10.0)
                --overlap-seconds <sec>  Overlap between chunks (default: 0.0)
                --threshold <value>      Clustering threshold (default: 0.7)
                --assignment-threshold   Threshold for assigning to existing speakers (default: 0.84)
                --update-threshold       Threshold for updating speaker embeddings (default: 0.56)
                --output <file>         Output JSON file for results
                --csv <file>            Output CSV file for summary
                --verbose               Enable verbose output
                --debug                 Enable debug output
                --auto-download         Auto-download dataset if missing
                --iterations <n>        Number of iterations per file (default: 1)
                --help                  Show this help message

            Streaming Modes (via chunk/overlap settings):
                Real-time:  --chunk-seconds 3 --overlap-seconds 2   (~15-30x RTFx)
                Balanced:   --chunk-seconds 10 --overlap-seconds 5  (~70x RTFx)
                Batch:      --chunk-seconds 10 --overlap-seconds 0  (~140x RTFx)

            Performance Targets:
                DER < 30%  (competitive with research systems)
                RTFx > 1x  (real-time capable)

            Examples:
                # Benchmark single file with real-time settings
                fluidaudio stream-diarization-benchmark --single-file ES2004a \\
                    --chunk-seconds 3 --overlap-seconds 2

                # Full AMI benchmark with balanced settings
                fluidaudio stream-diarization-benchmark --dataset ami-sdm \\
                    --chunk-seconds 10 --overlap-seconds 5 --csv results.csv

                # Quick test on 5 files
                fluidaudio stream-diarization-benchmark --max-files 5 --verbose
            """)
    }

    static func run(arguments: [String]) async {
        // Parse arguments
        var dataset = "ami-sdm"
        var singleFile: String?
        var maxFiles: Int?
        var chunkSeconds: Double = 10.0
        var overlapSeconds: Double = 0.0
        var threshold: Float = 0.7
        var assignmentThreshold: Float = 0.84
        var updateThreshold: Float = 0.56
        var outputFile: String?
        var csvFile: String?
        var verbose = false
        var debugMode = false
        var autoDownload = false
        var iterations = 1

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
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
            case "--chunk-seconds":
                if i + 1 < arguments.count {
                    chunkSeconds = Double(arguments[i + 1]) ?? 10.0
                    i += 1
                }
            case "--overlap-seconds":
                if i + 1 < arguments.count {
                    overlapSeconds = Double(arguments[i + 1]) ?? 0.0
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.7
                    i += 1
                }
            case "--assignment-threshold":
                if i + 1 < arguments.count {
                    assignmentThreshold = Float(arguments[i + 1]) ?? 0.84
                    i += 1
                }
            case "--update-threshold":
                if i + 1 < arguments.count {
                    updateThreshold = Float(arguments[i + 1]) ?? 0.56
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--csv":
                if i + 1 < arguments.count {
                    csvFile = arguments[i + 1]
                    i += 1
                }
            case "--verbose":
                verbose = true
            case "--debug":
                debugMode = true
            case "--auto-download":
                autoDownload = true
            case "--iterations":
                if i + 1 < arguments.count {
                    iterations = Int(arguments[i + 1]) ?? 1
                    i += 1
                }
            case "--help":
                printUsage()
                return
            default:
                if !arguments[i].starts(with: "--") {
                    print("⚠️ Unknown argument: \(arguments[i])")
                }
            }
            i += 1
        }

        // Validate settings
        let hopSize = max(chunkSeconds - overlapSeconds, 1.0)
        let overlapRatio = overlapSeconds / chunkSeconds

        print("🚀 Starting Stream Diarization Benchmark")
        print("   Dataset: \(dataset)")
        print("   Chunk size: \(chunkSeconds)s")
        print("   Overlap: \(overlapSeconds)s (\(String(format: "%.0f", overlapRatio * 100))%)")
        print("   Hop size: \(hopSize)s")
        print("   Clustering threshold: \(threshold)")
        print("   Assignment threshold: \(assignmentThreshold)")
        print("   Update threshold: \(updateThreshold)")

        // Determine streaming mode
        let mode: String
        if overlapSeconds == 0 {
            mode = "Batch (no overlap)"
        } else if overlapRatio >= 0.6 {
            mode = "Real-time (high overlap)"
        } else {
            mode = "Balanced"
        }
        print("   Mode: \(mode)")
        print("")

        // Download dataset if needed
        if autoDownload {
            print("📥 Downloading AMI dataset if needed...")
            // Download both audio and annotations
            await DatasetDownloader.downloadAMIDataset(
                variant: dataset == "ami-ihm" ? .ihm : .sdm,
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
            filesToProcess = getAMIFiles(dataset: dataset, maxFiles: maxFiles)
        }

        if filesToProcess.isEmpty {
            print("❌ No files found to process")
            return
        }

        print("📂 Processing \(filesToProcess.count) file(s)")
        print("")

        // Initialize models once and track timing
        print("🔧 Initializing models...")
        let modelStartTime = Date()
        let models: DiarizerModels
        do {
            models = try await DiarizerModels.downloadIfNeeded()
        } catch {
            print("❌ Failed to initialize models: \(error)")
            return
        }
        let modelInitTime = Date().timeIntervalSince(modelStartTime)
        print("✅ Models ready (took \(String(format: "%.2f", modelInitTime))s)")
        print("")

        // Process each file
        var allResults: [BenchmarkResult] = []

        for (fileIndex, meetingName) in filesToProcess.enumerated() {
            print(String(repeating: "=", count: 60))
            print("[\(fileIndex + 1)/\(filesToProcess.count)] Processing: \(meetingName)")
            print(String(repeating: "=", count: 60))

            var iterationResults: [BenchmarkResult] = []

            for iteration in 1...iterations {
                if iterations > 1 {
                    print("\n  Iteration \(iteration)/\(iterations)")
                }

                if let result = await processMeeting(
                    meetingName: meetingName,
                    models: models,
                    modelInitTime: modelInitTime,
                    chunkSeconds: chunkSeconds,
                    overlapSeconds: overlapSeconds,
                    threshold: threshold,
                    assignmentThreshold: assignmentThreshold,
                    updateThreshold: updateThreshold,
                    verbose: verbose,
                    debugMode: debugMode
                ) {
                    iterationResults.append(result)

                    // Print summary for this iteration
                    print("\n📊 Results for \(meetingName) (iteration \(iteration)):")
                    print("  DER: \(String(format: "%.1f", result.der))%")
                    print("  RTFx: \(String(format: "%.1f", result.rtfx))x")
                    print("  Speakers: \(result.detectedSpeakers) detected / \(result.groundTruthSpeakers) truth")
                }
            }

            // Average results if multiple iterations
            if !iterationResults.isEmpty {
                let avgResult = averageResults(iterationResults)
                allResults.append(avgResult)

                if iterations > 1 {
                    print("\n📊 Average over \(iterations) iterations:")
                    print(
                        "  DER: \(String(format: "%.1f", avgResult.der))% ± \(String(format: "%.1f", standardDeviation(iterationResults.map { $0.der })))%"
                    )
                    print(
                        "  RTFx: \(String(format: "%.1f", avgResult.rtfx))x ± \(String(format: "%.1f", standardDeviation(iterationResults.map { $0.rtfx })))x"
                    )
                }
            }
        }

        // Print final summary
        printFinalSummary(results: allResults)

        // Save results
        if let outputPath = outputFile {
            saveJSONResults(results: allResults, to: outputPath)
        }

        if let csvPath = csvFile {
            saveCSVResults(results: allResults, to: csvPath)
        }
    }

    private static func processMeeting(
        meetingName: String,
        models: DiarizerModels,
        modelInitTime: Double,
        chunkSeconds: Double,
        overlapSeconds: Double,
        threshold: Float,
        assignmentThreshold: Float,
        updateThreshold: Float,
        verbose: Bool,
        debugMode: Bool
    ) async -> BenchmarkResult? {

        // Load audio
        let audioPath = getAudioPath(for: meetingName)
        guard FileManager.default.fileExists(atPath: audioPath) else {
            print("❌ Audio file not found: \(audioPath)")
            return nil
        }

        do {
            // Track audio loading time
            let audioLoadStart = Date()
            let audioData = try loadAudioFile(at: audioPath)
            let audioLoadTime = Date().timeIntervalSince(audioLoadStart)
            let totalDuration = Double(audioData.count) / 16000.0

            if verbose {
                print("  Audio duration: \(String(format: "%.1f", totalDuration))s")
                print("  Audio load time: \(String(format: "%.3f", audioLoadTime))s")
            }

            // Initialize diarizer with streaming manager
            let config = DiarizerConfig(
                clusteringThreshold: threshold,
                minSpeechDuration: 1.0,
                minSilenceGap: 0.5,
                minActiveFramesCount: 10.0,
                debugMode: debugMode
            )

            let diarizerManager = DiarizerManager(
                config: config,
                chunkDuration: Float(chunkSeconds),
                chunkOverlap: Float(overlapSeconds)
            )
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
                let chunkResult = try diarizerManager.performCompleteDiarization(paddedChunk)
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
                        startTimeSeconds: segment.startTimeSeconds + Float(chunkStartTime),
                        endTimeSeconds: segment.endTimeSeconds + Float(chunkStartTime),
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

                    print(
                        String(
                            format: "    [Chunk %3d] %.1f%% | RTFx: %.1fx | Speakers: %d | Latency: %.3fs",
                            chunkIndex, progress, rtfx,
                            diarizerManager.speakerManager.speakerCount,
                            chunkLatency))
                }

                position += hopSamples
                chunkIndex += 1
            }

            let totalElapsed = Date().timeIntervalSince(startTime)
            let finalRTFx = totalDuration / totalElapsed

            // Load ground truth
            let groundTruth = await AMIParser.loadAMIGroundTruth(
                for: meetingName,
                duration: Float(totalDuration)
            )

            guard !groundTruth.isEmpty else {
                print("⚠️ No ground truth found for \(meetingName)")
                return nil
            }

            // Calculate metrics WITHOUT Hungarian mapping for true streaming evaluation
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
                detectedSpeakers: diarizerManager.speakerManager.speakerCount,
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
            print("❌ Error processing \(meetingName): \(error)")
            return nil
        }
    }

    /// Calculate DER metrics WITHOUT Hungarian mapping - for true streaming evaluation
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

        print("🔄 STREAMING MAPPING (first-occurrence): \(firstOccurrenceMap)")

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
            case let (gt?, pred?):
                // Use streaming mapping if available, otherwise treat as error
                let mappedPred = firstOccurrenceMap[pred]
                if mappedPred != gt {
                    speakerErrorFrames += 1
                }
            }
        }

        // Calculate JER (Jaccard Error Rate) without Hungarian mapping
        var segmentIntersection = 0
        var segmentUnion = 0

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

            if !gtSpeakers.isEmpty || !predSpeakers.isEmpty {
                segmentUnion += 1
                if gtSpeakers == predSpeakers && !gtSpeakers.isEmpty {
                    segmentIntersection += 1
                }
            }
        }

        let jaccardIndex = segmentUnion > 0 ? Float(segmentIntersection) / Float(segmentUnion) : 0
        let jer = (1.0 - jaccardIndex) * 100.0

        // Calculate rates
        let missRate = (Float(missedFrames) / Float(totalFrames)) * 100.0
        let falseAlarmRate = (Float(falseAlarmFrames) / Float(totalFrames)) * 100.0
        let speakerErrorRate = (Float(speakerErrorFrames) / Float(totalFrames)) * 100.0
        let der = missRate + falseAlarmRate + speakerErrorRate

        print(
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
        let allMeetings = [
            "ES2002a", "ES2002b", "ES2002c", "ES2002d",
            "ES2003a", "ES2003b", "ES2003c", "ES2003d",
            "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "ES2005a", "ES2005b", "ES2005c", "ES2005d",
            "ES2006a", "ES2006b", "ES2006c", "ES2006d",
            "ES2007a", "ES2007b", "ES2007c", "ES2007d",
            "ES2008a", "ES2008b", "ES2008c", "ES2008d",
            "ES2009a", "ES2009b", "ES2009c", "ES2009d",
            "ES2010a", "ES2010b", "ES2010c", "ES2010d",
            "ES2011a", "ES2011b", "ES2011c", "ES2011d",
            "ES2012a", "ES2012b", "ES2012c", "ES2012d",
            "ES2013a", "ES2013b", "ES2013c", "ES2013d",
            "ES2014a", "ES2014b", "ES2014c", "ES2014d",
            "ES2015a", "ES2015b", "ES2015c", "ES2015d",
            "ES2016a", "ES2016b", "ES2016c", "ES2016d",
            "IS1000a", "IS1000b", "IS1000c", "IS1000d",
            "IS1001a", "IS1001b", "IS1001c", "IS1001d",
            "IS1002b", "IS1002c", "IS1002d",
            "IS1003a", "IS1003b", "IS1003c", "IS1003d",
            "IS1004a", "IS1004b", "IS1004c", "IS1004d",
            "IS1005a", "IS1005b", "IS1005c",
            "IS1006a", "IS1006b", "IS1006c", "IS1006d",
            "IS1007a", "IS1007b", "IS1007c", "IS1007d",
            "IS1008a", "IS1008b", "IS1008c", "IS1008d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d",
            "TS3005a", "TS3005b", "TS3005c", "TS3005d",
            "TS3008a", "TS3008b", "TS3008c", "TS3008d",
            "TS3009a", "TS3009b", "TS3009c", "TS3009d",
            "TS3010a", "TS3010b", "TS3010c", "TS3010d",
            "TS3011a", "TS3011b", "TS3011c", "TS3011d",
            "TS3012a", "TS3012b", "TS3012c", "TS3012d",
        ]

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

    private static func loadAudioFile(at path: String) throws -> [Float] {
        let url = URL(fileURLWithPath: path)
        let file = try AVAudioFile(forReading: url)

        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16000,
            channels: 1,
            interleaved: false
        )!

        let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: UInt32(file.length)
        )!

        // Convert to 16kHz if needed
        if file.processingFormat.sampleRate != 16000 {
            let converter = AVAudioConverter(from: file.processingFormat, to: format)!
            let inputBuffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat,
                frameCapacity: AVAudioFrameCount(file.length)
            )!

            try file.read(into: inputBuffer)

            let inputBlock: AVAudioConverterInputBlock = { _, outStatus in
                outStatus.pointee = .haveData
                return inputBuffer
            }

            var error: NSError?
            _ = converter.convert(to: buffer, error: &error, withInputFrom: inputBlock)
            if let error = error {
                throw error
            }
        } else {
            try file.read(into: buffer)
        }

        let floatArray = Array(
            UnsafeBufferPointer(
                start: buffer.floatChannelData?[0],
                count: Int(buffer.frameLength)
            )
        )

        return floatArray
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

        // Visualization temporarily disabled - BenchmarkVisualizer to be implemented
        // // Generate ASCII chart visualization
        // let vizResults = results.map { result in
        //     BenchmarkVisualizer.BenchmarkResult(
        //         dataset: result.meetingName,
        //         der: Double(result.der),
        //         rtfx: Double(result.rtfx),
        //         missedSpeech: Double(result.missRate),
        //         falseAlarm: Double(result.falseAlarmRate),
        //         speakerError: Double(result.speakerErrorRate),
        //         detectedSpeakers: result.detectedSpeakers,
        //         truthSpeakers: result.groundTruthSpeakers
        //     )
        // }

        // let chart = BenchmarkVisualizer.generateASCIIChart(results: vizResults)
        // print(chart)

        // // Save charts to files
        // BenchmarkVisualizer.saveChartToFile(chart, filename: "benchmark_chart.txt")

        // // Generate and save HTML chart
        // let htmlChart = BenchmarkVisualizer.generateHTMLChart(results: vizResults)
        // let htmlPath = "benchmark_chart.html"
        // try? htmlChart.write(toFile: htmlPath, atomically: true, encoding: .utf8)
        // print("📊 Interactive HTML chart saved to: \(htmlPath)")

        // Continue with original summary
        print("\n" + String(repeating: "=", count: 60))
        print("📊 FINAL BENCHMARK SUMMARY")
        print(String(repeating: "=", count: 60))

        // Calculate aggregates
        let avgDER = results.map { $0.der }.reduce(0, +) / Float(results.count)
        let avgRTFx = results.map { $0.rtfx }.reduce(0, +) / Float(results.count)
        let avgFragmentation = results.map { $0.speakerFragmentation }.reduce(0, +) / Float(results.count)
        let avgLatency90 = results.map { $0.latency90th }.reduce(0, +) / Double(results.count)
        let avgLatency99 = results.map { $0.latency99th }.reduce(0, +) / Double(results.count)

        print("\n🎯 Performance Metrics:")
        print("  Average DER: \(String(format: "%.1f", avgDER))%")
        print(
            "    Missed speech: \(String(format: "%.1f", results.map { $0.missRate }.reduce(0, +) / Float(results.count)))%"
        )
        print(
            "    False alarm: \(String(format: "%.1f", results.map { $0.falseAlarmRate }.reduce(0, +) / Float(results.count)))%"
        )
        print(
            "    Speaker error: \(String(format: "%.1f", results.map { $0.speakerErrorRate }.reduce(0, +) / Float(results.count)))%"
        )

        print("\n⚡ Speed Metrics:")
        print("  Average RTFx: \(String(format: "%.1f", avgRTFx))x")
        print("  90th percentile latency: \(String(format: "%.3f", avgLatency90))s")
        print("  99th percentile latency: \(String(format: "%.3f", avgLatency99))s")

        print("\n🔍 Quality Metrics:")
        print("  Speaker fragmentation: \(String(format: "%.2f", avgFragmentation)) (1.0 = ideal)")

        // Show best and worst performers
        let sortedByDER = results.sorted { $0.der < $1.der }
        print("\n📈 Best DER: \(sortedByDER[0].meetingName) = \(String(format: "%.1f", sortedByDER[0].der))%")
        print("📉 Worst DER: \(sortedByDER.last!.meetingName) = \(String(format: "%.1f", sortedByDER.last!.der))%")

        // Check against targets
        print("\n✅ Target Check:")
        if avgDER < 30 {
            print("  ✅ DER < 30% (achieved: \(String(format: "%.1f", avgDER))%)")
        } else {
            print("  ❌ DER < 30% (achieved: \(String(format: "%.1f", avgDER))%)")
        }

        if avgRTFx > 1 {
            print("  ✅ RTFx > 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
        } else {
            print("  ❌ RTFx > 1x (achieved: \(String(format: "%.1f", avgRTFx))x)")
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
            print("\n💾 JSON results saved to: \(path)")
        } catch {
            print("❌ Failed to save JSON: \(error)")
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
            print("💾 CSV results saved to: \(path)")
        } catch {
            print("❌ Failed to save CSV: \(error)")
        }
    }
}
