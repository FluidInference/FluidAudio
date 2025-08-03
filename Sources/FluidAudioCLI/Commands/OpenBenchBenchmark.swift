import AVFoundation
import FluidAudio
import Foundation

/// Handler for OpenBench dataset benchmarking
enum OpenBenchBenchmark {
    static func run(arguments: [String]) async {
        let benchmarkStartTime = Date()

        // Configuration
        var datasets: [String] = []
        var maxFiles: Int? = nil
        var outputFile = "openbench_results.json"
        var threshold: Float = 0.7
        var debugMode = false
        var openBenchPath = "/Users/kikow/brandon/OpenBench"

        // Parse arguments
        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--datasets":
                // Parse comma-separated list
                if i + 1 < arguments.count {
                    datasets = arguments[i + 1].split(separator: ",").map(String.init)
                    i += 1
                }
            case "--all":
                // Use all 8 OpenBench datasets
                datasets = [
                    "earnings21", "ami-ihm", "ami-sdm", "aishell-4",
                    "voxconverse", "ava-avd", "ali-meetings", "icsi-meetings",
                ]
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--threshold":
                if i + 1 < arguments.count {
                    threshold = Float(arguments[i + 1]) ?? 0.7
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--path":
                if i + 1 < arguments.count {
                    openBenchPath = arguments[i + 1]
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--help":
                printUsage()
                return
            default:
                print("⚠️ Unknown option: \(arguments[i])")
            }
            i += 1
        }

        // Default to all datasets if none specified
        if datasets.isEmpty {
            datasets = [
                "earnings21", "ami-ihm", "ami-sdm", "aishell-4",
                "voxconverse", "ava-avd", "ali-meetings", "icsi-meetings",
            ]
        }

        print("🚀 OpenBench Diarization Benchmark")
        print("📊 Datasets: \(datasets.joined(separator: ", "))")
        print("📁 OpenBench path: \(openBenchPath)")
        print("🎯 Clustering threshold: \(threshold)")
        if let maxFiles = maxFiles {
            print("🔢 Max files per dataset: \(maxFiles)")
        }

        // Initialize diarizer
        let config = DiarizerConfig(
            clusteringThreshold: threshold,
            minDurationOn: 1.0,
            minDurationOff: 0.5,
            minActivityThreshold: 10.0,
            debugMode: debugMode
        )

        let manager = DiarizerManager(config: config)

        do {
            let models = try await DiarizerModels.downloadIfNeeded()
            manager.initialize(models: models)
            print("✅ Models initialized successfully\n")
        } catch {
            print("❌ Failed to initialize models: \(error)")
            exit(1)
        }

        // Process each dataset
        var allResults: [[String: Any]] = []
        var totalFiles = 0
        var totalDuration: Double = 0
        var totalProcessingTime: Double = 0
        var totalDER: Double = 0
        var filesWithDER = 0

        for dataset in datasets {
            if let result = await processDataset(
                dataset: dataset,
                manager: manager,
                openBenchPath: openBenchPath,
                maxFiles: maxFiles
            ) {
                allResults.append(result)

                // Update totals
                if let numFiles = result["num_files"] as? Int {
                    totalFiles += numFiles
                }
                if let duration = result["total_duration"] as? Double {
                    totalDuration += duration
                }
                if let processingTime = result["total_processing_time"] as? Double {
                    totalProcessingTime += processingTime
                }
                if let avgDER = result["avg_der"] as? Double,
                    let numFilesWithDER = result["num_files_with_der"] as? Int
                {
                    totalDER += avgDER * Double(numFilesWithDER)
                    filesWithDER += numFilesWithDER
                }
            }
        }

        // Overall summary
        print("\n" + String(repeating: "=", count: 60))
        print("🏁 OVERALL BENCHMARK RESULTS")
        print(String(repeating: "=", count: 60))

        print("\nDataset         Files    Avg RTF      Avg DER      Duration")
        print(String(repeating: "-", count: 65))

        for result in allResults {
            if let dataset = result["dataset"] as? String,
                let numFiles = result["num_files"] as? Int,
                let avgRTF = result["avg_rtf"] as? Double,
                let duration = result["total_duration"] as? Double
            {

                let derStr: String
                if let avgDER = result["avg_der"] as? Double {
                    derStr = String(format: "%.1f%%", avgDER * 100)
                } else {
                    derStr = "N/A"
                }

                // Use safer string concatenation instead of format specifiers
                let datasetPadded = dataset.padding(toLength: 15, withPad: " ", startingAt: 0)
                let numFilesPadded = String(numFiles).padding(toLength: 8, withPad: " ", startingAt: 0)
                let rtfStr = String(format: "%.3f", avgRTF).padding(toLength: 12, withPad: " ", startingAt: 0)
                let derStrPadded = derStr.padding(toLength: 12, withPad: " ", startingAt: 0)
                let durationStr = String(format: "%.1fmin", duration / 60)

                print("\(datasetPadded) \(numFilesPadded) \(rtfStr) \(derStrPadded) \(durationStr)")
            }
        }

        let overallRTF = totalProcessingTime / totalDuration
        let overallDER = filesWithDER > 0 ? totalDER / Double(filesWithDER) : nil

        print("\n📈 Performance Summary:")
        print("  Total files processed: \(totalFiles)")
        print("  Total audio duration: \(String(format: "%.1f", totalDuration / 3600)) hours")
        print("  Overall Average RTF: \(String(format: "%.3f", overallRTF))")
        print("  Performance: \(overallRTF < 1.0 ? "✅ Real-time capable!" : "❌ Slower than real-time")")

        if let overallDER = overallDER {
            print("  Overall Average DER: \(String(format: "%.1f%%", overallDER * 100))")
            print("  Accuracy: \(overallDER < 0.3 ? "✅ Good diarization!" : "⚠️  DER needs improvement")")
        }

        // Save results
        let benchmarkTime = Date().timeIntervalSince(benchmarkStartTime)
        let results: [String: Any] = [
            "model": "FluidAudio",
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "total_files": totalFiles,
            "overall_avg_rtf": overallRTF,
            "overall_avg_der": overallDER as Any,
            "total_duration_hours": totalDuration / 3600,
            "benchmark_time_seconds": benchmarkTime,
            "datasets": allResults,
        ]

        if let jsonData = try? JSONSerialization.data(withJSONObject: results, options: .prettyPrinted) {
            try? jsonData.write(to: URL(fileURLWithPath: outputFile))
            print("\n📄 Results saved to: \(outputFile)")
        }

        print("⏱️  Total benchmark time: \(String(format: "%.1f", benchmarkTime / 60)) minutes")
    }

    private static func processDataset(
        dataset: String,
        manager: DiarizerManager,
        openBenchPath: String,
        maxFiles: Int?
    ) async -> [String: Any]? {
        print("\n" + String(repeating: "=", count: 60))
        print("Benchmarking \(dataset)")
        print(String(repeating: "=", count: 60))

        // Use pre-extracted data in FluidAudioSwift directory
        let datasetPath = "/Users/kikow/brandon/FluidAudioSwift/OpenBenchData/\(dataset)"

        // Load ground truth
        let groundTruthPath = "\(datasetPath)/ground_truth.json"
        guard let groundTruthData = try? Data(contentsOf: URL(fileURLWithPath: groundTruthPath)),
            let groundTruthJSON = try? JSONSerialization.jsonObject(with: groundTruthData) as? [String: Any],
            let filesInfo = groundTruthJSON["files"] as? [[String: Any]]
        else {
            print("❌ No ground truth found at \(groundTruthPath)")
            return nil
        }

        // Get audio files with ground truth
        let files = filesInfo.prefix(maxFiles ?? Int.max).compactMap {
            fileInfo -> (id: String, path: String, groundTruth: [(start: Double, end: Double, speaker: String)]?)? in
            guard let id = fileInfo["id"] as? String,
                let audioPath = fileInfo["audio_path"] as? String
            else { return nil }

            let fullPath = "\(datasetPath)/\(audioPath)"

            // Parse ground truth segments
            var groundTruth: [(start: Double, end: Double, speaker: String)]? = nil
            if let segments = fileInfo["ground_truth"] as? [[String: Any]] {
                groundTruth = segments.compactMap { segment in
                    guard let start = segment["start"] as? Double,
                        let end = segment["end"] as? Double,
                        let speaker = segment["speaker"] as? String
                    else { return nil }
                    return (start: start, end: end, speaker: speaker)
                }
            }

            return (id: id, path: fullPath, groundTruth: groundTruth)
        }

        guard !files.isEmpty else {
            print("❌ No audio files found in \(datasetPath)")
            return nil
        }

        print("Found \(files.count) audio files to process")

        var results: [[String: Any]] = []
        var totalDuration: Double = 0
        var totalProcessingTime: Double = 0
        var derScores: [Double] = []

        // Process each audio file
        for (index, audioInfo) in files.enumerated() {
            let audioPath = audioInfo.path
            let groundTruth = audioInfo.groundTruth

            print("  [\(index + 1)/\(files.count)] Processing \(audioInfo.id)...", terminator: "")

            let startTime = Date()

            do {
                let audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: audioPath))
                let duration = Double(audioFile.length) / audioFile.fileFormat.sampleRate

                // Load audio samples
                let audioSamples = try await AudioProcessor.loadAudioFile(path: audioPath)

                // Run diarization
                let result = try manager.performCompleteDiarization(audioSamples, sampleRate: 16000)
                let processingTime = Date().timeIntervalSince(startTime)
                let rtf = processingTime / duration

                // Calculate DER if we have ground truth
                var der: Double? = nil
                if let groundTruth = groundTruth {
                    der = calculateDER(predictions: result.segments, groundTruth: groundTruth)
                    if let der = der {
                        derScores.append(der)
                    }
                }

                let numSpeakers = result.speakerDatabase.count

                print(
                    " ✅ RTF: \(String(format: "%.3f", rtf))"
                        + (der != nil ? " | DER: \(String(format: "%.1f%%", der! * 100))" : "")
                        + " | Speakers: \(numSpeakers)")

                results.append([
                    "file": audioInfo.id,
                    "duration": duration,
                    "rtf": rtf,
                    "der": der as Any,
                    "num_speakers": numSpeakers,
                    "num_segments": result.segments.count,
                    "processing_time": processingTime,
                ])

                totalDuration += duration
                totalProcessingTime += processingTime

            } catch {
                print(" ❌ Failed: \(error)")
            }
        }

        // No cleanup needed - we're using persistent extracted files

        // Calculate summary
        if !results.isEmpty {
            let avgRTF = results.compactMap { $0["rtf"] as? Double }.reduce(0, +) / Double(results.count)
            let avgDER = derScores.isEmpty ? nil : derScores.reduce(0, +) / Double(derScores.count)

            print("\n📊 \(dataset) Results:")
            print("  Files processed: \(results.count)")
            print("  Average RTF: \(String(format: "%.3f", avgRTF)) \(avgRTF < 1.0 ? "✅" : "❌")")
            if let avgDER = avgDER {
                print("  Average DER: \(String(format: "%.1f%%", avgDER * 100))")
            }
            print("  Total audio: \(String(format: "%.1f", totalDuration / 60)) minutes")
            print("  Total time: \(String(format: "%.1f", totalProcessingTime / 60)) minutes")

            return [
                "dataset": dataset,
                "num_files": results.count,
                "avg_rtf": avgRTF,
                "avg_der": avgDER as Any,
                "num_files_with_der": derScores.count,
                "total_duration": totalDuration,
                "total_processing_time": totalProcessingTime,
                "results": results,
            ]
        }

        return nil
    }

    private static func extractAudioFromParquet(
        datasetPath: String,
        tempDir: String,
        maxFiles: Int?
    ) async -> (
        audioFiles: [(id: String, path: String, groundTruth: [(start: Double, end: Double, speaker: String)]?)]?,
        error: Error?
    ) {
        // Call Python helper to extract audio from parquet
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/Users/kikow/brandon/OpenBench/venv_new/bin/python")
        process.arguments = [
            "/Users/kikow/brandon/OpenBench/extract_openbench_audio.py",
            datasetPath,
            tempDir,
        ]

        if let maxFiles = maxFiles {
            process.arguments?.append(String(maxFiles))
        }

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.standardError

        do {
            try process.run()
            process.waitUntilExit()

            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            guard let output = String(data: data, encoding: .utf8),
                let jsonData = output.data(using: .utf8),
                let json = try JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                let audioFilesArray = json["audio_files"] as? [[String: Any]]
            else {
                return (nil, nil)
            }

            var audioFiles:
                [(id: String, path: String, groundTruth: [(start: Double, end: Double, speaker: String)]?)] = []

            for fileInfo in audioFilesArray {
                guard let id = fileInfo["id"] as? String,
                    let path = fileInfo["path"] as? String
                else { continue }

                var groundTruth: [(start: Double, end: Double, speaker: String)]? = nil

                if let gtArray = fileInfo["ground_truth"] as? [[String: Any]] {
                    groundTruth = gtArray.compactMap { segment in
                        guard let start = segment["start"] as? Double,
                            let end = segment["end"] as? Double,
                            let speaker = segment["speaker"] as? String
                        else { return nil }
                        return (start: start, end: end, speaker: speaker)
                    }
                }

                audioFiles.append((id: id, path: path, groundTruth: groundTruth))
            }

            return (audioFiles, nil)
        } catch {
            return (nil, error)
        }
    }

    private static func calculateDER(
        predictions: [TimedSpeakerSegment],
        groundTruth: [(start: Double, end: Double, speaker: String)]
    ) -> Double {
        // Convert ground truth to TimedSpeakerSegment format
        let groundTruthSegments = groundTruth.map { segment in
            TimedSpeakerSegment(
                speakerId: segment.speaker,
                embedding: [],  // Not needed for DER calculation
                startTimeSeconds: Float(segment.start),
                endTimeSeconds: Float(segment.end),
                qualityScore: 1.0
            )
        }

        // Calculate total duration
        let totalDuration = Float(groundTruth.map { $0.end }.max() ?? 0)

        // Use existing MetricsCalculator
        let metrics = MetricsCalculator.calculateDiarizationMetrics(
            predicted: predictions,
            groundTruth: groundTruthSegments,
            totalDuration: totalDuration
        )

        return Double(metrics.der / 100.0)  // Convert percentage to fraction
    }

    private static func printUsage() {
        print(
            """

            OpenBench Benchmark Command Usage:
                fluidaudio openbench-benchmark [options]

            Options:
                --datasets <list>     Comma-separated list of datasets
                --all                 Run all 8 OpenBench datasets
                --max-files <int>     Max files per dataset (for testing)
                --threshold <float>   Clustering threshold (default: 0.7)
                --output <file>       Output JSON file (default: openbench_results.json)
                --path <path>         Path to OpenBench directory
                --debug               Enable debug mode

            Available datasets:
                earnings21, ami-ihm, ami-sdm, aishell-4,
                voxconverse, ava-avd, ali-meetings, icsi-meetings

            Examples:
                # Run all datasets
                fluidaudio openbench-benchmark --all
                
                # Run specific datasets with 10 files each
                fluidaudio openbench-benchmark --datasets ami-ihm,ami-sdm --max-files 10
                
                # Run with custom threshold
                fluidaudio openbench-benchmark --all --threshold 0.8
            """)
    }
}
