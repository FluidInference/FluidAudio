#if os(macOS)
import FluidAudio
import Foundation
import OSLog

/// Batch EOU ASR benchmark command for LibriSpeech evaluation
public enum BatchEouBenchmarkCommand {

    private static let logger = AppLogger(category: "BatchEouBenchmark")

    public static func run(arguments: [String]) async {
        // Check for help flag
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var subset = "test-clean"
        var maxFiles: Int?
        var outputFile = "batch_eou_benchmark_results.json"
        var localModelsPath: String?
        var useMLPackage = false

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--subset":
                if i + 1 < arguments.count {
                    subset = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--local-models":
                if i + 1 < arguments.count {
                    localModelsPath = arguments[i + 1]
                    i += 1
                }
            case "--use-mlpackage":
                useMLPackage = true
            default:
                break
            }
            i += 1
        }
        
        guard let modelsPath = localModelsPath else {
            print("Error: --local-models path is required (e.g. path to parakeet_eou_split_1.28s)")
            printUsage()
            return
        }

        print("Starting Batch EOU ASR benchmark on LibriSpeech \(subset)")
        print("   Models: \(modelsPath)")
        print("   Max files: \(maxFiles?.description ?? "all")")
        print("   Output file: \(outputFile)")

        do {
            // Download LibriSpeech if needed
            print("Checking LibriSpeech \(subset) dataset...")
            try await downloadLibriSpeech(subset: subset)

            // Collect audio files
            let datasetPath = getLibriSpeechDirectory().appendingPathComponent(subset)
            let audioFiles = try collectLibriSpeechFiles(from: datasetPath)
            let filesToProcess = maxFiles != nil ? Array(audioFiles.prefix(maxFiles!)) : audioFiles

            print("Found \(audioFiles.count) files, processing \(filesToProcess.count)")

            // Initialize Batch EOU model
            print("Initializing Batch EOU model...")
            let manager = BatchEouAsrManager()
            let modelsURL = URL(fileURLWithPath: modelsPath)
            try await manager.initializeFromLocalPath(modelsURL, useMLPackage: useMLPackage)
            print("Models loaded successfully")

            // Process files
            var results: [EouBenchmarkResult] = []
            let startTime = Date()

            for (index, file) in filesToProcess.enumerated() {
                let progress = String(format: "%.1f", Double(index + 1) / Double(filesToProcess.count) * 100)
                print("[\(progress)%] Processing: \(file.fileName)")

                do {
                    let result = try await processFile(manager: manager, file: file)
                    results.append(result)
                    
                    let werPct = String(format: "%.1f", result.wer * 100)
                    let rtfx = String(format: "%.1f", result.rtfx)
                    print("   WER: \(werPct)% | RTFx: \(rtfx)x")
                    
                } catch {
                    print("   ERROR: \(error.localizedDescription)")
                }
            }

            let totalTime = Date().timeIntervalSince(startTime)

            // Calculate summary statistics
            let summary = calculateSummary(results: results)

            // Write results to JSON
            try writeResults(results: results, summary: summary, subset: subset, outputFile: outputFile)

            // Print summary
            printSummary(summary: summary, totalTime: totalTime, subset: subset, outputFile: outputFile)

        } catch {
            print("ERROR: Benchmark failed: \(error)")
        }
    }

    // MARK: - LibriSpeech Dataset (Copied from EouBenchmarkCommand)

    private static func getLibriSpeechDirectory() -> URL {
        let applicationSupportURL = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first!
        let appDirectory = applicationSupportURL.appendingPathComponent("FluidAudio", isDirectory: true)
        return appDirectory.appendingPathComponent("Datasets/LibriSpeech", isDirectory: true)
    }

    private static func downloadLibriSpeech(subset: String) async throws {
        let datasetsDirectory = getLibriSpeechDirectory()
        let subsetDirectory = datasetsDirectory.appendingPathComponent(subset)

        // Check if already downloaded
        if FileManager.default.fileExists(atPath: subsetDirectory.path) {
            let enumerator = FileManager.default.enumerator(
                at: subsetDirectory, includingPropertiesForKeys: nil)
            var transcriptCount = 0

            while let url = enumerator?.nextObject() as? URL {
                if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                    transcriptCount += 1
                    if transcriptCount >= 5 {
                        print("LibriSpeech \(subset) already downloaded")
                        return
                    }
                }
            }
        }

        print("Downloading LibriSpeech \(subset)...")

        let downloadURL: String
        switch subset {
        case "test-clean":
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-clean.tar.gz")
                .absoluteString
        case "test-other":
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-other.tar.gz")
                .absoluteString
        default:
            throw ASRError.processingFailed("Unsupported LibriSpeech subset: \(subset)")
        }

        try await downloadAndExtractTarGz(
            url: downloadURL,
            extractTo: datasetsDirectory,
            expectedSubpath: "LibriSpeech/\(subset)"
        )

        print("LibriSpeech \(subset) downloaded successfully")
    }

    private static func downloadAndExtractTarGz(
        url: String, extractTo: URL, expectedSubpath: String
    ) async throws {
        let downloadURL = URL(string: url)!

        print("Downloading from \(url.prefix(60))...")
        let (tempFile, _) = try await DownloadUtils.sharedSession.download(from: downloadURL)

        try FileManager.default.createDirectory(at: extractTo, withIntermediateDirectories: true)

        print("Extracting archive...")

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", tempFile.path, "-C", extractTo.path]

        let errorPipe = Pipe()
        process.standardError = errorPipe

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
            let errorMessage = String(data: errorData, encoding: .utf8) ?? "Unknown error"
            throw ASRError.processingFailed("Failed to extract tar.gz file: \(errorMessage)")
        }

        let extractedPath = extractTo.appendingPathComponent(expectedSubpath)
        if FileManager.default.fileExists(atPath: extractedPath.path) {
            let targetPath = extractTo.appendingPathComponent(
                expectedSubpath.components(separatedBy: "/").last!)
            try? FileManager.default.removeItem(at: targetPath)
            try FileManager.default.moveItem(at: extractedPath, to: targetPath)
            try? FileManager.default.removeItem(at: extractTo.appendingPathComponent("LibriSpeech"))
        }

        print("Dataset extracted successfully")
    }

    private static func collectLibriSpeechFiles(from directory: URL) throws -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []

        let fileManager = FileManager.default
        let enumerator = fileManager.enumerator(at: directory, includingPropertiesForKeys: nil)

        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                let transcriptContent = try String(contentsOf: url)
                let lines = transcriptContent.components(separatedBy: .newlines).filter { !$0.isEmpty }

                for line in lines {
                    let parts = line.components(separatedBy: " ")
                    guard parts.count >= 2 else { continue }

                    let audioId = parts[0]
                    let transcript = parts.dropFirst().joined(separator: " ")

                    let audioFileName = "\(audioId).flac"
                    let audioPath = url.deletingLastPathComponent().appendingPathComponent(audioFileName)

                    if fileManager.fileExists(atPath: audioPath.path) {
                        files.append(
                            LibriSpeechFile(
                                fileName: audioFileName,
                                audioPath: audioPath,
                                transcript: transcript
                            ))
                    }
                }
            }
        }

        return files.sorted { $0.fileName < $1.fileName }
    }

    // MARK: - Processing

    private static func processFile(
        manager: BatchEouAsrManager,
        file: LibriSpeechFile
    ) async throws -> EouBenchmarkResult {
        let audioURL = file.audioPath

        // Load and convert audio
        let audioSamples = try AudioConverter().resampleAudioFile(path: audioURL.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Transcribe
        let inferenceStart = Date()
        let transcription = try await manager.transcribe(audioSamples)
        let processingTime = Date().timeIntervalSince(inferenceStart)

        // Calculate WER
        let metrics = WERCalculator.calculateWERAndCER(
            hypothesis: transcription.text,
            reference: file.transcript
        )

        return EouBenchmarkResult(
            fileName: file.fileName,
            hypothesis: transcription.text,
            reference: file.transcript,
            wer: metrics.wer,
            cer: metrics.cer,
            insertions: metrics.insertions,
            deletions: metrics.deletions,
            substitutions: metrics.substitutions,
            totalWords: metrics.totalWords,
            audioLength: audioLength,
            processingTime: processingTime,
            detectedEou: transcription.eouDetected
        )
    }

    // MARK: - Results (Reused from EouBenchmarkCommand)

    private struct EouBenchmarkResult: Codable {
        let fileName: String
        let hypothesis: String
        let reference: String
        let wer: Double
        let cer: Double
        let insertions: Int
        let deletions: Int
        let substitutions: Int
        let totalWords: Int
        let audioLength: Double
        let processingTime: Double
        let detectedEou: Bool

        var rtfx: Double {
            processingTime > 0 ? audioLength / processingTime : 0
        }
    }

    private struct BenchmarkSummary: Codable {
        let filesProcessed: Int
        let averageWER: Double
        let medianWER: Double
        let averageCER: Double
        let medianRTFx: Double
        let overallRTFx: Double
        let totalAudioDuration: Double
        let totalProcessingTime: Double
        let eouDetectionRate: Double
    }

    private static func calculateSummary(results: [EouBenchmarkResult]) -> BenchmarkSummary {
        guard !results.isEmpty else {
            return BenchmarkSummary(
                filesProcessed: 0, averageWER: 0, medianWER: 0, averageCER: 0,
                medianRTFx: 0, overallRTFx: 0, totalAudioDuration: 0,
                totalProcessingTime: 0, eouDetectionRate: 0
            )
        }

        let werValues = results.map { $0.wer }.sorted()
        let rtfxValues = results.map { $0.rtfx }.sorted()

        let avgWER = results.reduce(0.0) { $0 + $1.wer } / Double(results.count)
        let medianWER = werValues[werValues.count / 2]

        let avgCER = results.reduce(0.0) { $0 + $1.cer } / Double(results.count)
        let medianRTFx = rtfxValues[rtfxValues.count / 2]

        let totalAudio = results.reduce(0.0) { $0 + $1.audioLength }
        let totalProcessing = results.reduce(0.0) { $0 + $1.processingTime }
        let overallRTFx = totalProcessing > 0 ? totalAudio / totalProcessing : 0

        let eouCount = results.filter { $0.detectedEou }.count
        let eouRate = Double(eouCount) / Double(results.count)

        return BenchmarkSummary(
            filesProcessed: results.count,
            averageWER: avgWER,
            medianWER: medianWER,
            averageCER: avgCER,
            medianRTFx: medianRTFx,
            overallRTFx: overallRTFx,
            totalAudioDuration: totalAudio,
            totalProcessingTime: totalProcessing,
            eouDetectionRate: eouRate
        )
    }

    private static func writeResults(
        results: [EouBenchmarkResult],
        summary: BenchmarkSummary,
        subset: String,
        outputFile: String
    ) throws {
        let summaryDict: [String: Any] = [
            "filesProcessed": summary.filesProcessed,
            "averageWER": summary.averageWER,
            "medianWER": summary.medianWER,
            "averageCER": summary.averageCER,
            "medianRTFx": summary.medianRTFx,
            "overallRTFx": summary.overallRTFx,
            "totalAudioDuration": summary.totalAudioDuration,
            "totalProcessingTime": summary.totalProcessingTime,
            "eouDetectionRate": summary.eouDetectionRate,
        ]

        let output: [String: Any] = [
            "config": [
                "model": "parakeet-realtime-eou-120m-batch",
                "dataset": "librispeech",
                "subset": subset,
            ],
            "summary": summaryDict,
            "results": results.map { result in
                [
                    "fileName": result.fileName,
                    "hypothesis": result.hypothesis,
                    "reference": result.reference,
                    "wer": result.wer,
                    "cer": result.cer,
                    "rtfx": result.rtfx,
                    "audioLength": result.audioLength,
                    "processingTime": result.processingTime,
                    "detectedEou": result.detectedEou,
                ]
            },
        ]

        let jsonData = try JSONSerialization.data(
            withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: outputFile))

        print("Results written to: \(outputFile)")
    }

    private static func printSummary(
        summary: BenchmarkSummary,
        totalTime: TimeInterval,
        subset: String,
        outputFile: String
    ) {
        print("")
        print("=" + String(repeating: "=", count: 59))
        print("BATCH EOU BENCHMARK RESULTS - LibriSpeech \(subset)")
        print("=" + String(repeating: "=", count: 59))
        print("")
        print("Files processed:    \(summary.filesProcessed)")
        print("")
        print("--- Accuracy ---")
        print("Average WER:        \(String(format: "%.2f", summary.averageWER * 100))%")
        print("Median WER:         \(String(format: "%.2f", summary.medianWER * 100))%")
        print("Average CER:        \(String(format: "%.2f", summary.averageCER * 100))%")
        print("")
        print("--- Performance ---")
        print("Median RTFx:        \(String(format: "%.1f", summary.medianRTFx))x realtime")
        print("Overall RTFx:       \(String(format: "%.1f", summary.overallRTFx))x realtime")
        print("Total audio:        \(String(format: "%.1f", summary.totalAudioDuration))s")
        print("Total processing:   \(String(format: "%.1f", summary.totalProcessingTime))s")
        print("")
        print("--- EOU Detection ---")
        print("EOU detection rate: \(String(format: "%.1f", summary.eouDetectionRate * 100))%")
        print("")
        print("Benchmark time:     \(String(format: "%.1f", totalTime))s")
        print("Output:             \(outputFile)")
        print("=" + String(repeating: "=", count: 59))
    }

    private static func printUsage() {
        print(
            """
            Batch EOU ASR Benchmark Command

            Usage:
                fluidaudio batch-eou-benchmark [options]

            Options:
                --subset <name>       LibriSpeech subset (default: test-clean)
                --max-files <n>       Maximum files to process (default: all)
                --output <file>       Output JSON file (default: batch_eou_benchmark_results.json)
                --local-models <path> Path to batch models (REQUIRED)
                --use-mlpackage       Use .mlpackage extension
                --help, -h            Show this help

            Example:
                fluidaudio batch-eou-benchmark \\
                  --local-models ./parakeet_eou_split_1.28s \\
                  --use-mlpackage \\
                  --max-files 100
            """
        )
    }
}
#endif
