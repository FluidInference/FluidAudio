#if os(macOS)
import FluidAudio
import Foundation
import OSLog

/// Streaming EOU Benchmark Command
/// Benchmarks the cache-aware streaming implementation (StreamingEouAsrManager)
public enum StreamingEouBenchmarkCommand {

    private static let logger = AppLogger(category: "StreamingEouBenchmark")

    public static func run(arguments: [String]) async {
        // Check for help flag
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var subset = "test-clean"
        var maxFiles: Int?
        var outputFile = "streaming_eou_benchmark_results.json"
        var debugMode = false
        var localModelsPath: String?
        var chunkDurationMs: Double = 500  // Default 500ms chunks for streaming

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
            case "--debug":
                debugMode = true
            case "--local-models":
                if i + 1 < arguments.count {
                    localModelsPath = arguments[i + 1]
                    i += 1
                }
            case "--chunk-duration":
                if i + 1 < arguments.count {
                    chunkDurationMs = Double(arguments[i + 1]) ?? 500
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        print("Starting Streaming EOU ASR benchmark on LibriSpeech \(subset)")
        print("   Mode: Cache-Aware Streaming")
        print("   Chunk duration: \(Int(chunkDurationMs))ms")
        print("   Max files: \(maxFiles?.description ?? "all")")
        print("   Output file: \(outputFile)")
        print("   Debug mode: \(debugMode ? "enabled" : "disabled")")

        do {
            // Download LibriSpeech if needed
            print("Checking LibriSpeech \(subset) dataset...")
            try await downloadLibriSpeech(subset: subset)

            // Collect audio files
            let datasetPath = getLibriSpeechDirectory().appendingPathComponent(subset)
            let audioFiles = try collectLibriSpeechFiles(from: datasetPath)
            let filesToProcess = maxFiles != nil ? Array(audioFiles.prefix(maxFiles!)) : audioFiles

            print("Found \(audioFiles.count) files, processing \(filesToProcess.count)")

            // Initialize Streaming EOU model
            print("Initializing Streaming Parakeet EOU model...")
            let manager = StreamingEouAsrManager()

            if let localPath = localModelsPath {
                let directory = URL(fileURLWithPath: localPath)
                try await manager.initializeFromLocalPath(directory)
                print("Loaded models from: \(localPath)")
            } else {
                try await manager.initialize()
                print("Models downloaded and loaded")
            }

            // Process files
            var results: [StreamingBenchmarkResult] = []
            let startTime = Date()
            let chunkSamples = Int(chunkDurationMs / 1000.0 * 16000.0)

            for (index, file) in filesToProcess.enumerated() {
                let progress = String(format: "%.1f", Double(index + 1) / Double(filesToProcess.count) * 100)
                print("[\(progress)%] Processing: \(file.fileName)")

                do {
                    let result = try await processFile(
                        manager: manager,
                        file: file,
                        chunkSamples: chunkSamples,
                        debug: debugMode
                    )
                    results.append(result)

                    if debugMode {
                        let werPct = String(format: "%.1f", result.wer * 100)
                        let rtfx = String(format: "%.1f", result.rtfx)
                        let avgChunkMs = String(format: "%.1f", result.avgChunkLatencyMs)
                        print("   WER: \(werPct)% | RTFx: \(rtfx)x | Avg chunk latency: \(avgChunkMs)ms")
                    }
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

    // MARK: - LibriSpeech Dataset (Shared logic, duplicated for independence)

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

        if FileManager.default.fileExists(atPath: subsetDirectory.path) {
            let enumerator = FileManager.default.enumerator(
                at: subsetDirectory, includingPropertiesForKeys: nil)
            var transcriptCount = 0
            while let url = enumerator?.nextObject() as? URL {
                if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                    transcriptCount += 1
                    if transcriptCount >= 5 { return }
                }
            }
        }

        print("Downloading LibriSpeech \(subset)...")
        let downloadURL: String
        switch subset {
        case "test-clean":
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-clean.tar.gz").absoluteString
        case "test-other":
            downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "test-other.tar.gz").absoluteString
        default:
            throw ASRError.processingFailed("Unsupported LibriSpeech subset: \(subset)")
        }

        try await downloadAndExtractTarGz(
            url: downloadURL,
            extractTo: datasetsDirectory,
            expectedSubpath: "LibriSpeech/\(subset)"
        )
    }

    private static func downloadAndExtractTarGz(url: String, extractTo: URL, expectedSubpath: String) async throws {
        let downloadURL = URL(string: url)!
        let (tempFile, _) = try await DownloadUtils.sharedSession.download(from: downloadURL)
        try FileManager.default.createDirectory(at: extractTo, withIntermediateDirectories: true)
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", tempFile.path, "-C", extractTo.path]
        try process.run()
        process.waitUntilExit()

        let extractedPath = extractTo.appendingPathComponent(expectedSubpath)
        if FileManager.default.fileExists(atPath: extractedPath.path) {
            let targetPath = extractTo.appendingPathComponent(expectedSubpath.components(separatedBy: "/").last!)
            try? FileManager.default.removeItem(at: targetPath)
            try FileManager.default.moveItem(at: extractedPath, to: targetPath)
            try? FileManager.default.removeItem(at: extractTo.appendingPathComponent("LibriSpeech"))
        }
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
                        files.append(LibriSpeechFile(fileName: audioFileName, audioPath: audioPath, transcript: transcript))
                    }
                }
            }
        }
        return files.sorted { $0.fileName < $1.fileName }
    }

    // MARK: - Processing

    private static func processFile(
        manager: StreamingEouAsrManager,
        file: LibriSpeechFile,
        chunkSamples: Int,
        debug: Bool
    ) async throws -> StreamingBenchmarkResult {
        let audioURL = file.audioPath

        // Load and convert audio
        let audioSamples = try AudioConverter().resampleAudioFile(path: audioURL.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Streaming transcription
        var fullText = ""
        var totalProcessingTime: TimeInterval = 0
        var chunkLatencies: [Double] = []
        var firstTokenLatency: Double?
        var anyEouDetected = false

        let numChunks = (audioSamples.count + chunkSamples - 1) / chunkSamples
        
        // Reset state at start
        manager.resetState()

        for i in 0..<numChunks {
            let start = i * chunkSamples
            let end = min(start + chunkSamples, audioSamples.count)
            let chunk = Array(audioSamples[start..<end])
            let isFinal = (i == numChunks - 1)

            let chunkStart = Date()
            let result = try await manager.processChunk(chunk, isFinal: isFinal)
            let chunkLatency = Date().timeIntervalSince(chunkStart) * 1000  // ms

            chunkLatencies.append(chunkLatency)
            totalProcessingTime += chunkLatency / 1000

            if firstTokenLatency == nil && !result.text.isEmpty {
                firstTokenLatency = chunkLatency
            }

            if !result.text.isEmpty {
                fullText += result.text
            }

            if result.eouDetected {
                anyEouDetected = true
            }
        }
        
        // Get final result (flush)
        // Note: processChunk(isFinal: true) already handles flushing/resetting partial hypothesis

        // Calculate WER
        let metrics = WERCalculator.calculateWERAndCER(
            hypothesis: fullText,
            reference: file.transcript
        )

        let avgChunkLatency = chunkLatencies.isEmpty ? 0 : chunkLatencies.reduce(0, +) / Double(chunkLatencies.count)
        let maxChunkLatency = chunkLatencies.max() ?? 0

        return StreamingBenchmarkResult(
            fileName: file.fileName,
            hypothesis: fullText,
            reference: file.transcript,
            wer: metrics.wer,
            cer: metrics.cer,
            insertions: metrics.insertions,
            deletions: metrics.deletions,
            substitutions: metrics.substitutions,
            totalWords: metrics.totalWords,
            audioLength: audioLength,
            processingTime: totalProcessingTime,
            detectedEou: anyEouDetected,
            chunkCount: chunkLatencies.count,
            avgChunkLatencyMs: avgChunkLatency,
            maxChunkLatencyMs: maxChunkLatency,
            firstTokenLatencyMs: firstTokenLatency
        )
    }

    // MARK: - Results

    private struct StreamingBenchmarkResult: Codable {
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
        let chunkCount: Int
        let avgChunkLatencyMs: Double
        let maxChunkLatencyMs: Double
        let firstTokenLatencyMs: Double?

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
        let totalChunks: Int
        let avgChunkLatencyMs: Double
        let maxChunkLatencyMs: Double
        let avgFirstTokenLatencyMs: Double?
    }

    private static func calculateSummary(results: [StreamingBenchmarkResult]) -> BenchmarkSummary {
        guard !results.isEmpty else {
            return BenchmarkSummary(
                filesProcessed: 0, averageWER: 0, medianWER: 0, averageCER: 0,
                medianRTFx: 0, overallRTFx: 0, totalAudioDuration: 0,
                totalProcessingTime: 0, eouDetectionRate: 0,
                totalChunks: 0, avgChunkLatencyMs: 0, maxChunkLatencyMs: 0,
                avgFirstTokenLatencyMs: nil
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

        let totalChunks = results.reduce(0) { $0 + $1.chunkCount }
        let allAvgLatencies = results.map { $0.avgChunkLatencyMs }
        let avgChunkLatency = allAvgLatencies.reduce(0, +) / Double(allAvgLatencies.count)
        let maxChunkLatency = results.map { $0.maxChunkLatencyMs }.max() ?? 0
        
        let firstTokenLatencies = results.compactMap { $0.firstTokenLatencyMs }
        let avgFirstTokenLatency = firstTokenLatencies.isEmpty ? nil : firstTokenLatencies.reduce(0, +) / Double(firstTokenLatencies.count)

        return BenchmarkSummary(
            filesProcessed: results.count,
            averageWER: avgWER,
            medianWER: medianWER,
            averageCER: avgCER,
            medianRTFx: medianRTFx,
            overallRTFx: overallRTFx,
            totalAudioDuration: totalAudio,
            totalProcessingTime: totalProcessing,
            eouDetectionRate: eouRate,
            totalChunks: totalChunks,
            avgChunkLatencyMs: avgChunkLatency,
            maxChunkLatencyMs: maxChunkLatency,
            avgFirstTokenLatencyMs: avgFirstTokenLatency
        )
    }

    private static func writeResults(
        results: [StreamingBenchmarkResult],
        summary: BenchmarkSummary,
        subset: String,
        outputFile: String
    ) throws {
        var summaryDict: [String: Any] = [
            "filesProcessed": summary.filesProcessed,
            "averageWER": summary.averageWER,
            "medianWER": summary.medianWER,
            "averageCER": summary.averageCER,
            "medianRTFx": summary.medianRTFx,
            "overallRTFx": summary.overallRTFx,
            "totalAudioDuration": summary.totalAudioDuration,
            "totalProcessingTime": summary.totalProcessingTime,
            "eouDetectionRate": summary.eouDetectionRate,
            "totalChunks": summary.totalChunks,
            "avgChunkLatencyMs": summary.avgChunkLatencyMs,
            "maxChunkLatencyMs": summary.maxChunkLatencyMs
        ]
        if let ftl = summary.avgFirstTokenLatencyMs {
            summaryDict["avgFirstTokenLatencyMs"] = ftl
        }

        let output: [String: Any] = [
            "config": [
                "model": "parakeet-realtime-eou-120m-streaming",
                "dataset": "librispeech",
                "subset": subset,
            ],
            "summary": summaryDict,
            "results": results.map { result in
                var r: [String: Any] = [
                    "fileName": result.fileName,
                    "hypothesis": result.hypothesis,
                    "reference": result.reference,
                    "wer": result.wer,
                    "cer": result.cer,
                    "rtfx": result.rtfx,
                    "audioLength": result.audioLength,
                    "processingTime": result.processingTime,
                    "detectedEou": result.detectedEou,
                    "chunkCount": result.chunkCount,
                    "avgChunkLatencyMs": result.avgChunkLatencyMs,
                    "maxChunkLatencyMs": result.maxChunkLatencyMs
                ]
                if let ftl = result.firstTokenLatencyMs {
                    r["firstTokenLatencyMs"] = ftl
                }
                return r
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
        print("STREAMING EOU BENCHMARK RESULTS - LibriSpeech \(subset)")
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
        print("--- Streaming Latency ---")
        print("Total chunks:       \(summary.totalChunks)")
        print("Avg chunk latency:  \(String(format: "%.1f", summary.avgChunkLatencyMs))ms")
        print("Max chunk latency:  \(String(format: "%.1f", summary.maxChunkLatencyMs))ms")
        if let ftl = summary.avgFirstTokenLatencyMs {
            print("Avg first token:    \(String(format: "%.1f", ftl))ms")
        }
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
            Streaming EOU ASR Benchmark Command

            Usage:
                fluidaudio streaming-eou-benchmark [options]

            Options:
                --subset <name>       LibriSpeech subset (default: test-clean)
                --max-files <n>       Maximum files to process
                --output <file>       Output JSON file
                --local-models <path> Use local model directory
                --chunk-duration <ms> Chunk duration in milliseconds (default: 500)
                --debug               Enable debug output
                --help, -h            Show this help
            """
        )
    }
}
#endif
