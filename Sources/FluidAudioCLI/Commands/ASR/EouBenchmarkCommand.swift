#if os(macOS)
import FluidAudio
import Foundation
import OSLog

/// EOU (End-of-Utterance) ASR benchmark command for LibriSpeech evaluation
public enum EouBenchmarkCommand {

    private static let logger = AppLogger(category: "EouBenchmark")

    public static func run(arguments: [String]) async {
        // Check for help flag
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var subset = "test-clean"
        var maxFiles: Int?
        var outputFile = "eou_benchmark_results.json"
        var debugMode = false
        var localModelsPath: String?
        var streamingMode = false
        var chunkDurationMs: Double = 2000  // Default 2 second chunks for streaming

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
            case "--streaming":
                streamingMode = true
            case "--chunk-duration":
                if i + 1 < arguments.count {
                    chunkDurationMs = Double(arguments[i + 1]) ?? 2000
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        print("Starting EOU ASR benchmark on LibriSpeech \(subset)")
        print("   Mode: \(streamingMode ? "streaming (\(Int(chunkDurationMs))ms chunks)" : "batch")")
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

            // Initialize EOU model
            print("Initializing Parakeet EOU model...")
            let eouManager = EouAsrManager()

            if let localPath = localModelsPath {
                let directory = URL(fileURLWithPath: localPath)
                try await eouManager.initializeFromLocalPath(directory)
                print("Loaded models from: \(localPath)")
            } else {
                try await eouManager.initialize()
                print("Models downloaded and loaded")
            }

            // Process files
            var results: [EouBenchmarkResult] = []
            let startTime = Date()
            let chunkSamples = Int(chunkDurationMs / 1000.0 * 16000.0)

            for (index, file) in filesToProcess.enumerated() {
                let progress = String(format: "%.1f", Double(index + 1) / Double(filesToProcess.count) * 100)
                print("[\(progress)%] Processing: \(file.fileName)")

                do {
                    let result: EouBenchmarkResult
                    if streamingMode {
                        result = try await processFileStreaming(
                            eouManager: eouManager,
                            file: file,
                            chunkSamples: chunkSamples,
                            debug: debugMode
                        )
                    } else {
                        result = try await processFile(eouManager: eouManager, file: file, debug: debugMode)
                    }
                    results.append(result)

                    if debugMode {
                        let werPct = String(format: "%.1f", result.wer * 100)
                        let rtfx = String(format: "%.1f", result.rtfx)
                        if streamingMode {
                            let avgChunkMs = String(format: "%.1f", (result.avgChunkLatencyMs ?? 0))
                            print("   WER: \(werPct)% | RTFx: \(rtfx)x | Avg chunk latency: \(avgChunkMs)ms")
                        } else {
                            print("   WER: \(werPct)% | RTFx: \(rtfx)x")
                        }
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

    // MARK: - LibriSpeech Dataset

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
        eouManager: EouAsrManager,
        file: LibriSpeechFile,
        debug: Bool
    ) async throws -> EouBenchmarkResult {
        let audioURL = file.audioPath

        // Load and convert audio
        let audioSamples = try AudioConverter().resampleAudioFile(path: audioURL.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Transcribe
        let inferenceStart = Date()
        let transcription = try await eouManager.transcribe(audioSamples)
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

    /// Process file in streaming mode with configurable chunk size
    private static func processFileStreaming(
        eouManager: EouAsrManager,
        file: LibriSpeechFile,
        chunkSamples: Int,
        debug: Bool
    ) async throws -> EouBenchmarkResult {
        let audioURL = file.audioPath

        // Load and convert audio
        let audioSamples = try AudioConverter().resampleAudioFile(path: audioURL.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        // Streaming transcription with chunks
        let maxSamples = EouAsrManager.maxAudioSamples
        var fullText = ""
        var totalProcessingTime: TimeInterval = 0
        var chunkLatencies: [Double] = []
        var firstTokenLatency: Double?
        var anyEouDetected = false

        // Use overlap for better continuity (10% overlap)
        let overlapSamples = chunkSamples / 10
        let stride = chunkSamples - overlapSamples
        var offset = 0
        var chunkIndex = 0

        // Reset state at start
        eouManager.resetState()

        while offset < audioSamples.count {
            // Get chunk (limited by maxSamples)
            let endIndex = min(offset + min(chunkSamples, maxSamples), audioSamples.count)
            let chunk = Array(audioSamples[offset..<endIndex])

            let chunkStart = Date()
            let result = try await eouManager.transcribe(chunk)
            let chunkLatency = Date().timeIntervalSince(chunkStart) * 1000  // Convert to ms

            chunkLatencies.append(chunkLatency)
            totalProcessingTime += chunkLatency / 1000

            // Track first token latency
            if firstTokenLatency == nil && !result.text.isEmpty {
                firstTokenLatency = chunkLatency
            }

            if !result.text.isEmpty {
                if !fullText.isEmpty {
                    fullText += " "
                }
                fullText += result.text
            }

            if result.eouDetected {
                anyEouDetected = true
            }

            offset += stride
            chunkIndex += 1

            // Reset state between chunks for independent processing
            eouManager.resetState()
        }

        // Calculate WER
        let metrics = WERCalculator.calculateWERAndCER(
            hypothesis: fullText,
            reference: file.transcript
        )

        let avgChunkLatency = chunkLatencies.isEmpty ? 0 : chunkLatencies.reduce(0, +) / Double(chunkLatencies.count)
        let maxChunkLatency = chunkLatencies.max() ?? 0

        return EouBenchmarkResult(
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

        // Streaming-specific metrics
        let chunkCount: Int?
        let avgChunkLatencyMs: Double?
        let maxChunkLatencyMs: Double?
        let firstTokenLatencyMs: Double?

        var rtfx: Double {
            processingTime > 0 ? audioLength / processingTime : 0
        }

        // Non-streaming init
        init(
            fileName: String, hypothesis: String, reference: String,
            wer: Double, cer: Double, insertions: Int, deletions: Int,
            substitutions: Int, totalWords: Int, audioLength: Double,
            processingTime: Double, detectedEou: Bool
        ) {
            self.fileName = fileName
            self.hypothesis = hypothesis
            self.reference = reference
            self.wer = wer
            self.cer = cer
            self.insertions = insertions
            self.deletions = deletions
            self.substitutions = substitutions
            self.totalWords = totalWords
            self.audioLength = audioLength
            self.processingTime = processingTime
            self.detectedEou = detectedEou
            self.chunkCount = nil
            self.avgChunkLatencyMs = nil
            self.maxChunkLatencyMs = nil
            self.firstTokenLatencyMs = nil
        }

        // Streaming init
        init(
            fileName: String, hypothesis: String, reference: String,
            wer: Double, cer: Double, insertions: Int, deletions: Int,
            substitutions: Int, totalWords: Int, audioLength: Double,
            processingTime: Double, detectedEou: Bool,
            chunkCount: Int, avgChunkLatencyMs: Double, maxChunkLatencyMs: Double,
            firstTokenLatencyMs: Double?
        ) {
            self.fileName = fileName
            self.hypothesis = hypothesis
            self.reference = reference
            self.wer = wer
            self.cer = cer
            self.insertions = insertions
            self.deletions = deletions
            self.substitutions = substitutions
            self.totalWords = totalWords
            self.audioLength = audioLength
            self.processingTime = processingTime
            self.detectedEou = detectedEou
            self.chunkCount = chunkCount
            self.avgChunkLatencyMs = avgChunkLatencyMs
            self.maxChunkLatencyMs = maxChunkLatencyMs
            self.firstTokenLatencyMs = firstTokenLatencyMs
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

        // Streaming-specific metrics
        let totalChunks: Int?
        let avgChunkLatencyMs: Double?
        let maxChunkLatencyMs: Double?
        let avgFirstTokenLatencyMs: Double?
    }

    private static func calculateSummary(results: [EouBenchmarkResult]) -> BenchmarkSummary {
        guard !results.isEmpty else {
            return BenchmarkSummary(
                filesProcessed: 0, averageWER: 0, medianWER: 0, averageCER: 0,
                medianRTFx: 0, overallRTFx: 0, totalAudioDuration: 0,
                totalProcessingTime: 0, eouDetectionRate: 0,
                totalChunks: nil, avgChunkLatencyMs: nil, maxChunkLatencyMs: nil,
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

        // Calculate streaming metrics if available
        let streamingResults = results.filter { $0.chunkCount != nil }
        var totalChunks: Int?
        var avgChunkLatency: Double?
        var maxChunkLatency: Double?
        var avgFirstTokenLatency: Double?

        if !streamingResults.isEmpty {
            totalChunks = streamingResults.reduce(0) { $0 + ($1.chunkCount ?? 0) }
            let allAvgLatencies = streamingResults.compactMap { $0.avgChunkLatencyMs }
            avgChunkLatency =
                allAvgLatencies.isEmpty ? nil : allAvgLatencies.reduce(0, +) / Double(allAvgLatencies.count)
            maxChunkLatency = streamingResults.compactMap { $0.maxChunkLatencyMs }.max()
            let firstTokenLatencies = streamingResults.compactMap { $0.firstTokenLatencyMs }
            avgFirstTokenLatency =
                firstTokenLatencies.isEmpty ? nil : firstTokenLatencies.reduce(0, +) / Double(firstTokenLatencies.count)
        }

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
        results: [EouBenchmarkResult],
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
        ]

        // Add streaming metrics if available
        if let totalChunks = summary.totalChunks {
            summaryDict["totalChunks"] = totalChunks
        }
        if let avgChunkLatency = summary.avgChunkLatencyMs {
            summaryDict["avgChunkLatencyMs"] = avgChunkLatency
        }
        if let maxChunkLatency = summary.maxChunkLatencyMs {
            summaryDict["maxChunkLatencyMs"] = maxChunkLatency
        }
        if let avgFirstTokenLatency = summary.avgFirstTokenLatencyMs {
            summaryDict["avgFirstTokenLatencyMs"] = avgFirstTokenLatency
        }

        let output: [String: Any] = [
            "config": [
                "model": "parakeet-realtime-eou-120m",
                "dataset": "librispeech",
                "subset": subset,
            ],
            "summary": summaryDict,
            "results": results.map { result in
                var resultDict: [String: Any] = [
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
                // Add streaming metrics if available
                if let chunkCount = result.chunkCount {
                    resultDict["chunkCount"] = chunkCount
                }
                if let avgChunkLatency = result.avgChunkLatencyMs {
                    resultDict["avgChunkLatencyMs"] = avgChunkLatency
                }
                if let maxChunkLatency = result.maxChunkLatencyMs {
                    resultDict["maxChunkLatencyMs"] = maxChunkLatency
                }
                if let firstTokenLatency = result.firstTokenLatencyMs {
                    resultDict["firstTokenLatencyMs"] = firstTokenLatency
                }
                return resultDict
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
        print("EOU BENCHMARK RESULTS - LibriSpeech \(subset)")
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

        // Print streaming metrics if available
        if let totalChunks = summary.totalChunks {
            print("")
            print("--- Streaming Latency ---")
            print("Total chunks:       \(totalChunks)")
            if let avgChunkLatency = summary.avgChunkLatencyMs {
                print("Avg chunk latency:  \(String(format: "%.1f", avgChunkLatency))ms")
            }
            if let maxChunkLatency = summary.maxChunkLatencyMs {
                print("Max chunk latency:  \(String(format: "%.1f", maxChunkLatency))ms")
            }
            if let avgFirstTokenLatency = summary.avgFirstTokenLatencyMs {
                print("Avg first token:    \(String(format: "%.1f", avgFirstTokenLatency))ms")
            }
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
            EOU ASR Benchmark Command

            Usage:
                fluidaudio eou-benchmark [options]

            Options:
                --subset <name>       LibriSpeech subset (default: test-clean)
                                     Available: test-clean, test-other
                --max-files <n>       Maximum files to process (default: all)
                --output <file>       Output JSON file (default: eou_benchmark_results.json)
                --local-models <path> Use local model directory instead of downloading
                --streaming           Enable streaming mode (process audio in chunks)
                --chunk-duration <ms> Chunk duration in milliseconds (default: 2000)
                --debug               Enable debug output
                --help, -h            Show this help

            Examples:
                # Run full benchmark on test-clean (batch mode)
                fluidaudio eou-benchmark

                # Quick test with 10 files
                fluidaudio eou-benchmark --max-files 10 --debug

                # Run on test-other subset
                fluidaudio eou-benchmark --subset test-other --output results_other.json

                # Run in streaming mode with 2-second chunks
                fluidaudio eou-benchmark --streaming --chunk-duration 2000

                # Run in streaming mode with 500ms chunks
                fluidaudio eou-benchmark --streaming --chunk-duration 500 --max-files 10

            Expected Performance:
                - test-clean: ~3-8% WER (batch), ~5-12% WER (streaming)
                - test-other: ~8-15% WER (batch), ~12-20% WER (streaming)
                - RTFx: >10x on Apple Silicon
                - Streaming latency: ~20-50ms per chunk on Apple Silicon
            """
        )
    }
}
#endif
