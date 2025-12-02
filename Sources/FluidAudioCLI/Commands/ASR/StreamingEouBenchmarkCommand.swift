#if false
#if os(macOS)
import FluidAudio
import Foundation
import OSLog

/// Streaming EOU benchmark using cache-aware StreamingEouAsrManager
public enum StreamingEouBenchmarkCommand {

    private static let logger = AppLogger(category: "StreamingEouBenchmark")

    public static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }

        // Parse arguments
        var subset = "test-clean"
        var maxFiles: Int?
        var outputFile = "streaming_eou_benchmark_results.json"
        var debugMode = false
        var modelsPath = "HuggingFace/parakeet-realtime-eou-120m-coreml/models"
        var chunkDurationMs: Double = 730  // Default for mode 2 (73 mel frames)

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
            case "--models":
                if i + 1 < arguments.count {
                    modelsPath = arguments[i + 1]
                    i += 1
                }
            case "--chunk-duration":
                if i + 1 < arguments.count {
                    chunkDurationMs = Double(arguments[i + 1]) ?? 730
                    i += 1
                }
            case "--debug":
                debugMode = true
            default:
                break
            }
            i += 1
        }

        print("=== Streaming EOU Benchmark (Cache-Aware) ===")
        print("Models: \(modelsPath)")
        print("Subset: \(subset)")
        print("Chunk duration: \(Int(chunkDurationMs))ms")
        print("Max files: \(maxFiles?.description ?? "all")")
        print("Output: \(outputFile)")
        print("")

        do {
            // Download LibriSpeech if needed
            print("Checking LibriSpeech \(subset) dataset...")
            try await downloadLibriSpeech(subset: subset)

            // Collect audio files
            let datasetPath = getLibriSpeechDirectory().appendingPathComponent(subset)
            let audioFiles = try collectLibriSpeechFiles(from: datasetPath)
            let filesToProcess = maxFiles != nil ? Array(audioFiles.prefix(maxFiles!)) : audioFiles

            print("Found \(audioFiles.count) files, processing \(filesToProcess.count)")

            // Initialize streaming model
            print("Loading streaming models...")
            let modelsURL = URL(fileURLWithPath: modelsPath)
            let streamingManager = StreamingEouAsrManager()
            try await streamingManager.initializeFromLocalPath(modelsURL)
            print("Models loaded")
            print("")

            // Process files
            var results: [BenchmarkResult] = []
            let startTime = Date()
            let chunkSamples = Int(chunkDurationMs / 1000.0 * 16000.0)

            for (index, file) in filesToProcess.enumerated() {
                let progress = String(format: "%.1f", Double(index + 1) / Double(filesToProcess.count) * 100)
                print("[\(progress)%] \(file.fileName)", terminator: "")

                do {
                    let result = try await processFile(
                        manager: streamingManager,
                        file: file,
                        chunkSamples: chunkSamples,
                        debug: debugMode
                    )
                    results.append(result)

                    let werPct = String(format: "%.1f", result.wer * 100)
                    let rtfx = String(format: "%.1f", result.rtfx)
                    let latency = String(format: "%.0f", result.avgChunkLatencyMs)
                    print(" - WER: \(werPct)% | RTFx: \(rtfx)x | Latency: \(latency)ms")

                    // Reset for next file
                    streamingManager.resetState()
                } catch {
                    print(" - ERROR: \(error.localizedDescription)")
                }
            }

            let totalTime = Date().timeIntervalSince(startTime)

            // Calculate summary
            let summary = calculateSummary(results: results)

            // Write results
            try writeResults(results: results, summary: summary, subset: subset, outputFile: outputFile)

            // Print summary
            printSummary(summary: summary, totalTime: totalTime, subset: subset)

        } catch {
            print("ERROR: \(error)")
        }
    }

    // MARK: - Processing

    private static func processFile(
        manager: StreamingEouAsrManager,
        file: LibriSpeechFile,
        chunkSamples: Int,
        debug: Bool
    ) async throws -> BenchmarkResult {
        let audioSamples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
        let audioLength = TimeInterval(audioSamples.count) / 16000.0

        var fullText = ""
        var chunkLatencies: [Double] = []
        var eouDetected = false
        var offset = 0
        var chunkIndex = 0

        while offset < audioSamples.count {
            let endIndex = min(offset + chunkSamples, audioSamples.count)
            let chunk = Array(audioSamples[offset..<endIndex])
            let isFinal = endIndex >= audioSamples.count

            let chunkStart = Date()
            let result = try await manager.processChunk(chunk, isFinal: isFinal)
            let chunkLatency = Date().timeIntervalSince(chunkStart) * 1000

            chunkLatencies.append(chunkLatency)

            if !result.text.isEmpty {
                if !fullText.isEmpty && !fullText.hasSuffix(" ") {
                    fullText += " "
                }
                fullText += result.text.trimmingCharacters(in: .whitespaces)
            }

            if debug {
                print("  Chunk \(chunkIndex): '\(result.text)' (EOU=\(result.eouDetected))")
            }

            if result.eouDetected {
                eouDetected = true
            }

            offset += chunkSamples
            chunkIndex += 1
        }

        // Clean up text
        fullText = fullText
            .replacingOccurrences(of: "<EOU>", with: "")
            .trimmingCharacters(in: .whitespaces)

        let metrics = WERCalculator.calculateWERAndCER(
            hypothesis: fullText,
            reference: file.transcript
        )

        let totalProcessingTime = chunkLatencies.reduce(0, +) / 1000.0
        let avgChunkLatency = chunkLatencies.isEmpty ? 0 : chunkLatencies.reduce(0, +) / Double(chunkLatencies.count)

        return BenchmarkResult(
            fileName: file.fileName,
            hypothesis: fullText,
            reference: file.transcript,
            wer: metrics.wer,
            cer: metrics.cer,
            audioLength: audioLength,
            processingTime: totalProcessingTime,
            chunkCount: chunkLatencies.count,
            avgChunkLatencyMs: avgChunkLatency,
            maxChunkLatencyMs: chunkLatencies.max() ?? 0,
            eouDetected: eouDetected
        )
    }

    // MARK: - LibriSpeech

    private static func getLibriSpeechDirectory() -> URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("FluidAudio/Datasets/LibriSpeech", isDirectory: true)
    }

    private static func downloadLibriSpeech(subset: String) async throws {
        let subsetDir = getLibriSpeechDirectory().appendingPathComponent(subset)

        if FileManager.default.fileExists(atPath: subsetDir.path) {
            let enumerator = FileManager.default.enumerator(at: subsetDir, includingPropertiesForKeys: nil)
            var count = 0
            while let url = enumerator?.nextObject() as? URL {
                if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                    count += 1
                    if count >= 5 {
                        print("LibriSpeech \(subset) already downloaded")
                        return
                    }
                }
            }
        }

        print("Downloading LibriSpeech \(subset)...")
        let downloadURL = try ModelRegistry.resolveDataset("FluidInference/librispeech", "\(subset).tar.gz")

        let (tempFile, _) = try await DownloadUtils.sharedSession.download(from: downloadURL)
        let extractDir = getLibriSpeechDirectory()
        try FileManager.default.createDirectory(at: extractDir, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        process.arguments = ["-xzf", tempFile.path, "-C", extractDir.path]
        try process.run()
        process.waitUntilExit()

        // Move extracted content
        let extractedPath = extractDir.appendingPathComponent("LibriSpeech/\(subset)")
        if FileManager.default.fileExists(atPath: extractedPath.path) {
            let targetPath = extractDir.appendingPathComponent(subset)
            try? FileManager.default.removeItem(at: targetPath)
            try FileManager.default.moveItem(at: extractedPath, to: targetPath)
            try? FileManager.default.removeItem(at: extractDir.appendingPathComponent("LibriSpeech"))
        }

        print("Downloaded successfully")
    }

    private static func collectLibriSpeechFiles(from directory: URL) throws -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: nil)

        while let url = enumerator?.nextObject() as? URL {
            if url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") {
                let content = try String(contentsOf: url)
                for line in content.components(separatedBy: .newlines) where !line.isEmpty {
                    let parts = line.components(separatedBy: " ")
                    guard parts.count >= 2 else { continue }

                    let audioId = parts[0]
                    let transcript = parts.dropFirst().joined(separator: " ")
                    let audioPath = url.deletingLastPathComponent().appendingPathComponent("\(audioId).flac")

                    if fm.fileExists(atPath: audioPath.path) {
                        files.append(LibriSpeechFile(fileName: "\(audioId).flac", audioPath: audioPath, transcript: transcript))
                    }
                }
            }
        }

        return files.sorted { $0.fileName < $1.fileName }
    }

    // MARK: - Results

    private struct BenchmarkResult: Codable {
        let fileName: String
        let hypothesis: String
        let reference: String
        let wer: Double
        let cer: Double
        let audioLength: Double
        let processingTime: Double
        let chunkCount: Int
        let avgChunkLatencyMs: Double
        let maxChunkLatencyMs: Double
        let eouDetected: Bool

        var rtfx: Double { processingTime > 0 ? audioLength / processingTime : 0 }
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
        let totalChunks: Int
        let avgChunkLatencyMs: Double
        let maxChunkLatencyMs: Double
        let eouDetectionRate: Double
    }

    private static func calculateSummary(results: [BenchmarkResult]) -> BenchmarkSummary {
        guard !results.isEmpty else {
            return BenchmarkSummary(
                filesProcessed: 0, averageWER: 0, medianWER: 0, averageCER: 0,
                medianRTFx: 0, overallRTFx: 0, totalAudioDuration: 0, totalProcessingTime: 0,
                totalChunks: 0, avgChunkLatencyMs: 0, maxChunkLatencyMs: 0, eouDetectionRate: 0
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

        let totalChunks = results.reduce(0) { $0 + $1.chunkCount }
        let allLatencies = results.map { $0.avgChunkLatencyMs }
        let avgChunkLatency = allLatencies.reduce(0, +) / Double(allLatencies.count)
        let maxChunkLatency = results.map { $0.maxChunkLatencyMs }.max() ?? 0

        let eouCount = results.filter { $0.eouDetected }.count
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
            totalChunks: totalChunks,
            avgChunkLatencyMs: avgChunkLatency,
            maxChunkLatencyMs: maxChunkLatency,
            eouDetectionRate: eouRate
        )
    }

    private static func writeResults(
        results: [BenchmarkResult],
        summary: BenchmarkSummary,
        subset: String,
        outputFile: String
    ) throws {
        let output: [String: Any] = [
            "config": [
                "model": "parakeet-realtime-eou-120m (streaming)",
                "encoder": "cache-aware streaming encoder",
                "dataset": "librispeech",
                "subset": subset,
            ],
            "summary": [
                "filesProcessed": summary.filesProcessed,
                "averageWER": summary.averageWER,
                "medianWER": summary.medianWER,
                "averageCER": summary.averageCER,
                "medianRTFx": summary.medianRTFx,
                "overallRTFx": summary.overallRTFx,
                "totalAudioDuration": summary.totalAudioDuration,
                "totalProcessingTime": summary.totalProcessingTime,
                "totalChunks": summary.totalChunks,
                "avgChunkLatencyMs": summary.avgChunkLatencyMs,
                "maxChunkLatencyMs": summary.maxChunkLatencyMs,
                "eouDetectionRate": summary.eouDetectionRate,
            ],
            "results": results.map { r in
                [
                    "fileName": r.fileName,
                    "hypothesis": r.hypothesis,
                    "reference": r.reference,
                    "wer": r.wer,
                    "cer": r.cer,
                    "rtfx": r.rtfx,
                    "audioLength": r.audioLength,
                    "processingTime": r.processingTime,
                    "chunkCount": r.chunkCount,
                    "avgChunkLatencyMs": r.avgChunkLatencyMs,
                    "eouDetected": r.eouDetected,
                ] as [String: Any]
            },
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: outputFile))
        print("\nResults written to: \(outputFile)")
    }

    private static func printSummary(summary: BenchmarkSummary, totalTime: TimeInterval, subset: String) {
        print("")
        print(String(repeating: "=", count: 60))
        print("STREAMING EOU BENCHMARK - LibriSpeech \(subset)")
        print(String(repeating: "=", count: 60))
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
        print("")
        print("--- EOU Detection ---")
        print("EOU detection rate: \(String(format: "%.1f", summary.eouDetectionRate * 100))%")
        print("")
        print("Benchmark time:     \(String(format: "%.1f", totalTime))s")
        print(String(repeating: "=", count: 60))
    }

    private static func printUsage() {
        print("""
        Streaming EOU Benchmark (Cache-Aware Encoder)

        Usage:
            fluidaudio streaming-eou-benchmark [options]

        Options:
            --subset <name>       LibriSpeech subset (default: test-clean)
            --max-files <n>       Maximum files to process (default: all)
            --output <file>       Output JSON file (default: streaming_eou_benchmark_results.json)
            --models <path>       Path to streaming models (default: HuggingFace/parakeet-realtime-eou-120m-coreml/models)
            --chunk-duration <ms> Chunk duration in milliseconds (default: 730)
            --debug               Enable debug output
            --help, -h            Show this help

        Examples:
            # Quick test with 10 files
            fluidaudio streaming-eou-benchmark --max-files 10

            # Full test-clean benchmark
            fluidaudio streaming-eou-benchmark

            # Test with different chunk size
            fluidaudio streaming-eou-benchmark --chunk-duration 500 --max-files 20
        """)
    }
}
#endif
#endif
