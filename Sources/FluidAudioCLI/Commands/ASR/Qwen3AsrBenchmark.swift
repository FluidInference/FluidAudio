#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// LibriSpeech benchmark for Qwen3-ASR.
///
/// Reuses the standard LibriSpeech test-clean/test-other datasets with WER/CER evaluation,
/// but runs inference through `Qwen3AsrManager` instead of the Parakeet TDT pipeline.
enum Qwen3AsrBenchmark {
    private static let logger = AppLogger(category: "Qwen3Benchmark")

    static func runCLI(arguments: [String]) async {
        var subset = "test-clean"
        var maxFiles: Int? = nil
        var modelDir: String? = nil
        var outputFile = "qwen3_asr_benchmark_results.json"

        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

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
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        logger.info("Qwen3-ASR Benchmark")
        logger.info("  Subset: \(subset)")
        logger.info("  Max files: \(maxFiles?.description ?? "all")")
        logger.info("  Model dir: \(modelDir ?? "auto-download")")
        logger.info("  Output: \(outputFile)")

        do {
            // 1. Ensure dataset is available
            let benchmark = ASRBenchmark()
            try await benchmark.downloadLibriSpeech(subset: subset)
            let datasetPath = benchmark.getLibriSpeechDirectory().appendingPathComponent(subset)
            let allFiles = try collectLibriSpeechFiles(from: datasetPath)
            let files = Array(allFiles.prefix(maxFiles ?? allFiles.count))
            logger.info("Collected \(files.count) files from LibriSpeech \(subset)")

            // 2. Load Qwen3-ASR models
            let manager = Qwen3AsrManager()
            if let dir = modelDir {
                logger.info("Loading models from \(dir)")
                try await manager.loadModels(from: URL(fileURLWithPath: dir))
            } else {
                logger.info("Downloading Qwen3-ASR models...")
                let cacheDir = try await Qwen3AsrModels.download()
                try await manager.loadModels(from: cacheDir)
            }

            // 3. Run benchmark
            let startBenchmark = CFAbsoluteTimeGetCurrent()
            var results: [Qwen3BenchmarkResult] = []

            for (index, file) in files.enumerated() {
                do {
                    logger.info("[\(index + 1)/\(files.count)] \(file.fileName)")

                    let samples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
                    let audioLength = Double(samples.count) / 16000.0

                    let inferenceStart = CFAbsoluteTimeGetCurrent()
                    let hypothesis = try await manager.transcribe(audioSamples: samples)
                    let inferenceTime = CFAbsoluteTimeGetCurrent() - inferenceStart

                    let metrics = WERCalculator.calculateWERAndCER(
                        hypothesis: hypothesis, reference: file.transcript
                    )

                    let result = Qwen3BenchmarkResult(
                        fileName: file.fileName,
                        hypothesis: hypothesis,
                        reference: file.transcript,
                        wer: metrics.wer,
                        cer: metrics.cer,
                        audioLength: audioLength,
                        processingTime: inferenceTime
                    )
                    results.append(result)

                    let rtfx = audioLength / inferenceTime
                    let werPct = metrics.wer * 100
                    logger.info(
                        "  WER: \(String(format: "%.1f", werPct))% | RTFx: \(String(format: "%.1f", rtfx))x | \(String(format: "%.2f", audioLength))s audio in \(String(format: "%.2f", inferenceTime))s"
                    )
                    if werPct > 50.0 {
                        logger.info("  REF: \(file.transcript)")
                        logger.info("  HYP: \(hypothesis)")
                    }
                } catch {
                    logger.error("Failed \(file.fileName): \(error)")
                }
            }

            let totalElapsed = CFAbsoluteTimeGetCurrent() - startBenchmark

            // 4. Print summary
            guard !results.isEmpty else {
                logger.error("No results produced")
                exit(1)
            }

            let avgWER = results.map(\.wer).reduce(0, +) / Double(results.count)
            let avgCER = results.map(\.cer).reduce(0, +) / Double(results.count)
            let totalAudio = results.map(\.audioLength).reduce(0, +)
            let totalInference = results.map(\.processingTime).reduce(0, +)
            let overallRTFx = totalAudio / totalInference
            let medianWER = results.map(\.wer).sorted()[results.count / 2]

            let sortedRTFx = results.map { $0.audioLength / $0.processingTime }.sorted()
            let medianRTFx = sortedRTFx[sortedRTFx.count / 2]

            print("")
            print("--- Qwen3-ASR Benchmark Results ---")
            print("   Dataset: LibriSpeech \(subset)")
            print("   Files processed: \(results.count)")
            print("   Average WER: \(String(format: "%.1f", avgWER * 100))%")
            print("   Median WER: \(String(format: "%.1f", medianWER * 100))%")
            print("   Average CER: \(String(format: "%.1f", avgCER * 100))%")
            print("   Median RTFx: \(String(format: "%.1f", medianRTFx))x")
            print(
                "   Overall RTFx: \(String(format: "%.1f", overallRTFx))x (\(String(format: "%.1f", totalAudio))s / \(String(format: "%.1f", totalInference))s)"
            )
            print("   Wall clock: \(String(format: "%.0f", totalElapsed))s")

            // 5. Write JSON
            let summary: [String: Any] = [
                "model": "qwen3-asr-0.6b",
                "dataset": "librispeech",
                "subset": subset,
                "filesProcessed": results.count,
                "averageWER": avgWER,
                "medianWER": medianWER,
                "averageCER": avgCER,
                "medianRTFx": medianRTFx,
                "overallRTFx": overallRTFx,
                "totalAudioDuration": totalAudio,
                "totalInferenceTime": totalInference,
                "wallClockTime": totalElapsed,
            ]

            let jsonResults = results.map { r -> [String: Any] in
                [
                    "fileName": r.fileName,
                    "hypothesis": r.hypothesis,
                    "reference": r.reference,
                    "wer": r.wer,
                    "cer": r.cer,
                    "audioLength": r.audioLength,
                    "processingTime": r.processingTime,
                    "rtfx": r.audioLength / r.processingTime,
                ]
            }

            let output: [String: Any] = [
                "summary": summary,
                "results": jsonResults,
            ]

            let jsonData = try JSONSerialization.data(
                withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
            try jsonData.write(to: URL(fileURLWithPath: outputFile))
            logger.info("Results written to \(outputFile)")

        } catch {
            logger.error("Benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - LibriSpeech File Collection

    private static func collectLibriSpeechFiles(from directory: URL) throws -> [LibriSpeechFile] {
        var files: [LibriSpeechFile] = []
        let fileManager = FileManager.default
        let enumerator = fileManager.enumerator(at: directory, includingPropertiesForKeys: nil)

        while let url = enumerator?.nextObject() as? URL {
            guard url.pathExtension == "txt" && url.lastPathComponent.contains(".trans.") else {
                continue
            }
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

        return files.sorted { $0.fileName < $1.fileName }
    }

    // MARK: - Usage

    private static func printUsage() {
        logger.info(
            """

            Qwen3-ASR Benchmark

            Usage: fluidaudio qwen3-benchmark [options]

            Options:
                --subset <name>         LibriSpeech subset (default: test-clean)
                --max-files <number>    Max files to process (default: all)
                --model-dir <path>      Local model directory (skips download)
                --output <file>         Output JSON path (default: qwen3_asr_benchmark_results.json)
                --help, -h              Show this help

            Examples:
                fluidaudio qwen3-benchmark --max-files 100
                fluidaudio qwen3-benchmark --subset test-other --max-files 50
                fluidaudio qwen3-benchmark --model-dir /path/to/qwen3-asr
            """
        )
    }
}

// MARK: - Result Type

private struct Qwen3BenchmarkResult {
    let fileName: String
    let hypothesis: String
    let reference: String
    let wer: Double
    let cer: Double
    let audioLength: Double
    let processingTime: Double
}
#endif
