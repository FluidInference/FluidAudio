#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

enum CtcZhCnBenchmark {
    private static let logger = AppLogger(category: "CtcZhCnBenchmark")

    static func run(arguments: [String]) async {
        var numSamples = 100
        var useInt8 = true
        var outputFile: String?
        var verbose = false
        var datasetPath: String?
        var autoDownload = false

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--samples", "-n":
                if i + 1 < arguments.count {
                    numSamples = Int(arguments[i + 1]) ?? 100
                    i += 1
                }
            case "--fp32":
                useInt8 = false
            case "--int8":
                useInt8 = true
            case "--output", "-o":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--dataset-path":
                if i + 1 < arguments.count {
                    datasetPath = arguments[i + 1]
                    i += 1
                }
            case "--auto-download":
                autoDownload = true
            case "--verbose", "-v":
                verbose = true
            case "--help", "-h":
                printUsage()
                return
            default:
                break
            }
            i += 1
        }

        logger.info("=== Parakeet CTC zh-CN Benchmark ===")
        logger.info("Encoder: \(useInt8 ? "int8 (0.55GB)" : "fp32 (1.1GB)")")
        logger.info("Samples: \(numSamples)")
        logger.info("")

        do {
            // Load models
            logger.info("Loading CTC zh-CN models...")
            let manager = try await CtcZhCnManager.load(
                useInt8Encoder: useInt8,
                progressHandler: verbose ? createProgressHandler() : nil
            )
            logger.info("Models loaded successfully")

            // Load THCHS-30 dataset
            logger.info("")
            logger.info("Loading THCHS-30 test set...")
            let samples = try await loadTHCHS30Samples(
                maxSamples: numSamples,
                datasetPath: datasetPath,
                autoDownload: autoDownload
            )
            logger.info("Loaded \(samples.count) samples")

            // Run benchmark
            logger.info("")
            logger.info("Running transcription benchmark...")
            let results = try await runBenchmark(manager: manager, samples: samples)

            // Print results
            printResults(results: results, encoderType: useInt8 ? "int8" : "fp32")

            // Save to JSON if requested
            if let outputFile = outputFile {
                try saveResults(results: results, outputFile: outputFile)
                logger.info("")
                logger.info("Results saved to: \(outputFile)")
            }

        } catch {
            logger.error("Benchmark failed: \(error.localizedDescription)")
            if verbose {
                logger.error("Error details: \(String(describing: error))")
            }
        }
    }

    private struct BenchmarkSample {
        let audioPath: String
        let reference: String
        let sampleId: Int
    }

    private struct BenchmarkResult: Codable {
        let sampleId: Int
        let reference: String
        let hypothesis: String
        let normalizedRef: String
        let normalizedHyp: String
        let cer: Double
        let latencyMs: Double
        let audioDurationSec: Double
        let rtfx: Double
    }

    private struct MetadataEntry: Codable {
        let file_name: String
        let text: String
    }

    private static func loadTHCHS30Samples(
        maxSamples: Int, datasetPath: String?, autoDownload: Bool
    ) async throws -> [BenchmarkSample] {
        let baseDir: URL

        if let path = datasetPath {
            // Use provided path
            baseDir = URL(fileURLWithPath: path)
        } else if autoDownload {
            // Download from HuggingFace to cache directory
            #if os(macOS)
            let homeDir = FileManager.default.homeDirectoryForCurrentUser
            let cacheDir =
                homeDir
                .appendingPathComponent("Library/Application Support/FluidAudio/Datasets/THCHS-30")
            #else
            let cacheDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("FluidAudio/Datasets/THCHS-30")
            #endif

            try FileManager.default.createDirectory(
                at: cacheDir, withIntermediateDirectories: true)

            logger.info("Downloading THCHS-30 from HuggingFace...")
            try await downloadTHCHS30Dataset(to: cacheDir)
            baseDir = cacheDir
        } else {
            throw NSError(
                domain: "CtcZhCnBenchmark",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        """
                    THCHS-30 dataset not found.

                    Options:
                    1. Use --auto-download to download from HuggingFace
                    2. Use --dataset-path <path> to specify local dataset directory

                    Expected directory structure:
                        <path>/
                        ├── audio/           # WAV files
                        └── metadata.jsonl   # Transcripts
                    """
                ]
            )
        }

        // Load metadata.jsonl
        let metadataPath = baseDir.appendingPathComponent("metadata.jsonl")
        guard FileManager.default.fileExists(atPath: metadataPath.path) else {
            throw NSError(
                domain: "CtcZhCnBenchmark",
                code: 2,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "metadata.jsonl not found at: \(metadataPath.path)"
                ]
            )
        }

        let metadataContent = try String(contentsOf: metadataPath, encoding: .utf8)
        var samples: [BenchmarkSample] = []

        for (index, line) in metadataContent.components(separatedBy: .newlines).enumerated() {
            guard !line.isEmpty else { continue }
            guard samples.count < maxSamples else { break }

            let decoder = JSONDecoder()
            guard let data = line.data(using: .utf8),
                let entry = try? decoder.decode(MetadataEntry.self, from: data)
            else {
                logger.warning("Failed to decode line \(index): \(line)")
                continue
            }

            let audioPath = baseDir.appendingPathComponent(entry.file_name).path
            guard FileManager.default.fileExists(atPath: audioPath) else {
                logger.warning("Audio file not found: \(audioPath)")
                continue
            }

            samples.append(
                BenchmarkSample(
                    audioPath: audioPath,
                    reference: entry.text,
                    sampleId: index
                ))
        }

        return samples
    }

    private static func downloadTHCHS30Dataset(to directory: URL) async throws {
        // Download using git-lfs or HuggingFace Hub API
        // For now, use a simple approach: shell out to huggingface-cli
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "huggingface-cli",
            "download",
            "FluidInference/THCHS-30-tests",
            "--repo-type", "dataset",
            "--local-dir", directory.path,
        ]

        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw NSError(
                domain: "CtcZhCnBenchmark",
                code: 3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        """
                    Failed to download THCHS-30 dataset from HuggingFace.
                    Make sure huggingface-cli is installed: pip install huggingface_hub
                    """
                ]
            )
        }
    }

    private static func runBenchmark(
        manager: CtcZhCnManager, samples: [BenchmarkSample]
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        for (index, sample) in samples.enumerated() {
            let audioURL = URL(fileURLWithPath: sample.audioPath)

            let startTime = Date()
            let hypothesis = try await manager.transcribe(audioURL: audioURL)
            let elapsed = Date().timeIntervalSince(startTime)

            let normalizedRef = normalizeChineseText(sample.reference)
            let normalizedHyp = normalizeChineseText(hypothesis)

            let cer = calculateCER(reference: normalizedRef, hypothesis: normalizedHyp)

            // Get audio duration
            let audioFile = try AVAudioFile(forReading: audioURL)
            let duration = Double(audioFile.length) / audioFile.processingFormat.sampleRate

            let rtfx = duration / elapsed

            let result = BenchmarkResult(
                sampleId: sample.sampleId,
                reference: sample.reference,
                hypothesis: hypothesis,
                normalizedRef: normalizedRef,
                normalizedHyp: normalizedHyp,
                cer: cer,
                latencyMs: elapsed * 1000.0,
                audioDurationSec: duration,
                rtfx: rtfx
            )

            results.append(result)

            if (index + 1) % 10 == 0 {
                logger.info("Processed \(index + 1)/\(samples.count) samples...")
            }
        }

        return results
    }

    private static func normalizeChineseText(_ text: String) -> String {
        var normalized = text

        // Remove Chinese punctuation (including curly quotes U+201C, U+201D, U+2018, U+2019)
        let chinesePunct = "，。！？、；：\u{201C}\u{201D}\u{2018}\u{2019}"
        for char in chinesePunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove Chinese brackets and quotes
        let brackets = "「」『』（）《》【】"
        for char in brackets {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove common symbols
        let symbols = "…—·"
        for char in symbols {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove English punctuation
        let englishPunct = ",.!?;:()[]{}\\<>\"'-"
        for char in englishPunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Convert Arabic digits to Chinese characters
        let digitMap: [Character: String] = [
            "0": "零",
            "1": "一",
            "2": "二",
            "3": "三",
            "4": "四",
            "5": "五",
            "6": "六",
            "7": "七",
            "8": "八",
            "9": "九",
        ]
        for (digit, chinese) in digitMap {
            normalized = normalized.replacingOccurrences(of: String(digit), with: chinese)
        }

        // Normalize whitespace and remove spaces
        normalized = normalized.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined()

        return normalized.lowercased()
    }

    private static func calculateCER(reference: String, hypothesis: String) -> Double {
        let refChars = Array(reference)
        let hypChars = Array(hypothesis)

        // Levenshtein distance
        let distance = levenshteinDistance(refChars, hypChars)

        guard !refChars.isEmpty else { return hypChars.isEmpty ? 0.0 : 1.0 }

        return Double(distance) / Double(refChars.count)
    }

    private static func levenshteinDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        let m = a.count
        let n = b.count

        var dp = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)

        for i in 0...m {
            dp[i][0] = i
        }
        for j in 0...n {
            dp[0][j] = j
        }

        // Skip main loop if either array is empty (ranges 1...0 would be invalid)
        guard m > 0 && n > 0 else { return dp[m][n] }

        for i in 1...m {
            for j in 1...n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                } else {
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                }
            }
        }

        return dp[m][n]
    }

    private static func printResults(results: [BenchmarkResult], encoderType: String) {
        guard !results.isEmpty else {
            logger.info("No results to display")
            return
        }

        let cers = results.map { $0.cer }
        let latencies = results.map { $0.latencyMs }
        let rtfxs = results.map { $0.rtfx }

        let meanCER = cers.reduce(0, +) / Double(cers.count) * 100.0
        let medianCER = median(cers) * 100.0
        let meanLatency = latencies.reduce(0, +) / Double(latencies.count)
        let meanRTFx = rtfxs.reduce(0, +) / Double(rtfxs.count)

        logger.info("")
        logger.info("=== Benchmark Results ===")
        logger.info("Encoder: \(encoderType)")
        logger.info("Samples: \(results.count)")
        logger.info("")
        logger.info("Mean CER: \(String(format: "%.2f", meanCER))%")
        logger.info("Median CER: \(String(format: "%.2f", medianCER))%")
        logger.info("Mean Latency: \(String(format: "%.1f", meanLatency))ms")
        logger.info("Mean RTFx: \(String(format: "%.1f", meanRTFx))x")

        // CER distribution
        let below5 = cers.filter { $0 < 0.05 }.count
        let below10 = cers.filter { $0 < 0.10 }.count
        let below20 = cers.filter { $0 < 0.20 }.count

        logger.info("")
        logger.info("CER Distribution:")
        logger.info(
            "  <5%: \(below5) samples (\(String(format: "%.1f", Double(below5) / Double(results.count) * 100.0))%)")
        logger.info(
            "  <10%: \(below10) samples (\(String(format: "%.1f", Double(below10) / Double(results.count) * 100.0))%)")
        logger.info(
            "  <20%: \(below20) samples (\(String(format: "%.1f", Double(below20) / Double(results.count) * 100.0))%)")
    }

    private static func median(_ values: [Double]) -> Double {
        let sorted = values.sorted()
        let count = sorted.count
        if count == 0 { return 0.0 }
        if count % 2 == 0 {
            return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            return sorted[count / 2]
        }
    }

    private struct BenchmarkOutput: Codable {
        let summary: Summary
        let results: [BenchmarkResult]

        struct Summary: Codable {
            let mean_cer: Double
            let median_cer: Double
            let mean_latency_ms: Double
            let mean_rtfx: Double
            let total_samples: Int
            let below_5_pct: Int
            let below_10_pct: Int
            let below_20_pct: Int
        }
    }

    private static func saveResults(results: [BenchmarkResult], outputFile: String) throws {
        guard !results.isEmpty else {
            logger.warning("No results to save")
            return
        }

        let cers = results.map { $0.cer }
        let latencies = results.map { $0.latencyMs }
        let rtfxs = results.map { $0.rtfx }

        let summary = BenchmarkOutput.Summary(
            mean_cer: cers.reduce(0, +) / Double(cers.count),
            median_cer: median(cers),
            mean_latency_ms: latencies.reduce(0, +) / Double(latencies.count),
            mean_rtfx: rtfxs.reduce(0, +) / Double(rtfxs.count),
            total_samples: results.count,
            below_5_pct: cers.filter { $0 < 0.05 }.count,
            below_10_pct: cers.filter { $0 < 0.10 }.count,
            below_20_pct: cers.filter { $0 < 0.20 }.count
        )

        let output = BenchmarkOutput(summary: summary, results: results)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let jsonData = try encoder.encode(output)
        try jsonData.write(to: URL(fileURLWithPath: outputFile))
    }

    private static func createProgressHandler() -> DownloadUtils.ProgressHandler {
        return { progress in
            let percentage = progress.fractionCompleted * 100.0
            switch progress.phase {
            case .listing:
                logger.info("Listing files from repository...")
            case .downloading(let completed, let total):
                logger.info(
                    "Downloading models: \(completed)/\(total) files (\(String(format: "%.1f", percentage))%)"
                )
            case .compiling(let modelName):
                logger.info("Compiling \(modelName)...")
            }
        }
    }

    private static func printUsage() {
        logger.info(
            """
            CTC zh-CN Benchmark - Measure Character Error Rate on THCHS-30 dataset

            Usage: fluidaudiocli ctc-zh-cn-benchmark [options]

            Options:
                --samples, -n <num>      Number of samples to test (default: 100)
                --int8                   Use int8 quantized encoder (default)
                --fp32                   Use fp32 encoder
                --output, -o <file>      Save results to JSON file
                --dataset-path <path>    Path to THCHS-30 dataset directory
                --auto-download          Download THCHS-30 from HuggingFace (requires huggingface-cli)
                --verbose, -v            Show download progress
                --help, -h               Show this help message

            Examples:
                # Auto-download from HuggingFace
                fluidaudiocli ctc-zh-cn-benchmark --auto-download --samples 100

                # Use local dataset
                fluidaudiocli ctc-zh-cn-benchmark --dataset-path ./thchs30_test_hf

                # Save results to JSON
                fluidaudiocli ctc-zh-cn-benchmark --auto-download --output results.json

            Expected Results (THCHS-30, 100 samples):
                Int8 encoder: 8.37% mean CER, 6.67% median CER
                FP32 encoder: Similar performance

            Dataset: FluidInference/THCHS-30-tests on HuggingFace
                     2,495 Mandarin Chinese test utterances from THCHS-30 corpus
            """
        )
    }
}

#endif
