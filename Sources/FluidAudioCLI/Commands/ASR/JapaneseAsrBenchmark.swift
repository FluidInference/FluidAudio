#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Japanese ASR Benchmark - CER evaluation on JSUT and Common Voice datasets
enum JapaneseAsrBenchmark {
    private static let logger = AppLogger(category: "JapaneseAsrBenchmark")

    enum Dataset: String, CaseIterable {
        case jsut = "jsut"
        case cvTrain = "cv-train"
        case cvValidation = "cv-validation"
        case cvTest = "cv-test"

        var displayName: String {
            switch self {
            case .jsut: return "JSUT-basic5000"
            case .cvTrain: return "Common Voice Japanese (Train)"
            case .cvValidation: return "Common Voice Japanese (Validation)"
            case .cvTest: return "Common Voice Japanese (Test)"
            }
        }

        var cvSplit: DatasetDownloader.CVSplit? {
            switch self {
            case .cvTrain: return .train
            case .cvValidation: return .validation
            case .cvTest: return .test
            default: return nil
            }
        }
    }

    static func run(arguments: [String]) async {
        var dataset: Dataset = .jsut
        var numSamples = 100
        var outputFile: String?
        var verbose = false
        var autoDownload = false

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--dataset", "-d":
                if i + 1 < arguments.count {
                    if let ds = Dataset(rawValue: arguments[i + 1]) {
                        dataset = ds
                    } else {
                        logger.error("Unknown dataset: \(arguments[i + 1])")
                        logger.info("Available: \(Dataset.allCases.map { $0.rawValue }.joined(separator: ", "))")
                        return
                    }
                    i += 1
                }
            case "--samples", "-n":
                if i + 1 < arguments.count {
                    numSamples = Int(arguments[i + 1]) ?? 100
                    i += 1
                }
            case "--output", "-o":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
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

        logger.info("=== Japanese ASR Benchmark ===")
        logger.info("Dataset: \(dataset.displayName)")
        logger.info("Samples: \(numSamples)")
        logger.info("")

        do {
            // Load models
            logger.info("Loading CTC Japanese models...")
            let manager = try await CtcJaManager.load(
                progressHandler: verbose ? createProgressHandler() : nil
            )
            logger.info("Models loaded successfully")

            // Load dataset
            logger.info("")
            logger.info("Loading \(dataset.displayName)...")

            let samples: [JapaneseBenchmarkSample]
            do {
                samples = try await loadSamples(for: dataset, maxSamples: numSamples)
            } catch JapaneseDatasetError.datasetNotFound(let message) {
                if autoDownload {
                    logger.info("Dataset not found, auto-downloading...")
                    await downloadDataset(dataset, maxSamples: numSamples)
                    samples = try await loadSamples(for: dataset, maxSamples: numSamples)
                } else {
                    logger.error(message)
                    logger.info("Use --auto-download to download automatically")
                    return
                }
            }

            guard !samples.isEmpty else {
                logger.error("No samples loaded. Check dataset installation.")
                return
            }

            logger.info("Loaded \(samples.count) samples")

            // Run benchmark
            logger.info("")
            logger.info("Running transcription benchmark...")
            let results = try await runBenchmark(manager: manager, samples: samples)

            // Print results
            printResults(results: results, dataset: dataset)

            // Save to JSON if requested
            if let outputFile = outputFile {
                try saveResults(results: results, outputFile: outputFile, dataset: dataset)
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

    private static func loadSamples(
        for dataset: Dataset,
        maxSamples: Int
    ) async throws -> [JapaneseBenchmarkSample] {
        switch dataset {
        case .jsut:
            return try await JapaneseDatasetLoader.loadJSUTSamples(maxSamples: maxSamples)
        case .cvTrain, .cvValidation, .cvTest:
            guard let split = dataset.cvSplit else {
                return []
            }
            return try await JapaneseDatasetLoader.loadCommonVoiceSamples(
                split: split, maxSamples: maxSamples)
        }
    }

    private static func downloadDataset(_ dataset: Dataset, maxSamples: Int?) async {
        switch dataset {
        case .jsut:
            await DatasetDownloader.downloadJSUTBasic5000(force: false, maxSamples: maxSamples)
        case .cvTrain, .cvValidation, .cvTest:
            guard let split = dataset.cvSplit else { return }
            await DatasetDownloader.downloadCommonVoiceJapanese(
                force: false, maxSamples: maxSamples, split: split)
        }
    }

    // MARK: - Benchmark Result Types

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

    private struct BenchmarkOutput: Codable {
        let summary: Summary
        let results: [BenchmarkResult]

        struct Summary: Codable {
            let dataset: String
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

    // MARK: - Benchmark Execution

    private static func runBenchmark(
        manager: CtcJaManager,
        samples: [JapaneseBenchmarkSample]
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        for (index, sample) in samples.enumerated() {
            let startTime = Date()
            let hypothesis = try await manager.transcribe(audioURL: sample.audioPath)
            let elapsed = Date().timeIntervalSince(startTime)

            let normalizedRef = normalizeJapaneseText(sample.transcript)
            let normalizedHyp = normalizeJapaneseText(hypothesis)

            let cer = calculateCER(reference: normalizedRef, hypothesis: normalizedHyp)

            // Get audio duration
            let audioFile = try AVAudioFile(forReading: sample.audioPath)
            let duration = Double(audioFile.length) / audioFile.processingFormat.sampleRate

            let rtfx = duration / elapsed

            let result = BenchmarkResult(
                sampleId: sample.sampleId,
                reference: sample.transcript,
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

    // MARK: - Japanese Text Normalization

    /// Normalize Japanese text for fair CER calculation
    private static func normalizeJapaneseText(_ text: String) -> String {
        var normalized = text

        // Remove Japanese punctuation
        let japanesePunct = "、。！？・…「」『』（）［］｛｝【】"
        for char in japanesePunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Remove ASCII punctuation
        let asciiPunct = ",.!?;:\'\"()-[]{}"
        for char in asciiPunct {
            normalized = normalized.replacingOccurrences(of: String(char), with: "")
        }

        // Normalize whitespace
        normalized = normalized.components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }
            .joined()

        // Convert to lowercase for any romaji
        normalized = normalized.lowercased()

        return normalized
    }

    // MARK: - CER Calculation

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

    // MARK: - Results Output

    private static func printResults(results: [BenchmarkResult], dataset: Dataset) {
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
        logger.info("Dataset: \(dataset.displayName)")
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

    private static func saveResults(
        results: [BenchmarkResult],
        outputFile: String,
        dataset: Dataset
    ) throws {
        guard !results.isEmpty else {
            logger.warning("No results to save")
            return
        }

        let cers = results.map { $0.cer }
        let latencies = results.map { $0.latencyMs }
        let rtfxs = results.map { $0.rtfx }

        let below5 = cers.filter { $0 < 0.05 }.count
        let below10 = cers.filter { $0 < 0.10 }.count
        let below20 = cers.filter { $0 < 0.20 }.count

        let summary = BenchmarkOutput.Summary(
            dataset: dataset.rawValue,
            mean_cer: cers.reduce(0, +) / Double(cers.count),
            median_cer: median(cers),
            mean_latency_ms: latencies.reduce(0, +) / Double(latencies.count),
            mean_rtfx: rtfxs.reduce(0, +) / Double(rtfxs.count),
            total_samples: results.count,
            below_5_pct: below5,
            below_10_pct: below10,
            below_20_pct: below20
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
            Japanese ASR Benchmark - Measure Character Error Rate on Japanese datasets

            Usage: fluidaudiocli ja-benchmark [options]

            Options:
                --dataset, -d <name>    Dataset to use (default: jsut)
                                        Available: jsut, cv-train, cv-validation, cv-test
                --samples, -n <num>     Number of samples to test (default: 100)
                --output, -o <file>     Save results to JSON file
                --auto-download         Download dataset if not found
                --verbose, -v           Show download progress
                --help, -h              Show this help message

            Examples:
                # Benchmark on JSUT-basic5000
                fluidaudiocli ja-benchmark --dataset jsut --samples 100

                # Benchmark on Common Voice Japanese test set
                fluidaudiocli ja-benchmark --dataset cv-test --samples 500

                # Save results to JSON
                fluidaudiocli ja-benchmark --dataset jsut --output results.json

            Datasets:
                jsut            JSUT-basic5000 (5,000 utterances, single speaker)
                cv-train        Common Voice Japanese train split (~400k utterances)
                cv-validation   Common Voice Japanese validation split (~9k utterances)
                cv-test         Common Voice Japanese test split (~9k utterances)
            """
        )
    }
}

#endif
