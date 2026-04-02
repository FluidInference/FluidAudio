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

            // Load FLEURS dataset
            logger.info("")
            logger.info("Loading FLEURS Mandarin Chinese test set...")
            let samples = try await loadFleursSamples(maxSamples: numSamples)
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

    private static func loadFleursSamples(maxSamples: Int) async throws -> [BenchmarkSample] {
        // For now, we'll document that users need to download FLEURS manually
        // In a production system, this would use HuggingFace datasets API
        throw NSError(
            domain: "CtcZhCnBenchmark",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    """
                FLEURS dataset not yet auto-downloadable in FluidAudio.

                To run this benchmark:
                1. Download FLEURS manually from HuggingFace
                2. Or use the mobius benchmark: cd mobius/models/stt/parakeet-ctc-0.6b-zh-cn/coreml
                3. Run: uv run python benchmark-full-pipeline.py --num-samples \(maxSamples)

                Expected CER (from mobius benchmarks):
                - int8 encoder: 10.54% CER (100 samples)
                - fp32 encoder: 10.45% CER (100 samples)
                """
            ]
        )
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

        // Remove Chinese punctuation
        let chinesePunct = "，。！？、；："
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

        // Remove spaces
        normalized = normalized.replacingOccurrences(of: " ", with: "")

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

    private static func saveResults(results: [BenchmarkResult], outputFile: String) throws {
        let jsonData = try JSONEncoder().encode(results)
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
            CTC zh-CN Benchmark - Measure Character Error Rate on FLEURS dataset

            Usage: fluidaudiocli ctc-zh-cn-benchmark [options]

            Options:
                --samples, -n <num>   Number of samples to test (default: 100)
                --int8                Use int8 quantized encoder (default)
                --fp32                Use fp32 encoder
                --output, -o <file>   Save results to JSON file
                --verbose, -v         Show download progress
                --help, -h            Show this help message

            Examples:
                fluidaudiocli ctc-zh-cn-benchmark --samples 100
                fluidaudiocli ctc-zh-cn-benchmark --fp32 --output results.json

            Expected Results (from mobius benchmarks):
                Int8 encoder: 10.54% CER (100 samples)
                FP32 encoder: 10.45% CER (100 samples)

            Note: FLEURS dataset auto-download not yet implemented.
                  Use mobius benchmark for full CER evaluation.
            """
        )
    }
}

#endif
