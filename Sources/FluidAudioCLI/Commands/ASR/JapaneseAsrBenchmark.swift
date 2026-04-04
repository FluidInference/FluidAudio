#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Japanese ASR Benchmark - CER evaluation on JSUT and Common Voice datasets
enum JapaneseAsrBenchmark {
    private static let logger = AppLogger(category: "JapaneseAsrBenchmark")

    enum Dataset: String, CaseIterable {
        case jsut = "jsut"
        case cvTest = "cv-test"

        var displayName: String {
            switch self {
            case .jsut: return "JSUT-basic5000"
            case .cvTest: return "Common Voice Japanese (Test)"
            }
        }

        var cvSplit: DatasetDownloader.CVSplit? {
            switch self {
            case .cvTest: return .test
            default: return nil
            }
        }
    }

    enum DecoderType: String {
        case ctc
        case tdt
    }

    static func run(arguments: [String]) async {
        var dataset: Dataset = .jsut
        var numSamples = 100
        var outputFile: String?
        var verbose = false
        var autoDownload = false
        var decoder: DecoderType = .ctc

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--decoder":
                if i + 1 < arguments.count {
                    if let decoderType = DecoderType(rawValue: arguments[i + 1]) {
                        decoder = decoderType
                    } else {
                        logger.error("Unknown decoder: \(arguments[i + 1])")
                        logger.info("Available: ctc, tdt")
                        return
                    }
                    i += 1
                }
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
        logger.info("Decoder: \(decoder.rawValue.uppercased())")
        logger.info("Samples: \(numSamples)")
        logger.info("")

        do {
            // Load dataset
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
            logger.info("")

            // Run benchmark with selected decoder
            let results: [BenchmarkResult]
            switch decoder {
            case .ctc:
                logger.info("Loading CTC Japanese models...")
                let ctcManager = try await CtcJaManager.load(
                    progressHandler: verbose ? createProgressHandler() : nil
                )
                logger.info("Models loaded successfully")
                logger.info("")
                logger.info("Running transcription benchmark...")
                results = try await runBenchmark(samples: samples) { audioURL in
                    try await ctcManager.transcribe(audioURL: audioURL)
                }

            case .tdt:
                logger.info("Loading TDT Japanese models...")
                let tdtManager = try await TdtJaManager.load(
                    progressHandler: verbose ? createProgressHandler() : nil
                )
                logger.info("Models loaded successfully")
                logger.info("")
                logger.info("Running transcription benchmark...")
                results = try await runBenchmark(samples: samples) { audioURL in
                    try await tdtManager.transcribe(audioURL: audioURL)
                }
            }

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
        case .cvTest:
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
        case .cvTest:
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
        samples: [JapaneseBenchmarkSample],
        transcribe: (URL) async throws -> String
    ) async throws -> [BenchmarkResult] {
        var results: [BenchmarkResult] = []

        for (index, sample) in samples.enumerated() {
            let startTime = Date()
            let hypothesis = try await transcribe(sample.audioPath)
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

        // Normalize numbers first (before removing punctuation)
        normalized = normalizeJapaneseNumbers(normalized)

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

    /// Normalize number formats (full-width digits, kanji numbers) to half-width Arabic digits
    /// This matches NVIDIA's evaluation methodology which normalizes numbers
    private static func normalizeJapaneseNumbers(_ text: String) -> String {
        var result = text

        // Convert full-width digits to half-width (０-９ → 0-9)
        let fullWidthDigits = "０１２３４５６７８９"
        let halfWidthDigits = "0123456789"
        for (fullWidth, halfWidth) in zip(fullWidthDigits, halfWidthDigits) {
            result = result.replacingOccurrences(of: String(fullWidth), with: String(halfWidth))
        }

        // Convert kanji numbers to Arabic digits
        // IMPORTANT: Order matters! Compound numbers must be matched before their components.
        // e.g., "二十一" (21) must match before "二十" (20) or "十一" (11) to avoid producing "211" or "2011"
        let kanjiToArabic: [(String, String)] = [
            // Compound tens (21-99) - MUST come first to prevent incorrect sub-pattern matching
            ("二十一", "21"), ("二十二", "22"), ("二十三", "23"), ("二十四", "24"),
            ("二十五", "25"), ("二十六", "26"), ("二十七", "27"), ("二十八", "28"), ("二十九", "29"),
            ("三十一", "31"), ("三十二", "32"), ("三十三", "33"), ("三十四", "34"),
            ("三十五", "35"), ("三十六", "36"), ("三十七", "37"), ("三十八", "38"), ("三十九", "39"),
            ("四十一", "41"), ("四十二", "42"), ("四十三", "43"), ("四十四", "44"),
            ("四十五", "45"), ("四十六", "46"), ("四十七", "47"), ("四十八", "48"), ("四十九", "49"),
            ("五十一", "51"), ("五十二", "52"), ("五十三", "53"), ("五十四", "54"),
            ("五十五", "55"), ("五十六", "56"), ("五十七", "57"), ("五十八", "58"), ("五十九", "59"),
            ("六十一", "61"), ("六十二", "62"), ("六十三", "63"), ("六十四", "64"),
            ("六十五", "65"), ("六十六", "66"), ("六十七", "67"), ("六十八", "68"), ("六十九", "69"),
            ("七十一", "71"), ("七十二", "72"), ("七十三", "73"), ("七十四", "74"),
            ("七十五", "75"), ("七十六", "76"), ("七十七", "77"), ("七十八", "78"), ("七十九", "79"),
            ("八十一", "81"), ("八十二", "82"), ("八十三", "83"), ("八十四", "84"),
            ("八十五", "85"), ("八十六", "86"), ("八十七", "87"), ("八十八", "88"), ("八十九", "89"),
            ("九十一", "91"), ("九十二", "92"), ("九十三", "93"), ("九十四", "94"),
            ("九十五", "95"), ("九十六", "96"), ("九十七", "97"), ("九十八", "98"), ("九十九", "99"),

            // Full-width digit compounds (一〇 = 10, 二〇 = 20, etc.)
            ("一〇", "10"), ("二〇", "20"), ("三〇", "30"), ("四〇", "40"),
            ("五〇", "50"), ("六〇", "60"), ("七〇", "70"), ("八〇", "80"), ("九〇", "90"),

            // 10-19 range
            ("十一", "11"), ("十二", "12"), ("十三", "13"), ("十四", "14"),
            ("十五", "15"), ("十六", "16"), ("十七", "17"), ("十八", "18"), ("十九", "19"),

            // Simple tens (20, 30, ..., 90)
            ("二十", "20"), ("三十", "30"), ("四十", "40"), ("五十", "50"),
            ("六十", "60"), ("七十", "70"), ("八十", "80"), ("九十", "90"),

            // Larger units
            ("百", "100"), ("千", "1000"), ("万", "10000"),

            // Ten (must come after all compound tens)
            ("十", "10"),

            // Basic single digits (must be last)
            ("一", "1"), ("二", "2"), ("三", "3"), ("四", "4"),
            ("五", "5"), ("六", "6"), ("七", "7"), ("八", "8"), ("九", "9"),

            // Zero
            ("〇", "0"), ("零", "0"),
        ]

        for (kanji, arabic) in kanjiToArabic {
            result = result.replacingOccurrences(of: kanji, with: arabic)
        }

        return result
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
                --decoder <type>        Decoder type: ctc or tdt (default: ctc)
                --dataset, -d <name>    Dataset to use (default: jsut)
                                        Available: jsut, cv-test
                --samples, -n <num>     Number of samples to test (default: 100)
                --output, -o <file>     Save results to JSON file
                --auto-download         Download dataset if not found
                --verbose, -v           Show download progress
                --help, -h              Show this help message

            Examples:
                # Benchmark CTC decoder on JSUT-basic5000
                fluidaudiocli ja-benchmark --dataset jsut --samples 100

                # Benchmark TDT decoder on Common Voice Japanese test set
                fluidaudiocli ja-benchmark --decoder tdt --dataset cv-test --samples 500

                # Save results to JSON
                fluidaudiocli ja-benchmark --dataset jsut --output results.json

            Datasets:
                jsut            JSUT-basic5000 (5,000 utterances, single speaker)
                cv-test         Common Voice Japanese test split (~9k utterances)
            """
        )
    }
}

#endif
