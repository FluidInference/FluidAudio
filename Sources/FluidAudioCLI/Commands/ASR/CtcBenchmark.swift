#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation
import OSLog

/// CTC Keyword Boosting Benchmark
public class CtcBenchmark {

    private let logger = AppLogger(category: "CtcBenchmark")

    public init() {}

    /// Calculate Word Error Rate between reference and hypothesis
    private func calculateWER(reference: String, hypothesis: String) -> Double {
        let refWords = normalizeText(reference)
        let hypWords = normalizeText(hypothesis)

        guard !refWords.isEmpty else { return hypWords.isEmpty ? 0.0 : 100.0 }

        // Levenshtein distance for words
        var d = Array(repeating: Array(repeating: 0, count: hypWords.count + 1), count: refWords.count + 1)

        for i in 0...refWords.count {
            d[i][0] = i
        }
        for j in 0...hypWords.count {
            d[0][j] = j
        }

        for i in 1...refWords.count {
            for j in 1...hypWords.count {
                if refWords[i - 1] == hypWords[j - 1] {
                    d[i][j] = d[i - 1][j - 1]
                } else {
                    d[i][j] = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1]) + 1
                }
            }
        }

        let distance = d[refWords.count][hypWords.count]
        return Double(distance) / Double(refWords.count) * 100.0
    }

    /// Normalize text for WER calculation
    private func normalizeText(_ text: String) -> [String] {
        let normalized = text.lowercased()
            .replacingOccurrences(of: "[^a-z0-9\\s]", with: "", options: .regularExpression)
        return normalized.split(whereSeparator: { $0.isWhitespace }).map { String($0) }
    }

    /// Run benchmark on a dataset
    public func runBenchmark(
        datasetPath: String,
        customVocabPath: String?,
        maxFiles: Int?,
        startIndex: Int = 0,
        outputPath: String?
    ) async throws {
        logger.info("=== CTC Keyword Boosting Benchmark ===")
        logger.info("Dataset: \(datasetPath)")
        if let vocabPath = customVocabPath {
            logger.info("Custom vocabulary: \(vocabPath)")
        }

        // Load metadata
        logger.info("Loading metadata...")
        let metadataURL = URL(fileURLWithPath: datasetPath).appendingPathComponent("metadata.json")
        guard let metadataData = try? Data(contentsOf: metadataURL) else {
            logger.error("Failed to load metadata.json from \(metadataURL.path)")
            throw NSError(
                domain: "CtcBenchmark", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load metadata.json"])
        }

        logger.info("Decoding metadata...")
        let decoder = JSONDecoder()
        let metadata = try decoder.decode([AudioMetadata].self, from: metadataData)
        logger.info("Loaded \(metadata.count) items from metadata")

        let clampedStart = min(max(0, startIndex), metadata.count)
        let slicedMetadata = Array(metadata.dropFirst(clampedStart))
        let samplesToProcess = maxFiles.map { min($0, slicedMetadata.count) } ?? slicedMetadata.count
        logger.info("Processing \(samplesToProcess) samples starting at index \(clampedStart)")

        // Load custom vocabulary if provided
        // Supports both JSON format (.json) and simple text format (.txt)
        var customVocab: CustomVocabularyContext?
        if let vocabPath = customVocabPath {
            let vocabURL = URL(fileURLWithPath: vocabPath)
            let isJson = vocabURL.pathExtension.lowercased() == "json"
            if isJson {
                customVocab = try CustomVocabularyContext.loadWithSentencePieceTokenization(from: vocabURL)
            } else {
                customVocab = try CustomVocabularyContext.loadFromSimpleFormatWithTokenization(from: vocabURL)
            }
            logger.info("Loaded vocabulary with \(customVocab?.terms.count ?? 0) terms (tokenized for CTC)")
        }

        // Initialize ASR manager
        logger.info("Initializing ASR system...")
        let asrConfig = ASRConfig(tdtConfig: TdtConfig())
        let asrManager = AsrManager(config: asrConfig)

        do {
            let models = try await AsrModels.downloadAndLoad(version: .v3)
            try await asrManager.initialize(models: models)
            logger.info("ASR system initialized successfully")
        } catch {
            logger.error("Failed to initialize ASR system: \(error.localizedDescription)")
            throw error
        }

        var results: [BenchmarkResult] = []
        var totalBaselineWER = 0.0
        var totalCtcWER = 0.0

        for (index, item) in slicedMetadata.prefix(samplesToProcess).enumerated() {
            let audioURL = URL(fileURLWithPath: datasetPath).appendingPathComponent(item.audioFile)
            let audioPath = audioURL.path

            guard FileManager.default.fileExists(atPath: audioPath) else {
                logger.error("Missing audio file at \(audioPath)")
                throw NSError(
                    domain: "CtcBenchmark",
                    code: 2,
                    userInfo: [NSLocalizedDescriptionKey: "Missing audio file at \(audioPath)"])
            }
            do {
                let attributes = try FileManager.default.attributesOfItem(atPath: audioPath)
                let size = (attributes[.size] as? NSNumber)?.intValue ?? -1
                if size <= 0 {
                    logger.error("Audio file is empty at \(audioPath)")
                    throw NSError(
                        domain: "CtcBenchmark",
                        code: 3,
                        userInfo: [NSLocalizedDescriptionKey: "Audio file is empty at \(audioPath)"])
                }
            } catch {
                logger.error("Unable to stat audio file at \(audioPath): \(error.localizedDescription)")
                throw error
            }

            logger.info("[\(index+1)/\(samplesToProcess)] Processing \(item.audioFile)...")

            do {
                // Transcribe without CTC boosting (baseline)
                let baselineResult = try await transcribeFile(audioPath, asrManager: asrManager, customVocab: nil)

                // Transcribe with CTC boosting
                let ctcResult: String
                if let vocab = customVocab {
                    ctcResult = try await transcribeFile(audioPath, asrManager: asrManager, customVocab: vocab)
                } else {
                    ctcResult = baselineResult
                }

                // Calculate WER
                let baselineWER = calculateWER(reference: item.transcript, hypothesis: baselineResult)
                let ctcWER = calculateWER(reference: item.transcript, hypothesis: ctcResult)

                totalBaselineWER += baselineWER
                totalCtcWER += ctcWER

                let result = BenchmarkResult(
                    audioFile: item.audioFile,
                    reference: item.transcript,
                    baseline: baselineResult,
                    withCtc: ctcResult,
                    baselineWER: baselineWER,
                    ctcWER: ctcWER,
                    improvement: baselineWER - ctcWER
                )
                results.append(result)

                logger.info("  Baseline WER: \(String(format: "%.2f", baselineWER))%")
                logger.info("  CTC WER:      \(String(format: "%.2f", ctcWER))%")
                logger.info("  Improvement:  \(String(format: "%.2f", baselineWER - ctcWER))%")
            } catch {
                logger.error("Failed on \(audioPath): \(error.localizedDescription)")
                throw error
            }
        }

        // Calculate summary statistics
        let avgBaselineWER = totalBaselineWER / Double(samplesToProcess)
        let avgCtcWER = totalCtcWER / Double(samplesToProcess)
        let relativeImprovement = ((avgBaselineWER - avgCtcWER) / avgBaselineWER) * 100.0

        logger.info("")
        logger.info("=== Summary ===")
        logger.info("Samples processed: \(samplesToProcess)")
        logger.info("Average Baseline WER: \(String(format: "%.2f", avgBaselineWER))%")
        logger.info("Average CTC WER:      \(String(format: "%.2f", avgCtcWER))%")
        logger.info("Relative WER Reduction: \(String(format: "%.2f", relativeImprovement))%")

        // Save results if output path provided
        if let outputPath = outputPath {
            let summary = BenchmarkSummary(
                dataset: datasetPath,
                samplesProcessed: samplesToProcess,
                vocabularyTerms: customVocab?.terms.count ?? 0,
                avgBaselineWER: avgBaselineWER,
                avgCtcWER: avgCtcWER,
                relativeImprovement: relativeImprovement,
                results: results
            )

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let jsonData = try encoder.encode(summary)
            try jsonData.write(to: URL(fileURLWithPath: outputPath))
            logger.info("Results saved to: \(outputPath)")
        }
    }

    /// Transcribe a single audio file
    private func transcribeFile(
        _ audioPath: String,
        asrManager: AsrManager,
        customVocab: CustomVocabularyContext?
    ) async throws -> String {
        let audioURL = URL(fileURLWithPath: audioPath)

        // Use AsrManager's built-in file transcription
        let result = try await asrManager.transcribe(
            audioURL,
            customVocabulary: customVocab
        )

        return result.text
    }

    // MARK: - Data Models

    struct AudioMetadata: Codable {
        let id: Int
        let audioFile: String
        let transcript: String

        enum CodingKeys: String, CodingKey {
            case id
            case audioFileSnake = "audio_file"
            case audioFileCamel = "audioFile"
            case transcript
            case sourceId  // ignored
        }

        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(Int.self, forKey: .id)
            transcript = try container.decode(String.self, forKey: .transcript)

            if let file = try? container.decode(String.self, forKey: .audioFileSnake) {
                audioFile = file
            } else if let file = try? container.decode(String.self, forKey: .audioFileCamel) {
                audioFile = file
            } else if let file = try? container.decode(String.self, forKey: .sourceId) {
                audioFile = file
            } else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: container.codingPath,
                        debugDescription: "Missing audio file field"))
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .id)
            try container.encode(audioFile, forKey: .audioFileSnake)
            try container.encode(transcript, forKey: .transcript)
        }
    }

    struct BenchmarkResult: Codable {
        let audioFile: String
        let reference: String
        let baseline: String
        let withCtc: String
        let baselineWER: Double
        let ctcWER: Double
        let improvement: Double
    }

    struct BenchmarkSummary: Codable {
        let dataset: String
        let samplesProcessed: Int
        let vocabularyTerms: Int
        let avgBaselineWER: Double
        let avgCtcWER: Double
        let relativeImprovement: Double
        let results: [BenchmarkResult]
    }

    // MARK: - CLI Entry Point

    public static func runCLI(arguments: [String]) async {
        let logger = AppLogger(category: "CtcBenchmark")
        logger.info("CTC Benchmark CLI started")

        var datasetPath: String?
        var customVocabPath: String?
        var maxFiles: Int?
        var startIndex: Int = 0
        var outputPath: String?

        // Check for help flag
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        logger.info("Parsing arguments...")

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    datasetPath = arguments[i + 1]
                    i += 1
                }
            case "--vocab", "--custom-vocab":
                if i + 1 < arguments.count {
                    customVocabPath = arguments[i + 1]
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count {
                    maxFiles = Int(arguments[i + 1])
                    i += 1
                }
            case "--start-index":
                if i + 1 < arguments.count {
                    startIndex = max(0, Int(arguments[i + 1]) ?? 0)
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputPath = arguments[i + 1]
                    i += 1
                }
            default:
                logger.warning("Unknown argument: \(arguments[i])")
            }
            i += 1
        }

        guard let dataset = datasetPath else {
            logger.error("Error: --dataset is required")
            printUsage()
            exit(1)
        }

        let benchmark = CtcBenchmark()
        do {
            try await benchmark.runBenchmark(
                datasetPath: dataset,
                customVocabPath: customVocabPath,
                maxFiles: maxFiles,
                startIndex: startIndex,
                outputPath: outputPath
            )
        } catch {
            logger.error("Benchmark failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func printUsage() {
        print(
            """
            CTC Keyword Boosting Benchmark

            Usage: fluidaudio ctc-benchmark --dataset <path> [options]

            Required:
                --dataset <path>         Path to dataset directory (must contain metadata.json)

            Options:
                --vocab <path>           Path to custom vocabulary JSON file
                --custom-vocab <path>    Alias for --vocab
                --max-files <n>          Maximum number of files to process
                --start-index <n>        Skip the first n entries from metadata
                --output <path>          Output JSON file for results
                -h, --help               Show this help message

            Examples:
                # Run on Earnings22 with CTC boosting
                fluidaudio ctc-benchmark --dataset ~/Datasets/Earnings22 \\
                    --vocab custom_vocab.json --max-files 50 \\
                    --output earnings22_ctc_results.json

                # Run on Earnings22 without CTC (baseline only)
                fluidaudio ctc-benchmark --dataset ~/Datasets/Earnings22 \\
                    --max-files 50 --output earnings22_baseline.json

                # Slice Earnings22 into 25-file shards with per-slice vocab
                fluidaudio ctc-benchmark --dataset ~/Datasets/Earnings22 \\
                    --start-index 0 --max-files 25 \\
                    --vocab Datasets/Earnings22/custom_vocabulary_000_024.json \\
                    --output earnings22_ctc_000_024.json

                # Next slice (files 25-49)
                fluidaudio ctc-benchmark --dataset ~/Datasets/Earnings22 \\
                    --start-index 25 --max-files 25 \\
                    --vocab Datasets/Earnings22/custom_vocabulary_025_049.json \\
                    --output earnings22_ctc_025_049.json
            """)
    }
}
#endif
