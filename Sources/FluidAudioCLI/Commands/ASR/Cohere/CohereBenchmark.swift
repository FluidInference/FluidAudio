#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Benchmark for Cohere Transcribe supporting LibriSpeech (English) and FLEURS (multilingual).
enum CohereBenchmark {
    private static let logger = AppLogger(category: "CohereBenchmark")

    /// Map FLEURS language codes to Cohere supported languages (14 languages).
    private static nonisolated(unsafe) let fleursToCohereLanguage: [String: CohereAsrConfig.Language] = [
        "en_us": .english,
        "fr_fr": .french,
        "de_de": .german,
        "es_419": .spanish,
        "it_it": .italian,
        "pt_br": .portuguese,
        "nl_nl": .dutch,
        "pl_pl": .polish,
        "el_gr": .greek,
        "ar_eg": .arabic,
        "ja_jp": .japanese,
        "cmn_hans_cn": .chinese,
        "ko_kr": .korean,
        "vi_vn": .vietnamese,
    ]

    static func run(arguments: [String]) async {
        var dataset = "librispeech"
        var subset = "test-clean"
        var maxFiles: Int? = nil
        var modelDir: String? = nil
        var outputFile = "cohere_benchmark_results.json"
        var languages: [String] = ["en_us"]
        var fleursDir: String? = nil
        var variant: CohereAsrVariant = .int8
        var maxTokens = 200

        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1]
                    i += 1
                }
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
            case "--languages":
                if i + 1 < arguments.count {
                    languages = arguments[i + 1].components(separatedBy: ",").map {
                        $0.trimmingCharacters(in: .whitespaces)
                    }
                    i += 1
                }
            case "--fleurs-dir":
                if i + 1 < arguments.count {
                    fleursDir = arguments[i + 1]
                    i += 1
                }
            case "--variant":
                if i + 1 < arguments.count {
                    let v = arguments[i + 1].lowercased()
                    if let parsed = CohereAsrVariant(rawValue: v) {
                        variant = parsed
                    } else {
                        logger.error("Unknown variant '\(arguments[i + 1])'. Use 'fp16' or 'int8'.")
                        exit(1)
                    }
                    i += 1
                }
            case "--max-tokens":
                if i + 1 < arguments.count, let tokens = Int(arguments[i + 1]) {
                    maxTokens = tokens
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        logger.info("Cohere Transcribe Benchmark (\(variant.rawValue))")
        logger.info("  Dataset: \(dataset)")
        if dataset == "librispeech" {
            logger.info("  Subset: \(subset)")
        } else {
            logger.info("  Languages: \(languages.joined(separator: ", "))")
        }
        logger.info("  Max files: \(maxFiles?.description ?? "all")")
        logger.info("  Model dir: \(modelDir ?? "auto-download")")
        logger.info("  Output: \(outputFile)")

        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Cohere Transcribe requires macOS 14 or later")
            exit(1)
        }

        do {
            // 1. Load Cohere Transcribe models
            let manager = CohereAsrManager()
            if let dir = modelDir {
                logger.info("Loading models from \(dir)")
                try await manager.loadModels(from: URL(fileURLWithPath: dir))
            } else {
                logger.info("Downloading Cohere Transcribe \(variant.rawValue) models...")
                let cacheDir = try await CohereAsrModels.download(variant: variant)
                try await manager.loadModels(from: cacheDir)
            }

            // 2. Run benchmark based on dataset
            switch dataset {
            case "fleurs":
                try await runFleursBenchmark(
                    manager: manager,
                    languages: languages,
                    maxFiles: maxFiles,
                    fleursDir: fleursDir,
                    outputFile: outputFile,
                    maxTokens: maxTokens
                )
            default:
                try await runLibriSpeechBenchmark(
                    manager: manager,
                    subset: subset,
                    maxFiles: maxFiles,
                    outputFile: outputFile,
                    maxTokens: maxTokens
                )
            }

        } catch {
            logger.error("Benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - LibriSpeech Benchmark

    @available(macOS 14, iOS 17, *)
    private static func runLibriSpeechBenchmark(
        manager: CohereAsrManager,
        subset: String,
        maxFiles: Int?,
        outputFile: String,
        maxTokens: Int
    ) async throws {
        let benchmark = ASRBenchmark()
        try await benchmark.downloadLibriSpeech(subset: subset)
        let datasetPath = benchmark.getLibriSpeechDirectory().appendingPathComponent(subset)
        let allFiles = try collectBenchmarkAudioFiles(from: datasetPath)
        let files = Array(allFiles.prefix(maxFiles ?? allFiles.count))
        logger.info("Collected \(files.count) files from LibriSpeech \(subset)")

        let results = try await runBenchmarkLoop(
            manager: manager,
            files: files,
            maxTokens: maxTokens
        )

        try saveCohereBenchmarkResults(results, to: outputFile)
        printSummary(results)
    }

    // MARK: - FLEURS Benchmark

    @available(macOS 14, iOS 17, *)
    private static func runFleursBenchmark(
        manager: CohereAsrManager,
        languages: [String],
        maxFiles: Int?,
        fleursDir: String?,
        outputFile: String,
        maxTokens: Int
    ) async throws {
        var allResults: [CohereBenchmarkResult] = []

        for langCode in languages {
            guard fleursToCohereLanguage.keys.contains(langCode) else {
                logger.warning("Unsupported language for Cohere: \(langCode)")
                continue
            }

            logger.info("Processing language: \(langCode)")

            // Get FLEURS files for this language
            let files = try collectFleursFiles(
                language: langCode,
                maxFiles: maxFiles,
                fleursDir: fleursDir
            )

            logger.info("  Collected \(files.count) files for \(langCode)")

            // Run benchmark for this language
            let langResults = try await runBenchmarkLoop(
                manager: manager,
                files: files,
                maxTokens: maxTokens
            )

            allResults.append(contentsOf: langResults)

            // Print language-specific summary
            let langWER = calculateWER(from: langResults)
            let langRTFx = langResults.map(\.rtfx).reduce(0, +) / Double(langResults.count)
            logger.info(
                "  \(langCode): WER = \(String(format: "%.2f", langWER))%, RTFx = \(String(format: "%.2f", langRTFx))x"
            )
        }

        try saveCohereBenchmarkResults(allResults, to: outputFile)
        printSummary(allResults)
    }

    // MARK: - Benchmark Loop

    @available(macOS 14, iOS 17, *)
    private static func runBenchmarkLoop(
        manager: CohereAsrManager,
        files: [BenchmarkAudioFile],
        maxTokens: Int
    ) async throws -> [CohereBenchmarkResult] {
        var results: [CohereBenchmarkResult] = []

        for (index, file) in files.enumerated() {
            logger.info("[\(index + 1)/\(files.count)] Processing: \(file.fileName)")

            do {
                // Load audio
                let samples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
                let duration = Double(samples.count) / Double(CohereAsrConfig.sampleRate)

                // Transcribe
                let startTime = CFAbsoluteTimeGetCurrent()
                let hypothesis = try await manager.transcribe(
                    audioSamples: samples,
                    maxNewTokens: maxTokens
                )
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime

                let rtfx = duration / elapsed

                // Calculate WER
                let metrics = WERCalculator.calculateWERAndCER(
                    hypothesis: hypothesis,
                    reference: file.transcript
                )
                let wer = metrics.wer
                let cer = metrics.cer

                results.append(
                    CohereBenchmarkResult(
                        fileName: file.fileName,
                        reference: file.transcript,
                        hypothesis: hypothesis,
                        wer: wer,
                        cer: cer,
                        duration: duration,
                        processingTime: elapsed,
                        rtfx: rtfx
                    )
                )

                logger.info("  WER: \(String(format: "%.2f", wer))%, RTFx: \(String(format: "%.2f", rtfx))x")

            } catch {
                logger.error("  Failed: \(error)")
            }
        }

        return results
    }

    // MARK: - Helper Functions

    private static func collectBenchmarkAudioFiles(from directory: URL) throws -> [BenchmarkAudioFile] {
        var files: [BenchmarkAudioFile] = []
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
                        BenchmarkAudioFile(
                            fileName: audioFileName,
                            audioPath: audioPath,
                            transcript: transcript
                        ))
                }
            }
        }

        return files.sorted { $0.fileName < $1.fileName }
    }

    private static func collectFleursFiles(
        language: String,
        maxFiles: Int?,
        fleursDir: String?
    ) throws -> [BenchmarkAudioFile] {
        let baseDir = fleursDir ?? NSHomeDirectory() + "/Library/Application Support/FluidAudio/Datasets/fleurs"
        let langDir = URL(fileURLWithPath: baseDir).appendingPathComponent(language)

        guard FileManager.default.fileExists(atPath: langDir.path) else {
            throw NSError(
                domain: "CohereBenchmark",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "FLEURS dataset not found for \(language) at \(langDir.path)"]
            )
        }

        // Read transcripts
        let transcriptPath = langDir.appendingPathComponent("\(language).trans.txt")
        let transcriptData = try String(contentsOf: transcriptPath)
        let lines = transcriptData.components(separatedBy: .newlines).filter { !$0.isEmpty }

        var files: [BenchmarkAudioFile] = []
        for line in lines.prefix(maxFiles ?? lines.count) {
            let parts = line.components(separatedBy: " ")
            guard parts.count >= 2 else { continue }

            let fileId = parts[0]
            let transcript = parts.dropFirst().joined(separator: " ")
            let audioPath = langDir.appendingPathComponent("\(fileId).wav")

            if FileManager.default.fileExists(atPath: audioPath.path) {
                files.append(
                    BenchmarkAudioFile(
                        fileName: fileId,
                        audioPath: audioPath,
                        transcript: transcript
                    ))
            }
        }

        return files
    }

    private static func saveCohereBenchmarkResults(_ results: [CohereBenchmarkResult], to outputFile: String) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(results)
        try data.write(to: URL(fileURLWithPath: outputFile))
        logger.info("Results saved to: \(outputFile)")
    }

    private static func calculateWER(from results: [CohereBenchmarkResult]) -> Double {
        let totalWER = results.map(\.wer).reduce(0, +)
        return results.isEmpty ? 0 : totalWER / Double(results.count)
    }

    private static func printSummary(_ results: [CohereBenchmarkResult]) {
        let avgWER = calculateWER(from: results)
        let avgCER = results.map(\.cer).reduce(0, +) / Double(results.count)
        let avgRTFx = results.map(\.rtfx).reduce(0, +) / Double(results.count)

        logger.info(String(repeating: "=", count: 50))
        logger.info("BENCHMARK SUMMARY")
        logger.info(String(repeating: "=", count: 50))
        logger.info("  Files processed: \(results.count)")
        logger.info("  Average WER: \(String(format: "%.2f", avgWER))%")
        logger.info("  Average CER: \(String(format: "%.2f", avgCER))%")
        logger.info("  Average RTFx: \(String(format: "%.2f", avgRTFx))x")
    }

    private static func printUsage() {
        logger.info(
            """

            Cohere Transcribe Benchmark

            Usage: fluidaudio cohere-benchmark [options]

            Options:
                --help, -h                Show this help message
                --dataset <name>          Dataset to use: librispeech (default), fleurs
                --subset <name>           LibriSpeech subset (test-clean, test-other, dev-clean)
                --max-files <n>           Maximum number of files to process per language
                --model-dir <path>        Path to local model directory (skips download)
                --variant <fp16|int8>     Model variant (default: int8)
                --max-tokens <n>          Maximum tokens to generate (default: 200)
                --languages <codes>       Comma-separated FLEURS language codes for FLEURS
                --fleurs-dir <path>       Custom FLEURS dataset directory
                --output <file>           Output JSON file (default: cohere_benchmark_results.json)

            Supported FLEURS languages (14 total):
                en_us (English), fr_fr (French), de_de (German), es_419 (Spanish),
                it_it (Italian), pt_br (Portuguese), nl_nl (Dutch), pl_pl (Polish),
                el_gr (Greek), ar_eg (Arabic), ja_jp (Japanese), cmn_hans_cn (Chinese),
                ko_kr (Korean), vi_vn (Vietnamese)

            Examples:
                # LibriSpeech test-clean (100 files)
                fluidaudio cohere-benchmark --subset test-clean --max-files 100

                # FLEURS multilingual (English, French, German)
                fluidaudio cohere-benchmark --dataset fleurs --languages en_us,fr_fr,de_de --max-files 50

                # Use FP16 variant
                fluidaudio cohere-benchmark --variant fp16 --max-files 200
            """
        )
    }
}

// MARK: - Supporting Types

private struct BenchmarkAudioFile {
    let fileName: String
    let audioPath: URL
    let transcript: String
}

struct CohereBenchmarkResult: Codable {
    let fileName: String
    let reference: String
    let hypothesis: String
    let wer: Double
    let cer: Double
    let duration: Double
    let processingTime: Double
    let rtfx: Double
}
#endif
