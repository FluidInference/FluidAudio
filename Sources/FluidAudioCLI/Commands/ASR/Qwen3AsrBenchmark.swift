#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Benchmark for Qwen3-ASR supporting LibriSpeech (English) and FLEURS (multilingual).
///
/// Runs inference through `Qwen3AsrManager` with WER/CER evaluation.
enum Qwen3AsrBenchmark {
    private static let logger = AppLogger(category: "Qwen3Benchmark")

    /// Map FLEURS language codes to the short language tags Qwen3-ASR expects.
    private static let fleursToQwen3Language: [String: String] = [
        "cmn_hans_cn": "zh",
        "yue_hant_hk": "yue",
        "ja_jp": "ja",
        "ko_kr": "ko",
        "vi_vn": "vi",
        "th_th": "th",
        "id_id": "id",
        "ms_my": "ms",
        "hi_in": "hi",
        "ar_eg": "ar",
        "tr_tr": "tr",
        "ru_ru": "ru",
        "de_de": "de",
        "fr_fr": "fr",
        "es_419": "es",
        "en_us": "en",
    ]

    static func runCLI(arguments: [String]) async {
        var dataset = "librispeech"
        var subset = "test-clean"
        var maxFiles: Int? = nil
        var modelDir: String? = nil
        var outputFile = "qwen3_asr_benchmark_results.json"
        var languages: [String] = ["cmn_hans_cn"]
        var fleursDir: String? = nil

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
            default:
                break
            }
            i += 1
        }

        logger.info("Qwen3-ASR Benchmark")
        logger.info("  Dataset: \(dataset)")
        if dataset == "librispeech" {
            logger.info("  Subset: \(subset)")
        } else {
            logger.info("  Languages: \(languages.joined(separator: ", "))")
        }
        logger.info("  Max files: \(maxFiles?.description ?? "all")")
        logger.info("  Model dir: \(modelDir ?? "auto-download")")
        logger.info("  Output: \(outputFile)")

        guard #available(macOS 15, iOS 18, *) else {
            logger.error("Qwen3-ASR requires macOS 15 or later")
            exit(1)
        }

        do {
            // 1. Load Qwen3-ASR models
            let manager = Qwen3AsrManager()
            if let dir = modelDir {
                logger.info("Loading models from \(dir)")
                try await manager.loadModels(from: URL(fileURLWithPath: dir))
            } else {
                logger.info("Downloading Qwen3-ASR models...")
                let cacheDir = try await Qwen3AsrModels.download()
                try await manager.loadModels(from: cacheDir)
            }

            // 2. Collect files based on dataset
            switch dataset {
            case "fleurs":
                try await runFleursBenchmark(
                    manager: manager,
                    languages: languages,
                    maxFiles: maxFiles,
                    fleursDir: fleursDir,
                    outputFile: outputFile
                )
            default:
                try await runLibriSpeechBenchmark(
                    manager: manager,
                    subset: subset,
                    maxFiles: maxFiles,
                    outputFile: outputFile
                )
            }

        } catch {
            logger.error("Benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - LibriSpeech Benchmark

    @available(macOS 15, iOS 18, *)
    private static func runLibriSpeechBenchmark(
        manager: Qwen3AsrManager,
        subset: String,
        maxFiles: Int?,
        outputFile: String
    ) async throws {
        let benchmark = ASRBenchmark()
        try await benchmark.downloadLibriSpeech(subset: subset)
        let datasetPath = benchmark.getLibriSpeechDirectory().appendingPathComponent(subset)
        let allFiles = try collectLibriSpeechFiles(from: datasetPath)
        let files = Array(allFiles.prefix(maxFiles ?? allFiles.count))
        logger.info("Collected \(files.count) files from LibriSpeech \(subset)")

        let results = try await runBenchmarkLoop(
            manager: manager,
            files: files.map { ($0.fileName, $0.audioPath, $0.transcript) },
            language: nil
        )

        printSummary(results: results, dataset: "librispeech", subset: subset, language: nil)
        try writeJSON(
            results: results, outputFile: outputFile,
            dataset: "librispeech", subset: subset, language: nil
        )
    }

    // MARK: - FLEURS Benchmark

    @available(macOS 15, iOS 18, *)
    private static func runFleursBenchmark(
        manager: Qwen3AsrManager,
        languages: [String],
        maxFiles: Int?,
        fleursDir: String?,
        outputFile: String
    ) async throws {
        let baseFleursDir: URL
        if let dir = fleursDir {
            baseFleursDir = URL(fileURLWithPath: dir)
        } else {
            baseFleursDir =
                FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support/FluidAudio/FLEURS")
        }

        for language in languages {
            let languageDir = baseFleursDir.appendingPathComponent(language)
            guard FileManager.default.fileExists(atPath: languageDir.path) else {
                logger.error(
                    "FLEURS data not found for \(language) at \(languageDir.path). "
                        + "Run prepare_fleurs_chinese.py first."
                )
                continue
            }

            let allFiles = try collectFLEURSFiles(language: language, directory: languageDir)
            let files = Array(allFiles.prefix(maxFiles ?? allFiles.count))
            logger.info("Collected \(files.count) files for FLEURS \(language)")

            let qwen3Lang = fleursToQwen3Language[language]
            let results = try await runBenchmarkLoop(
                manager: manager,
                files: files.map { ($0.fileName, $0.audioPath, $0.transcript) },
                language: qwen3Lang
            )

            let langOutputFile: String
            if languages.count > 1 {
                let base = (outputFile as NSString).deletingPathExtension
                let ext = (outputFile as NSString).pathExtension
                langOutputFile = "\(base)_\(language).\(ext.isEmpty ? "json" : ext)"
            } else {
                langOutputFile = outputFile
            }

            printSummary(results: results, dataset: "fleurs", subset: nil, language: language)
            try writeJSON(
                results: results, outputFile: langOutputFile,
                dataset: "fleurs", subset: nil, language: language
            )
        }
    }

    // MARK: - FLEURS File Collection

    private static func collectFLEURSFiles(
        language: String, directory: URL
    ) throws -> [LibriSpeechFile] {
        let transFile = directory.appendingPathComponent("\(language).trans.txt")
        guard FileManager.default.fileExists(atPath: transFile.path) else {
            throw Qwen3AsrError.generationFailed(
                "Transcript file not found: \(transFile.path)"
            )
        }

        let content = try String(contentsOf: transFile)
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        var files: [LibriSpeechFile] = []

        for line in lines {
            // Format: file_id transcription
            guard let spaceIndex = line.firstIndex(of: " ") else { continue }
            let fileId = String(line[line.startIndex..<spaceIndex])
            let transcript = String(line[line.index(after: spaceIndex)...])

            // Try .wav first, then .flac
            let wavPath = directory.appendingPathComponent("\(fileId).wav")
            let flacPath = directory.appendingPathComponent("\(fileId).flac")

            let audioPath: URL
            if FileManager.default.fileExists(atPath: wavPath.path) {
                audioPath = wavPath
            } else if FileManager.default.fileExists(atPath: flacPath.path) {
                audioPath = flacPath
            } else {
                continue
            }

            files.append(
                LibriSpeechFile(
                    fileName: audioPath.lastPathComponent,
                    audioPath: audioPath,
                    transcript: transcript
                )
            )
        }

        return files
    }

    // MARK: - Shared Benchmark Loop

    @available(macOS 15, iOS 18, *)
    private static func runBenchmarkLoop(
        manager: Qwen3AsrManager,
        files: [(fileName: String, audioPath: URL, transcript: String)],
        language: String?
    ) async throws -> [Qwen3BenchmarkResult] {
        let startBenchmark = CFAbsoluteTimeGetCurrent()
        var results: [Qwen3BenchmarkResult] = []

        for (index, file) in files.enumerated() {
            do {
                logger.info("[\(index + 1)/\(files.count)] \(file.fileName)")

                let samples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
                let audioLength = Double(samples.count) / 16000.0

                let inferenceStart = CFAbsoluteTimeGetCurrent()
                let hypothesis = try await manager.transcribe(
                    audioSamples: samples,
                    language: language
                )
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
                let cerPct = metrics.cer * 100
                logger.info(
                    "  WER: \(String(format: "%.1f", werPct))% | CER: \(String(format: "%.1f", cerPct))% | RTFx: \(String(format: "%.1f", rtfx))x | \(String(format: "%.2f", audioLength))s audio in \(String(format: "%.2f", inferenceTime))s"
                )
                if werPct > 50.0 {
                    logger.info("  REF: \(file.transcript)")
                    logger.info("  HYP: \(hypothesis)")
                }
            } catch {
                logger.error("Failed \(file.fileName): \(error)")
            }
        }

        return results
    }

    // MARK: - Summary & Output

    private static func printSummary(
        results: [Qwen3BenchmarkResult],
        dataset: String,
        subset: String?,
        language: String?
    ) {
        guard !results.isEmpty else {
            logger.error("No results produced")
            return
        }

        let avgWER = results.map(\.wer).reduce(0, +) / Double(results.count)
        let avgCER = results.map(\.cer).reduce(0, +) / Double(results.count)
        let totalAudio = results.map(\.audioLength).reduce(0, +)
        let totalInference = results.map(\.processingTime).reduce(0, +)
        let overallRTFx = totalAudio / totalInference
        let medianWER = results.map(\.wer).sorted()[results.count / 2]
        let medianCER = results.map(\.cer).sorted()[results.count / 2]

        let sortedRTFx = results.map { $0.audioLength / $0.processingTime }.sorted()
        let medianRTFx = sortedRTFx[sortedRTFx.count / 2]

        let datasetLabel: String
        if let language = language {
            datasetLabel = "FLEURS \(language)"
        } else if let subset = subset {
            datasetLabel = "LibriSpeech \(subset)"
        } else {
            datasetLabel = dataset
        }

        print("")
        print("--- Qwen3-ASR Benchmark Results ---")
        print("   Dataset: \(datasetLabel)")
        print("   Files processed: \(results.count)")
        print("   Average WER: \(String(format: "%.1f", avgWER * 100))%")
        print("   Median WER: \(String(format: "%.1f", medianWER * 100))%")
        print("   Average CER: \(String(format: "%.1f", avgCER * 100))%")
        print("   Median CER: \(String(format: "%.1f", medianCER * 100))%")
        print("   Median RTFx: \(String(format: "%.1f", medianRTFx))x")
        print(
            "   Overall RTFx: \(String(format: "%.1f", overallRTFx))x (\(String(format: "%.1f", totalAudio))s / \(String(format: "%.1f", totalInference))s)"
        )
    }

    private static func writeJSON(
        results: [Qwen3BenchmarkResult],
        outputFile: String,
        dataset: String,
        subset: String?,
        language: String?
    ) throws {
        guard !results.isEmpty else { return }

        let avgWER = results.map(\.wer).reduce(0, +) / Double(results.count)
        let avgCER = results.map(\.cer).reduce(0, +) / Double(results.count)
        let totalAudio = results.map(\.audioLength).reduce(0, +)
        let totalInference = results.map(\.processingTime).reduce(0, +)
        let overallRTFx = totalAudio / totalInference
        let medianWER = results.map(\.wer).sorted()[results.count / 2]

        let sortedRTFx = results.map { $0.audioLength / $0.processingTime }.sorted()
        let medianRTFx = sortedRTFx[sortedRTFx.count / 2]

        var summary: [String: Any] = [
            "model": "qwen3-asr-0.6b",
            "dataset": dataset,
            "filesProcessed": results.count,
            "averageWER": avgWER,
            "medianWER": medianWER,
            "averageCER": avgCER,
            "medianRTFx": medianRTFx,
            "overallRTFx": overallRTFx,
            "totalAudioDuration": totalAudio,
            "totalInferenceTime": totalInference,
        ]

        if let subset = subset {
            summary["subset"] = subset
        }
        if let language = language {
            summary["language"] = language
        }

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
                --dataset <name>        Dataset: librispeech (default) or fleurs
                --subset <name>         LibriSpeech subset (default: test-clean)
                --languages <list>      FLEURS language codes, comma-separated (default: cmn_hans_cn)
                --max-files <number>    Max files to process (default: all)
                --model-dir <path>      Local model directory (skips download)
                --fleurs-dir <path>     FLEURS data directory (default: ~/Library/Application Support/FluidAudio/FLEURS)
                --output <file>         Output JSON path (default: qwen3_asr_benchmark_results.json)
                --help, -h              Show this help

            Examples:
                # English (LibriSpeech)
                fluidaudio qwen3-benchmark --max-files 100
                fluidaudio qwen3-benchmark --subset test-other --max-files 50

                # Chinese (FLEURS)
                fluidaudio qwen3-benchmark --dataset fleurs --languages cmn_hans_cn
                fluidaudio qwen3-benchmark --dataset fleurs --languages cmn_hans_cn --max-files 20

                # Multiple languages
                fluidaudio qwen3-benchmark --dataset fleurs --languages cmn_hans_cn,ja_jp,ko_kr

            Supported FLEURS languages:
                cmn_hans_cn  Chinese (Mandarin)     ja_jp   Japanese
                yue_hant_hk  Chinese (Cantonese)    ko_kr   Korean
                vi_vn        Vietnamese             th_th   Thai
                id_id        Indonesian             ms_my   Malay
                hi_in        Hindi                  ar_eg   Arabic
                tr_tr        Turkish                ru_ru   Russian
                de_de        German                 fr_fr   French
                es_419       Spanish                en_us   English

            Note: FLEURS data must be prepared first using prepare_fleurs_chinese.py
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
