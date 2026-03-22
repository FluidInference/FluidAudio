#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Benchmark for Qwen3-ASR supporting LibriSpeech (English) and FLEURS (multilingual).
///
/// Runs inference through `Qwen3AsrManager` with WER/CER evaluation.
enum Qwen3AsrBenchmark {
    private static let logger = AppLogger(category: "Qwen3Benchmark")

    /// Map FLEURS language codes to Qwen3AsrConfig.Language.
    /// All 30 languages auto-download from FluidInference/fleurs-full.
    private static let fleursToQwen3Language: [String: Qwen3AsrConfig.Language] = [
        // Asian languages
        "cmn_hans_cn": .chinese,
        "yue_hant_hk": .cantonese,
        "ja_jp": .japanese,
        "ko_kr": .korean,
        "vi_vn": .vietnamese,
        "th_th": .thai,
        "id_id": .indonesian,
        "ms_my": .malay,
        "hi_in": .hindi,
        "ar_eg": .arabic,
        "tr_tr": .turkish,
        "fa_ir": .persian,
        "fil_ph": .filipino,
        // European languages
        "en_us": .english,
        "de_de": .german,
        "fr_fr": .french,
        "es_419": .spanish,
        "pt_br": .portuguese,
        "it_it": .italian,
        "nl_nl": .dutch,
        "ru_ru": .russian,
        "pl_pl": .polish,
        "sv_se": .swedish,
        "da_dk": .danish,
        "fi_fi": .finnish,
        "cs_cz": .czech,
        "el_gr": .greek,
        "hu_hu": .hungarian,
        "ro_ro": .romanian,
        "mk_mk": .macedonian,
    ]

    static func runCLI(arguments: [String]) async {
        var dataset = "librispeech"
        var subset = "test-clean"
        var maxFiles: Int? = nil
        var modelDir: String? = nil
        var outputFile = "qwen3_asr_benchmark_results.json"
        var languages: [String] = ["cmn_hans_cn"]
        var fleursDir: String? = nil
        var variant: Qwen3AsrVariant = .f32

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
                    if let parsed = Qwen3AsrVariant(rawValue: v) {
                        variant = parsed
                    } else {
                        logger.error("Unknown variant '\(arguments[i + 1])'. Use 'f32' or 'int8'.")
                        exit(1)
                    }
                    i += 1
                }
            default:
                break
            }
            i += 1
        }

        logger.info("Qwen3-ASR Benchmark (2-model pipeline, \(variant.rawValue))")
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
                logger.info("Downloading Qwen3-ASR \(variant.rawValue) models...")
                let cacheDir = try await Qwen3AsrModels.download(variant: variant)
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
            case "aishell":
                try await runAishellBenchmark(
                    manager: manager,
                    maxFiles: maxFiles,
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
        let allFiles = try collectBenchmarkAudioFiles(from: datasetPath)
        let files = Array(allFiles.prefix(maxFiles ?? allFiles.count))
        logger.info("Collected \(files.count) files from LibriSpeech \(subset)")

        let results = try await runBenchmarkLoop(
            manager: manager,
            files: files.map { ($0.fileName, $0.audioPath, $0.transcript) },
            language: nil
        )

        let summary = Qwen3BenchmarkSummary(results: results)
        printSummary(summary: summary, datasetLabel: "LibriSpeech \(subset)")
        try writeJSON(
            results: results,
            summary: summary,
            outputFile: outputFile,
            dataset: "librispeech",
            subset: subset,
            language: nil
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

            // Auto-download if not present
            if !FileManager.default.fileExists(atPath: languageDir.path) {
                logger.info("FLEURS data not found for \(language), downloading...")
                do {
                    try await downloadFLEURSLanguage(
                        language: language,
                        targetDir: languageDir,
                        maxFiles: maxFiles
                    )
                } catch {
                    logger.error("Failed to download FLEURS \(language): \(error.localizedDescription)")
                    continue
                }
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

            let summary = Qwen3BenchmarkSummary(results: results)
            printSummary(summary: summary, datasetLabel: "FLEURS \(language)")
            try writeJSON(
                results: results,
                summary: summary,
                outputFile: langOutputFile,
                dataset: "fleurs",
                subset: nil,
                language: language
            )
        }
    }

    // MARK: - FLEURS Download

    /// Download FLEURS data for a language from HuggingFace.
    private static func downloadFLEURSLanguage(
        language: String,
        targetDir: URL,
        maxFiles: Int?
    ) async throws {
        try FileManager.default.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let datasetRepo = "FluidInference/fleurs-full"
        logger.info("Downloading from HuggingFace: \(datasetRepo)/\(language)...")

        // List files in the language directory
        let apiURL = try ModelRegistry.apiDatasets(datasetRepo, "tree/main/\(language)")
        let (listData, _) = try await DownloadUtils.fetchWithAuth(from: apiURL)

        guard let items = try JSONSerialization.jsonObject(with: listData) as? [[String: Any]] else {
            throw Qwen3AsrError.generationFailed("Could not parse file list from HuggingFace")
        }

        // Find transcript file and audio files
        var audioFiles: [String] = []
        let transFile = targetDir.appendingPathComponent("\(language).trans.txt")

        for item in items {
            guard let itemPath = item["path"] as? String,
                let itemType = item["type"] as? String,
                itemType == "file"
            else { continue }

            let fileName = URL(fileURLWithPath: itemPath).lastPathComponent

            if fileName == "\(language).trans.txt" {
                // Download transcript file
                let downloadURL = try ModelRegistry.resolveDataset(datasetRepo, itemPath)
                let transData = try await DownloadUtils.fetchHuggingFaceFile(
                    from: downloadURL,
                    description: "\(language) transcript"
                )
                try transData.write(to: transFile, options: .atomic)

                let transcriptContent = String(data: transData, encoding: .utf8) ?? ""
                let lines = transcriptContent.components(separatedBy: .newlines).filter { !$0.isEmpty }
                logger.info("Downloaded \(lines.count) transcriptions")
            } else if fileName.hasSuffix(".wav") {
                audioFiles.append(itemPath)
            }
        }

        // Download audio files
        let maxDownload = maxFiles ?? audioFiles.count
        var downloadedCount = 0

        for audioPath in audioFiles.prefix(maxDownload) {
            let fileName = URL(fileURLWithPath: audioPath).lastPathComponent
            let audioFile = targetDir.appendingPathComponent(fileName)

            // Skip if already exists and valid
            if FileManager.default.fileExists(atPath: audioFile.path) {
                if isValidAudioFile(audioFile) {
                    downloadedCount += 1
                    continue
                }
                try? FileManager.default.removeItem(at: audioFile)
            }

            // Download audio file
            let downloadURL = try ModelRegistry.resolveDataset(datasetRepo, audioPath)
            let audioData = try await DownloadUtils.fetchHuggingFaceFile(
                from: downloadURL,
                description: "\(language)/\(fileName)"
            )
            try audioData.write(to: audioFile, options: .atomic)
            downloadedCount += 1

            if downloadedCount % 20 == 0 {
                logger.info("Downloaded \(downloadedCount)/\(maxDownload) audio files...")
            }
        }

        logger.info("Downloaded \(downloadedCount) audio files for \(language)")
    }

    private static func isValidAudioFile(_ url: URL) -> Bool {
        do {
            _ = try AVAudioFile(forReading: url)
            return true
        } catch {
            return false
        }
    }

    // MARK: - AISHELL-1 Benchmark

    @available(macOS 15, iOS 18, *)
    private static func runAishellBenchmark(
        manager: Qwen3AsrManager,
        maxFiles: Int?,
        outputFile: String
    ) async throws {
        let aishellDir =
            FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/FluidAudio/AISHELL/cmn_hans_cn")

        guard FileManager.default.fileExists(atPath: aishellDir.path) else {
            logger.error(
                "AISHELL-1 data not found at \(aishellDir.path). "
                    + "Please extract AISHELL-1 test set first."
            )
            exit(1)
        }

        let allFiles = try collectAishellFiles(directory: aishellDir)
        let files = Array(allFiles.prefix(maxFiles ?? allFiles.count))
        logger.info("Collected \(files.count) files from AISHELL-1 test set")

        let results = try await runBenchmarkLoop(
            manager: manager,
            files: files.map { ($0.fileName, $0.audioPath, $0.transcript) },
            language: .chinese
        )

        let summary = Qwen3BenchmarkSummary(results: results)
        printSummary(summary: summary, datasetLabel: "AISHELL-1 test")
        try writeJSON(
            results: results,
            summary: summary,
            outputFile: outputFile,
            dataset: "aishell",
            subset: "test",
            language: "zh"
        )
    }

    private static func collectAishellFiles(directory: URL) throws -> [BenchmarkAudioFile] {
        let transFile = directory.appendingPathComponent("cmn_hans_cn.trans.txt")
        guard FileManager.default.fileExists(atPath: transFile.path) else {
            throw Qwen3AsrError.generationFailed(
                "Transcript file not found: \(transFile.path)"
            )
        }

        let content = try String(contentsOf: transFile)
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        var files: [BenchmarkAudioFile] = []

        for line in lines {
            guard let spaceIndex = line.firstIndex(of: " ") else { continue }
            let fileId = String(line[line.startIndex..<spaceIndex])
            let transcript = String(line[line.index(after: spaceIndex)...])

            let wavPath = directory.appendingPathComponent("\(fileId).wav")
            guard FileManager.default.fileExists(atPath: wavPath.path) else { continue }

            files.append(
                BenchmarkAudioFile(
                    fileName: wavPath.lastPathComponent,
                    audioPath: wavPath,
                    transcript: transcript
                )
            )
        }

        return files
    }

    // MARK: - FLEURS File Collection

    private static func collectFLEURSFiles(
        language: String, directory: URL
    ) throws -> [BenchmarkAudioFile] {
        let transFile = directory.appendingPathComponent("\(language).trans.txt")
        guard FileManager.default.fileExists(atPath: transFile.path) else {
            throw Qwen3AsrError.generationFailed(
                "Transcript file not found: \(transFile.path)"
            )
        }

        let content = try String(contentsOf: transFile)
        let lines = content.components(separatedBy: .newlines).filter { !$0.isEmpty }
        var files: [BenchmarkAudioFile] = []

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
                BenchmarkAudioFile(
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
        language: Qwen3AsrConfig.Language?
    ) async throws -> [Qwen3BenchmarkResult] {
        var results: [Qwen3BenchmarkResult] = []
        let audioConverter = AudioConverter()

        for (index, file) in files.enumerated() {
            do {
                logger.info("[\(index + 1)/\(files.count)] \(file.fileName)")

                let samples = try audioConverter.resampleAudioFile(path: file.audioPath.path)
                let audioLength = Double(samples.count) / Double(Qwen3AsrConfig.sampleRate)

                let inferenceStart = CFAbsoluteTimeGetCurrent()
                let hypothesis = try await manager.transcribe(
                    audioSamples: samples,
                    language: language,
                    maxNewTokens: 512
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

            // Give system time to reclaim CoreML MLState IOSurface resources every 25 files.
            // Without this pause, IOSurface limit (~200) is exhausted causing crashes.
            if (index + 1) % 25 == 0 {
                logger.info("Memory cleanup pause...")
                try? await Task.sleep(for: .seconds(1))
            }
        }

        return results
    }

    // MARK: - Summary & Output

    private static func printSummary(summary: Qwen3BenchmarkSummary, datasetLabel: String) {
        guard summary.filesProcessed > 0 else {
            logger.error("No results produced")
            return
        }

        print("")
        print("--- Qwen3-ASR Benchmark Results ---")
        print("   Dataset: \(datasetLabel)")
        print("   Files processed: \(summary.filesProcessed)")
        print("   Average WER: \(String(format: "%.1f", summary.avgWER * 100))%")
        print("   Median WER: \(String(format: "%.1f", summary.medianWER * 100))%")
        print("   Average CER: \(String(format: "%.1f", summary.avgCER * 100))%")
        print("   Median CER: \(String(format: "%.1f", summary.medianCER * 100))%")
        print("   Median RTFx: \(String(format: "%.1f", summary.medianRTFx))x")
        print(
            "   Overall RTFx: \(String(format: "%.1f", summary.overallRTFx))x (\(String(format: "%.1f", summary.totalAudio))s / \(String(format: "%.1f", summary.totalInference))s)"
        )
    }

    private static func writeJSON(
        results: [Qwen3BenchmarkResult],
        summary: Qwen3BenchmarkSummary,
        outputFile: String,
        dataset: String,
        subset: String?,
        language: String?
    ) throws {
        guard !results.isEmpty else { return }

        var summaryDict: [String: Any] = [
            "model": "qwen3-asr-0.6b",
            "dataset": dataset,
            "filesProcessed": summary.filesProcessed,
            "averageWER": summary.avgWER,
            "medianWER": summary.medianWER,
            "averageCER": summary.avgCER,
            "medianCER": summary.medianCER,
            "medianRTFx": summary.medianRTFx,
            "overallRTFx": summary.overallRTFx,
            "totalAudioDuration": summary.totalAudio,
            "totalInferenceTime": summary.totalInference,
        ]

        if let subset = subset {
            summaryDict["subset"] = subset
        }
        if let language = language {
            summaryDict["language"] = language
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
            "summary": summaryDict,
            "results": jsonResults,
        ]

        let jsonData = try JSONSerialization.data(
            withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try jsonData.write(to: URL(fileURLWithPath: outputFile))
        logger.info("Results written to \(outputFile)")
    }

    // MARK: - LibriSpeech File Collection

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

    // MARK: - Usage

    private static func printUsage() {
        logger.info(
            """

            Qwen3-ASR Benchmark (2-model pipeline with Swift-side embedding)

            Usage: fluidaudio qwen3-benchmark [options]

            Options:
                --dataset <name>        Dataset: librispeech (default), fleurs, or aishell
                --subset <name>         LibriSpeech subset (default: test-clean)
                --languages <list>      FLEURS language codes, comma-separated (default: cmn_hans_cn)
                --max-files <number>    Max files to process (default: all)
                --model-dir <path>      Local model directory (skips download)
                --variant <f32|int8>    Model variant (default: f32). int8 uses ~50% less RAM.
                --fleurs-dir <path>     FLEURS data directory (default: ~/Library/Application Support/FluidAudio/FLEURS)
                --output <file>         Output JSON path (default: qwen3_asr_benchmark_results.json)
                --help, -h              Show this help

            Examples:
                # English (LibriSpeech)
                fluidaudio qwen3-benchmark --max-files 100
                fluidaudio qwen3-benchmark --subset test-other --max-files 50

                # Chinese (AISHELL-1)
                fluidaudio qwen3-benchmark --dataset aishell
                fluidaudio qwen3-benchmark --dataset aishell --max-files 100

                # Chinese (FLEURS)
                fluidaudio qwen3-benchmark --dataset fleurs --languages cmn_hans_cn

                # Multiple languages (FLEURS)
                fluidaudio qwen3-benchmark --dataset fleurs --languages cmn_hans_cn,ja_jp,ko_kr

            Supported FLEURS languages (30 Qwen3 languages, all auto-download):

            Asian (13 languages):
                cmn_hans_cn  Chinese (Mandarin)     yue_hant_hk  Cantonese
                ja_jp        Japanese               ko_kr        Korean
                vi_vn        Vietnamese             th_th        Thai
                id_id        Indonesian             ms_my        Malay
                hi_in        Hindi                  ar_eg        Arabic
                tr_tr        Turkish                fa_ir        Persian
                fil_ph       Filipino

            European (17 languages):
                en_us   English         de_de   German          fr_fr   French
                es_419  Spanish         pt_br   Portuguese      it_it   Italian
                nl_nl   Dutch           ru_ru   Russian         pl_pl   Polish
                sv_se   Swedish         da_dk   Danish          fi_fi   Finnish
                cs_cz   Czech           el_gr   Greek           hu_hu   Hungarian
                ro_ro   Romanian        mk_mk   Macedonian

            All languages auto-download from FluidInference/fleurs-full.
            """
        )
    }
}

// MARK: - Types

private struct Qwen3BenchmarkResult {
    let fileName: String
    let hypothesis: String
    let reference: String
    let wer: Double
    let cer: Double
    let audioLength: Double
    let processingTime: Double
}

private struct BenchmarkAudioFile {
    let fileName: String
    let audioPath: URL
    let transcript: String
}

private struct Qwen3BenchmarkSummary {
    let filesProcessed: Int
    let avgWER: Double
    let medianWER: Double
    let avgCER: Double
    let medianCER: Double
    let medianRTFx: Double
    let overallRTFx: Double
    let totalAudio: Double
    let totalInference: Double

    init(results: [Qwen3BenchmarkResult]) {
        guard !results.isEmpty else {
            self.filesProcessed = 0
            self.avgWER = 0
            self.medianWER = 0
            self.avgCER = 0
            self.medianCER = 0
            self.medianRTFx = 0
            self.overallRTFx = 0
            self.totalAudio = 0
            self.totalInference = 0
            return
        }

        self.filesProcessed = results.count
        self.avgWER = results.map(\.wer).reduce(0, +) / Double(results.count)
        self.avgCER = results.map(\.cer).reduce(0, +) / Double(results.count)
        self.totalAudio = results.map(\.audioLength).reduce(0, +)
        self.totalInference = results.map(\.processingTime).reduce(0, +)
        self.overallRTFx = totalAudio / totalInference
        self.medianWER = results.map(\.wer).sorted()[results.count / 2]
        self.medianCER = results.map(\.cer).sorted()[results.count / 2]

        let sortedRTFx = results.map { $0.audioLength / $0.processingTime }.sorted()
        self.medianRTFx = sortedRTFx[sortedRTFx.count / 2]
    }
}
#endif
