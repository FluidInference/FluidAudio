#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// FLEURS benchmark for the corrected mixed-precision Cohere pipeline
/// (INT8 encoder + FP16 decoder, or any single-precision combo) using
/// `CoherePipeline` with the bug fixes applied (filterbank mel, fp16-safe
/// cross-attention mask, repetition penalty, no-repeat-ngram, byte-fallback
/// detokenization).
enum CohereBenchmark {
    private static let logger = AppLogger(category: "CohereBenchmark")

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
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            exit(0)
        }

        var encoderDir: String?
        var decoderDir: String?
        var vocabDir: String?
        var modelDir: String?
        var dataset = "fleurs"
        var subset = "test-clean"
        var languages: [String] = ["en_us"]
        var maxFiles: Int?
        var fleursDir: String?
        var outputFile = "cohere_benchmark_results.json"
        var maxTokens = 108
        var repetitionPenalty: Float = 1.1
        var noRepeatNgram = 3
        var computeUnits: MLComputeUnits = .all
        var autoDownload = false
        var checkpointEvery = 100

        var i = 0
        while i < arguments.count {
            let arg = arguments[i]
            switch arg {
            case "--encoder-dir":
                if i + 1 < arguments.count {
                    encoderDir = arguments[i + 1]
                    i += 1
                }
            case "--decoder-dir":
                if i + 1 < arguments.count {
                    decoderDir = arguments[i + 1]
                    i += 1
                }
            case "--vocab-dir":
                if i + 1 < arguments.count {
                    vocabDir = arguments[i + 1]
                    i += 1
                }
            case "--model-dir":
                if i + 1 < arguments.count {
                    modelDir = arguments[i + 1]
                    i += 1
                }
            case "--languages":
                if i + 1 < arguments.count {
                    languages = arguments[i + 1].components(separatedBy: ",").map {
                        $0.trimmingCharacters(in: .whitespaces)
                    }
                    i += 1
                }
            case "--max-files":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    maxFiles = v
                    i += 1
                }
            case "--fleurs-dir":
                if i + 1 < arguments.count {
                    fleursDir = arguments[i + 1]
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--max-tokens":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    maxTokens = v
                    i += 1
                }
            case "--repetition-penalty":
                if i + 1 < arguments.count, let v = Float(arguments[i + 1]) {
                    repetitionPenalty = v
                    i += 1
                }
            case "--no-repeat-ngram":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    noRepeatNgram = v
                    i += 1
                }
            case "--dataset":
                if i + 1 < arguments.count {
                    dataset = arguments[i + 1].lowercased()
                    i += 1
                }
            case "--subset":
                if i + 1 < arguments.count {
                    subset = arguments[i + 1]
                    i += 1
                }
            case "--cpu-only":
                computeUnits = .cpuOnly
            case "--cpu-gpu":
                computeUnits = .cpuAndGPU
            case "--auto-download":
                autoDownload = true
            case "--checkpoint-every":
                if i + 1 < arguments.count, let v = Int(arguments[i + 1]) {
                    checkpointEvery = max(1, v)
                    i += 1
                }
            default:
                logger.warning("Ignoring unknown option: \(arg)")
            }
            i += 1
        }

        // Resolve model directories: explicit flags win, otherwise all fall back
        // to --model-dir. Vocab falls back to decoder then encoder dir.
        let encDir = encoderDir ?? modelDir
        let decDir = decoderDir ?? modelDir
        let vocDir = vocabDir ?? modelDir ?? decoderDir ?? encoderDir
        guard let encDir, let decDir, let vocDir else {
            logger.error(
                "Need --model-dir, or --encoder-dir + --decoder-dir (+ optional --vocab-dir)")
            printUsage()
            exit(1)
        }

        guard #available(macOS 14, iOS 17, *) else {
            logger.error("Cohere benchmark requires macOS 14 or later")
            exit(1)
        }

        logger.info("Cohere Transcribe Benchmark")
        logger.info("  Dataset:          \(dataset)")
        if dataset == "librispeech" {
            logger.info("  Subset:           \(subset)")
        } else {
            logger.info("  Languages:        \(languages.joined(separator: ", "))")
        }
        logger.info("  Encoder dir:      \(encDir)")
        logger.info("  Decoder dir:      \(decDir)")
        logger.info("  Vocab dir:        \(vocDir)")
        logger.info("  Max files:        \(maxFiles?.description ?? "all")")
        logger.info("  Max tokens:       \(maxTokens)")
        logger.info("  Rep penalty:      \(repetitionPenalty)")
        logger.info("  No-repeat-ngram:  \(noRepeatNgram)")
        logger.info("  Auto-download:    \(autoDownload)")
        logger.info("  Checkpoint every: \(checkpointEvery) files")

        do {
            // Load models once
            let loadStart = CFAbsoluteTimeGetCurrent()
            let models = try await CoherePipeline.loadModels(
                encoderDir: URL(fileURLWithPath: encDir),
                decoderDir: URL(fileURLWithPath: decDir),
                vocabDir: URL(fileURLWithPath: vocDir),
                computeUnits: computeUnits
            )
            logger.info(
                "Models loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - loadStart))s")

            let pipeline = CoherePipeline()

            let allResults: [CohereBenchmarkResult]
            let perLanguageSummaries: [LanguageSummary]

            switch dataset {
            case "librispeech":
                (allResults, perLanguageSummaries) = try await runLibriSpeech(
                    models: models,
                    pipeline: pipeline,
                    subset: subset,
                    maxFiles: maxFiles,
                    autoDownload: autoDownload,
                    maxTokens: maxTokens,
                    repetitionPenalty: repetitionPenalty,
                    noRepeatNgram: noRepeatNgram,
                    outputFile: outputFile,
                    checkpointEvery: checkpointEvery
                )
            case "fleurs":
                (allResults, perLanguageSummaries) = try await runFleurs(
                    models: models,
                    pipeline: pipeline,
                    languages: languages,
                    maxFiles: maxFiles,
                    fleursDir: fleursDir,
                    autoDownload: autoDownload,
                    maxTokens: maxTokens,
                    repetitionPenalty: repetitionPenalty,
                    noRepeatNgram: noRepeatNgram,
                    outputFile: outputFile,
                    checkpointEvery: checkpointEvery
                )
            default:
                logger.error("Unknown --dataset \(dataset). Supported: librispeech, fleurs")
                exit(1)
            }

            try saveResults(
                allResults: allResults,
                perLanguage: perLanguageSummaries,
                to: outputFile
            )
            printFinalSummary(perLanguage: perLanguageSummaries)
        } catch {
            logger.error("Benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - FLEURS

    private static func runFleurs(
        models: CoherePipeline.LoadedModels,
        pipeline: CoherePipeline,
        languages: [String],
        maxFiles: Int?,
        fleursDir: String?,
        autoDownload: Bool,
        maxTokens: Int,
        repetitionPenalty: Float,
        noRepeatNgram: Int,
        outputFile: String,
        checkpointEvery: Int
    ) async throws -> ([CohereBenchmarkResult], [LanguageSummary]) {
        let fleursCacheDir =
            fleursDir
            ?? NSHomeDirectory() + "/Library/Application Support/FluidAudio/Datasets/fleurs"

        if autoDownload {
            let supportedCodes = languages.filter { fleursToCohereLanguage.keys.contains($0) }
            // `samplesPerLanguage: Int.max` is FLEURSBenchmark's sentinel for
            // "download all available". Passing 100 when --max-files is
            // omitted (Devin Review finding) would silently cap downloads.
            let fleurs = FLEURSBenchmark(
                config: FLEURSBenchmark.FLEURSConfig(
                    languages: supportedCodes,
                    samplesPerLanguage: maxFiles ?? Int.max,
                    outputFile: "/dev/null",
                    cacheDir: fleursCacheDir,
                    debugMode: false
                ))
            try await fleurs.downloadFLEURS(languages: supportedCodes)
        }

        var allResults: [CohereBenchmarkResult] = []
        var perLanguageSummaries: [LanguageSummary] = []

        for langCode in languages {
            guard let cohereLang = fleursToCohereLanguage[langCode] else {
                logger.warning("Unsupported language for Cohere: \(langCode)")
                continue
            }

            logger.info("Processing language: \(langCode)")

            let files: [BenchmarkAudioFile]
            do {
                files = try collectFleursFiles(
                    language: langCode,
                    maxFiles: maxFiles,
                    fleursDir: fleursCacheDir
                )
            } catch {
                logger.error("  Failed to collect files for \(langCode): \(error)")
                continue
            }

            logger.info("  Collected \(files.count) files for \(langCode)")

            let priorResults = allResults
            let priorSummaries = perLanguageSummaries
            let langResults = await transcribeFiles(
                files: files,
                language: cohereLang,
                languageLabel: langCode,
                models: models,
                pipeline: pipeline,
                maxTokens: maxTokens,
                repetitionPenalty: repetitionPenalty,
                noRepeatNgram: noRepeatNgram,
                checkpointEvery: checkpointEvery,
                onCheckpoint: { partial in
                    let partialSummary = summarize(language: langCode, results: partial)
                    let combinedResults = priorResults + partial
                    let combinedSummaries = priorSummaries + [partialSummary]
                    do {
                        try saveResults(
                            allResults: combinedResults,
                            perLanguage: combinedSummaries,
                            to: outputFile
                        )
                        logger.info(
                            "  [checkpoint] saved \(combinedResults.count) results → \(outputFile)"
                        )
                    } catch {
                        logger.warning("  [checkpoint] save failed: \(error)")
                    }
                }
            )

            let summary = summarize(language: langCode, results: langResults)
            perLanguageSummaries.append(summary)
            logger.info(
                "  \(langCode) summary: "
                    + "WER=\(String(format: "%.2f", summary.avgWER))% "
                    + "CER=\(String(format: "%.2f", summary.avgCER))% "
                    + "RTFx=\(String(format: "%.2f", summary.avgRTFx))x "
                    + "(\(summary.samplesProcessed) samples)"
            )

            allResults.append(contentsOf: langResults)
        }

        return (allResults, perLanguageSummaries)
    }

    // MARK: - LibriSpeech

    private static func runLibriSpeech(
        models: CoherePipeline.LoadedModels,
        pipeline: CoherePipeline,
        subset: String,
        maxFiles: Int?,
        autoDownload: Bool,
        maxTokens: Int,
        repetitionPenalty: Float,
        noRepeatNgram: Int,
        outputFile: String,
        checkpointEvery: Int
    ) async throws -> ([CohereBenchmarkResult], [LanguageSummary]) {
        // Reuse Parakeet's LibriSpeech downloader/cache layout.
        let downloader = ASRBenchmark()
        if autoDownload {
            try await downloader.downloadLibriSpeech(subset: subset)
        }
        let datasetPath = downloader.getLibriSpeechDirectory().appendingPathComponent(subset)

        guard FileManager.default.fileExists(atPath: datasetPath.path) else {
            throw NSError(
                domain: "CohereBenchmark",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "LibriSpeech subset \(subset) not found at \(datasetPath.path). "
                        + "Pass --auto-download to fetch."
                ]
            )
        }

        var files = try collectLibriSpeechFiles(from: datasetPath)
        if let cap = maxFiles, files.count > cap {
            files = Array(files.prefix(cap))
        }
        logger.info("Collected \(files.count) LibriSpeech files (\(subset))")

        let label = "en (\(subset))"
        let langResults = await transcribeFiles(
            files: files,
            language: .english,
            languageLabel: label,
            models: models,
            pipeline: pipeline,
            maxTokens: maxTokens,
            repetitionPenalty: repetitionPenalty,
            noRepeatNgram: noRepeatNgram,
            checkpointEvery: checkpointEvery,
            onCheckpoint: { partial in
                let partialSummary = summarize(language: label, results: partial)
                do {
                    try saveResults(
                        allResults: partial,
                        perLanguage: [partialSummary],
                        to: outputFile
                    )
                    logger.info(
                        "  [checkpoint] saved \(partial.count) results → \(outputFile)"
                    )
                } catch {
                    logger.warning("  [checkpoint] save failed: \(error)")
                }
            }
        )

        let summary = summarize(language: "en (\(subset))", results: langResults)
        logger.info(
            "  LibriSpeech \(subset) summary: "
                + "WER=\(String(format: "%.2f", summary.avgWER))% "
                + "CER=\(String(format: "%.2f", summary.avgCER))% "
                + "RTFx=\(String(format: "%.2f", summary.avgRTFx))x "
                + "(\(summary.samplesProcessed) samples)"
        )

        return (langResults, [summary])
    }

    /// Shared per-file inference loop used by both FLEURS and LibriSpeech paths.
    /// `onCheckpoint` is invoked with the running results array every
    /// `checkpointEvery` successful transcriptions; callers persist the
    /// partial run to disk so a crash mid-benchmark loses at most that many.
    private static func transcribeFiles(
        files: [BenchmarkAudioFile],
        language: CohereAsrConfig.Language,
        languageLabel: String,
        models: CoherePipeline.LoadedModels,
        pipeline: CoherePipeline,
        maxTokens: Int,
        repetitionPenalty: Float,
        noRepeatNgram: Int,
        checkpointEvery: Int = Int.max,
        onCheckpoint: (([CohereBenchmarkResult]) -> Void)? = nil
    ) async -> [CohereBenchmarkResult] {
        var results: [CohereBenchmarkResult] = []
        for (idx, file) in files.enumerated() {
            do {
                let samples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
                let duration = Double(samples.count) / Double(CohereAsrConfig.sampleRate)
                if duration > Double(CohereAsrConfig.maxAudioSeconds) {
                    logger.warning(
                        "  Skipping \(file.fileName) (\(String(format: "%.1f", duration))s > "
                            + "\(CohereAsrConfig.maxAudioSeconds)s single-chunk limit)"
                    )
                    continue
                }

                let result = try await pipeline.transcribe(
                    audio: samples,
                    models: models,
                    language: language,
                    maxNewTokens: maxTokens,
                    repetitionPenalty: repetitionPenalty,
                    noRepeatNgram: noRepeatNgram
                )

                let rtfx = duration / max(result.totalSeconds, 1e-9)
                let metrics = WERCalculator.calculateWERAndCER(
                    hypothesis: result.text,
                    reference: file.transcript
                )
                let werPct = metrics.wer * 100
                let cerPct = metrics.cer * 100

                results.append(
                    CohereBenchmarkResult(
                        language: languageLabel,
                        fileName: file.fileName,
                        reference: file.transcript,
                        hypothesis: result.text,
                        wer: werPct,
                        cer: cerPct,
                        duration: duration,
                        encoderSeconds: result.encoderSeconds,
                        decoderSeconds: result.decoderSeconds,
                        processingTime: result.totalSeconds,
                        rtfx: rtfx
                    ))

                logger.info(
                    "  [\(idx + 1)/\(files.count)] \(file.fileName) "
                        + "WER=\(String(format: "%.2f", werPct))% "
                        + "CER=\(String(format: "%.2f", cerPct))% "
                        + "RTFx=\(String(format: "%.2f", rtfx))x"
                )

                if let onCheckpoint, results.count % checkpointEvery == 0 {
                    onCheckpoint(results)
                }
            } catch {
                logger.error("  Failed on \(file.fileName): \(error)")
            }
        }
        return results
    }

    // MARK: - Helpers

    private struct BenchmarkAudioFile {
        let fileName: String
        let audioPath: URL
        let transcript: String
    }

    private static func collectFleursFiles(
        language: String,
        maxFiles: Int?,
        fleursDir: String
    ) throws -> [BenchmarkAudioFile] {
        let langDir = URL(fileURLWithPath: fleursDir).appendingPathComponent(language)

        guard FileManager.default.fileExists(atPath: langDir.path) else {
            throw NSError(
                domain: "CohereBenchmark",
                code: 1,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "FLEURS dataset not found for \(language) at \(langDir.path). "
                        + "Pass --auto-download to fetch, or --fleurs-dir to point at an existing copy."
                ]
            )
        }

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

    private static func collectLibriSpeechFiles(from directory: URL) throws -> [BenchmarkAudioFile] {
        var files: [BenchmarkAudioFile] = []
        let fm = FileManager.default
        let enumerator = fm.enumerator(at: directory, includingPropertiesForKeys: nil)

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

                if fm.fileExists(atPath: audioPath.path) {
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

    private static func summarize(
        language: String,
        results: [CohereBenchmarkResult]
    ) -> LanguageSummary {
        let n = results.count
        guard n > 0 else {
            return LanguageSummary(
                language: language,
                samplesProcessed: 0,
                avgWER: 0,
                avgCER: 0,
                avgRTFx: 0,
                totalDuration: 0,
                totalProcessing: 0
            )
        }
        let avgWER = results.map(\.wer).reduce(0, +) / Double(n)
        let avgCER = results.map(\.cer).reduce(0, +) / Double(n)
        let avgRTFx = results.map(\.rtfx).reduce(0, +) / Double(n)
        let totalDur = results.map(\.duration).reduce(0, +)
        let totalProc = results.map(\.processingTime).reduce(0, +)
        return LanguageSummary(
            language: language,
            samplesProcessed: n,
            avgWER: avgWER,
            avgCER: avgCER,
            avgRTFx: avgRTFx,
            totalDuration: totalDur,
            totalProcessing: totalProc
        )
    }

    private static func saveResults(
        allResults: [CohereBenchmarkResult],
        perLanguage: [LanguageSummary],
        to outputFile: String
    ) throws {
        struct Report: Codable {
            let perLanguage: [LanguageSummary]
            let results: [CohereBenchmarkResult]
        }
        let report = Report(perLanguage: perLanguage, results: allResults)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        try data.write(to: URL(fileURLWithPath: outputFile))
        logger.info("Results saved to: \(outputFile)")
    }

    private static func printFinalSummary(perLanguage: [LanguageSummary]) {
        // Avoid `String(format: "%s", swiftString)` — Swift's String maps to %@,
        // and on macOS 26 the Foundation runtime aborts on the mismatch.
        func row(_ language: String, _ samples: String, _ wer: String, _ cer: String, _ rtfx: String) -> String {
            language.padding(toLength: 14, withPad: " ", startingAt: 0)
                + " " + samples.leftPad(to: 8)
                + " " + wer.leftPad(to: 8)
                + " " + cer.leftPad(to: 8)
                + " " + rtfx.leftPad(to: 8)
        }

        logger.info(String(repeating: "=", count: 60))
        logger.info("COHERE TRANSCRIBE BENCHMARK SUMMARY")
        logger.info(String(repeating: "=", count: 60))
        logger.info(row("language", "samples", "WER%", "CER%", "RTFx"))
        for s in perLanguage {
            logger.info(
                row(
                    s.language,
                    "\(s.samplesProcessed)",
                    String(format: "%.2f", s.avgWER),
                    String(format: "%.2f", s.avgCER),
                    String(format: "%.2f", s.avgRTFx)
                ))
        }

        let totalSamples = perLanguage.map(\.samplesProcessed).reduce(0, +)
        guard totalSamples > 0 else { return }
        let avgWER =
            perLanguage.reduce(0.0) { $0 + $1.avgWER * Double($1.samplesProcessed) }
            / Double(totalSamples)
        let avgCER =
            perLanguage.reduce(0.0) { $0 + $1.avgCER * Double($1.samplesProcessed) }
            / Double(totalSamples)
        let avgRTFx =
            perLanguage.reduce(0.0) { $0 + $1.avgRTFx * Double($1.samplesProcessed) }
            / Double(totalSamples)
        logger.info(String(repeating: "-", count: 60))
        logger.info(
            row(
                "OVERALL",
                "\(totalSamples)",
                String(format: "%.2f", avgWER),
                String(format: "%.2f", avgCER),
                String(format: "%.2f", avgRTFx)
            ))
    }

    private static func printUsage() {
        logger.info(
            """

            Cohere Transcribe benchmark (FLEURS or LibriSpeech)

            Usage: fluidaudio cohere-benchmark [options]

            Model locations (choose one pattern):
                --model-dir <path>              Single dir with encoder + decoder + vocab.json
                --encoder-dir <path>            Encoder .mlmodelc dir (overrides --model-dir)
                --decoder-dir <path>            Decoder .mlmodelc dir (overrides --model-dir)
                --vocab-dir <path>              vocab.json dir (defaults to decoder-dir)

            Dataset:
                --dataset <name>                fleurs (default) or librispeech
                --subset <name>                 LibriSpeech subset (default: test-clean)
                --languages <codes>             FLEURS: comma-separated codes (default: en_us)
                --max-files <n>                 Cap samples processed (default: all)
                --fleurs-dir <path>             Local FLEURS cache root
                                                 (default: ~/Library/Application Support/FluidAudio/Datasets/fleurs)
                --auto-download                 Fetch missing dataset splits

            Decode:
                --max-tokens <n>                Max decoded tokens (default: 108)
                --repetition-penalty <f>        CTRL-style penalty, 1.0 disables (default: 1.1)
                --no-repeat-ngram <n>           Forbid repeating n-grams, 0 disables (default: 3)

            Compute units:
                --cpu-only                      Force CPU
                --cpu-gpu                       CPU + GPU (skip ANE)
                (default: all, includes ANE)

            Output:
                --output <file>                 JSON report path
                                                 (default: cohere_benchmark_results.json)
                --checkpoint-every <n>          Persist partial results every N files
                                                 (default: 100)

            Supported FLEURS codes (14 total):
                en_us, fr_fr, de_de, es_419, it_it, pt_br, nl_nl, pl_pl,
                el_gr, ar_eg, ja_jp, cmn_hans_cn, ko_kr, vi_vn

            Note:
                Cohere Transcribe is single-chunk with a 35s audio limit. Files
                exceeding that are skipped with a warning.

            Examples:
                # LibriSpeech test-clean, English, 100 utterances
                fluidaudio cohere-benchmark \\
                    --model-dir /path/to/q8 \\
                    --dataset librispeech \\
                    --subset test-clean \\
                    --max-files 100 \\
                    --auto-download

                # FLEURS, 3 languages, 20 samples each
                fluidaudio cohere-benchmark \\
                    --model-dir /path/to/q8 \\
                    --dataset fleurs \\
                    --languages en_us,fr_fr,ja_jp \\
                    --max-files 20 \\
                    --auto-download
            """
        )
    }
}

// MARK: - Supporting Types

struct CohereBenchmarkResult: Codable {
    let language: String
    let fileName: String
    let reference: String
    let hypothesis: String
    let wer: Double
    let cer: Double
    let duration: Double
    let encoderSeconds: Double
    let decoderSeconds: Double
    let processingTime: Double
    let rtfx: Double
}

extension String {
    fileprivate func leftPad(to width: Int) -> String {
        count >= width ? self : String(repeating: " ", count: width - count) + self
    }
}

struct LanguageSummary: Codable {
    let language: String
    let samplesProcessed: Int
    let avgWER: Double
    let avgCER: Double
    let avgRTFx: Double
    let totalDuration: Double
    let totalProcessing: Double
}
#endif
