#if os(macOS)
import AVFoundation
import CoreML
import FluidAudio
import Foundation

/// FLEURS benchmark for the corrected mixed-precision Cohere pipeline
/// (INT8 encoder + FP16 decoder, or any single-precision combo) using
/// `CohereFixedPipeline` with the bug fixes applied (filterbank mel, fp16-safe
/// cross-attention mask, repetition penalty, no-repeat-ngram, byte-fallback
/// detokenization).
enum CohereMixedBenchmark {
    private static let logger = AppLogger(category: "CohereMixedBenchmark")

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
        var languages: [String] = ["en_us"]
        var maxFiles: Int?
        var fleursDir: String?
        var outputFile = "cohere_mixed_benchmark_results.json"
        var maxTokens = 108
        var repetitionPenalty: Float = 1.1
        var noRepeatNgram = 3
        var computeUnits: MLComputeUnits = .all
        var autoDownload = false

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
            case "--cpu-only":
                computeUnits = .cpuOnly
            case "--cpu-gpu":
                computeUnits = .cpuAndGPU
            case "--auto-download":
                autoDownload = true
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
            logger.error("Cohere mixed benchmark requires macOS 14 or later")
            exit(1)
        }

        logger.info("Cohere Mixed-Precision FLEURS Benchmark")
        logger.info("  Encoder dir:      \(encDir)")
        logger.info("  Decoder dir:      \(decDir)")
        logger.info("  Vocab dir:        \(vocDir)")
        logger.info("  Languages:        \(languages.joined(separator: ", "))")
        logger.info("  Max files/lang:   \(maxFiles?.description ?? "all")")
        logger.info("  Max tokens:       \(maxTokens)")
        logger.info("  Rep penalty:      \(repetitionPenalty)")
        logger.info("  No-repeat-ngram:  \(noRepeatNgram)")
        logger.info("  Auto-download:    \(autoDownload)")

        do {
            // Load models once
            let loadStart = CFAbsoluteTimeGetCurrent()
            let models = try await CohereFixedPipeline.loadModels(
                encoderDir: URL(fileURLWithPath: encDir),
                decoderDir: URL(fileURLWithPath: decDir),
                vocabDir: URL(fileURLWithPath: vocDir),
                computeUnits: computeUnits
            )
            logger.info(
                "Models loaded in \(String(format: "%.2f", CFAbsoluteTimeGetCurrent() - loadStart))s")

            let pipeline = CohereFixedPipeline()

            // Resolve FLEURS cache directory
            let fleursCacheDir =
                fleursDir
                ?? NSHomeDirectory() + "/Library/Application Support/FluidAudio/Datasets/fleurs"

            // Optionally auto-download FLEURS splits we need
            if autoDownload {
                let supportedCodes = languages.filter { fleursToCohereLanguage.keys.contains($0) }
                let fleurs = FLEURSBenchmark(
                    config: FLEURSBenchmark.FLEURSConfig(
                        languages: supportedCodes,
                        samplesPerLanguage: maxFiles ?? 100,
                        outputFile: "/dev/null",
                        cacheDir: fleursCacheDir,
                        debugMode: false
                    ))
                try await fleurs.downloadFLEURS(languages: supportedCodes)
            }

            var allResults: [CohereMixedBenchmarkResult] = []
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

                var langResults: [CohereMixedBenchmarkResult] = []
                for (idx, file) in files.enumerated() {
                    do {
                        let samples = try AudioConverter().resampleAudioFile(path: file.audioPath.path)
                        let duration = Double(samples.count) / Double(CohereAsrConfig.sampleRate)

                        let result = try await pipeline.transcribe(
                            audio: samples,
                            models: models,
                            language: cohereLang,
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

                        langResults.append(
                            CohereMixedBenchmarkResult(
                                language: langCode,
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
                    } catch {
                        logger.error("  Failed on \(file.fileName): \(error)")
                    }
                }

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
                domain: "CohereMixedBenchmark",
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

    private static func summarize(
        language: String,
        results: [CohereMixedBenchmarkResult]
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
        allResults: [CohereMixedBenchmarkResult],
        perLanguage: [LanguageSummary],
        to outputFile: String
    ) throws {
        struct Report: Codable {
            let perLanguage: [LanguageSummary]
            let results: [CohereMixedBenchmarkResult]
        }
        let report = Report(perLanguage: perLanguage, results: allResults)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        try data.write(to: URL(fileURLWithPath: outputFile))
        logger.info("Results saved to: \(outputFile)")
    }

    private static func printFinalSummary(perLanguage: [LanguageSummary]) {
        logger.info(String(repeating: "=", count: 60))
        logger.info("COHERE MIXED-PRECISION FLEURS SUMMARY")
        logger.info(String(repeating: "=", count: 60))
        logger.info(
            String(
                format: "%-14s %8s %8s %8s %8s",
                "language", "samples", "WER%", "CER%", "RTFx"
            ))
        for s in perLanguage {
            logger.info(
                String(
                    format: "%-14s %8d %8.2f %8.2f %8.2f",
                    s.language, s.samplesProcessed, s.avgWER, s.avgCER, s.avgRTFx
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
            String(
                format: "%-14s %8d %8.2f %8.2f %8.2f",
                "OVERALL", totalSamples, avgWER, avgCER, avgRTFx
            ))
    }

    private static func printUsage() {
        logger.info(
            """

            Cohere Transcribe — mixed-precision FLEURS benchmark

            Usage: fluidaudio cohere-mixed-benchmark [options]

            Model locations (choose one pattern):
                --model-dir <path>              Single dir with encoder + decoder + vocab.json
                --encoder-dir <path>            (mixed) INT8 or FP16 encoder .mlmodelc dir
                --decoder-dir <path>            (mixed) INT8 or FP16 decoder .mlmodelc dir
                --vocab-dir <path>              vocab.json dir (defaults to decoder-dir)

            Dataset:
                --languages <codes>             Comma-separated FLEURS codes (default: en_us)
                --max-files <n>                 Cap per-language samples (default: all)
                --fleurs-dir <path>             Local FLEURS cache root
                                                 (default: ~/Library/Application Support/FluidAudio/Datasets/fleurs)
                --auto-download                 Fetch missing language splits from HuggingFace

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
                                                 (default: cohere_mixed_benchmark_results.json)

            Supported FLEURS codes (14 total):
                en_us, fr_fr, de_de, es_419, it_it, pt_br, nl_nl, pl_pl,
                el_gr, ar_eg, ja_jp, cmn_hans_cn, ko_kr, vi_vn

            Examples:
                # Mixed INT8 encoder + FP16 decoder across 3 languages, 20 samples each
                fluidaudio cohere-mixed-benchmark \\
                    --encoder-dir /path/to/q8 \\
                    --decoder-dir /path/to/f16 \\
                    --vocab-dir  /path/to/f16 \\
                    --languages en_us,fr_fr,ja_jp \\
                    --max-files 20 \\
                    --auto-download

                # Single-precision FP16, all English samples, save report
                fluidaudio cohere-mixed-benchmark \\
                    --model-dir /path/to/f16 \\
                    --languages en_us \\
                    --output cohere_fp16_en.json
            """
        )
    }
}

// MARK: - Supporting Types

struct CohereMixedBenchmarkResult: Codable {
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
