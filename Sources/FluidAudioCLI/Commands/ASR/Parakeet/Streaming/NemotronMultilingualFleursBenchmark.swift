#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// FLEURS multilingual benchmark for the Nemotron Speech Streaming Multilingual
/// 0.6B model. Reuses `FLEURSBenchmark`'s dataset download / cache layout so the
/// same `~/Library/Application Support/FluidAudio/FLEURS/<lang>/` directories
/// are populated and shared with the Parakeet TDT benchmark.
///
/// Local-model-path-only: the multilingual model is not auto-downloaded.
/// Pass `--model-dir <path>` pointing at the directory containing
/// `metadata.json`, `tokenizer.json`, and the `.mlmodelc`/`.mlpackage` bundles.
public class NemotronMultilingualFleursBenchmark {
    private let logger = AppLogger(category: "NemotronMultilingualFleurs")

    public struct Config {
        var languages: [String]
        var samplesPerLanguage: Int
        var outputFile: String
        var cacheDir: String
        var modelDir: URL
        var debugMode: Bool
        /// HuggingFace dataset repo. Defaults to `FluidInference/fleurs-full`
        /// (30 languages including CJK / Arabic / Indic) instead of the
        /// European-only `FluidInference/fleurs` used by Parakeet TDT.
        var datasetRepo: String = "FluidInference/fleurs-full"
        /// When true, seed the decoder LSTM state with the lang-tag token id
        /// matching the current language before each session (Whisper-style
        /// hard language lock). Encoder still gets `prompt_id` as usual.
        var forcedPrefix: Bool = false
        /// Optional JSONL path. If set, writes one line per processed sample
        /// with raw hypothesis/reference and both English-normalized and basic-
        /// normalized variants plus per-sample WER under each. For debugging
        /// the gap between `normalize()` and `basicNormalize()`.
        var dumpSamplesPath: String? = nil
    }

    public struct LanguageResult {
        public let language: String
        public let promptLanguageCode: String
        public let wer: Double
        public let cer: Double
        public let rtfx: Double
        public let samplesProcessed: Int
        public let samplesSkipped: Int
        public let totalDuration: Double
        public let processingTime: Double
    }

    private let config: Config

    public init(config: Config) {
        self.config = config
    }

    /// Map a FLEURS language code (e.g. `en_us`) to the multilingual model's
    /// prompt-dictionary key format (e.g. `en-US`). Unknown codes are returned
    /// untouched and let `StreamingNemotronMultilingualAsrManager.setLanguage`
    /// fall back to the `default_prompt_id`.
    public static func fleursToMultilingualLanguage(_ fleursCode: String) -> String {
        switch fleursCode {
        case "cmn_hans_cn": return "zh-CN"
        case "es_419": return "es-ES"
        case "pt_br": return "pt-BR"
        case "ar_eg": return "ar-EG"
        default:
            let parts = fleursCode.split(separator: "_")
            if parts.count == 2 {
                return "\(parts[0])-\(parts[1].uppercased())"
            }
            return fleursCode
        }
    }

    /// Map a FLEURS language code to a `Locale` suitable for
    /// `NumberFormatter`'s `.spellOut` style. Returns nil for languages where
    /// digit-to-word ITN doesn't apply (English uses `normalize()` not
    /// `basicNormalize()`, CJK uses character-level scoring with `normalize()`
    /// — both bypass `basicNormalize`). Used to match NVIDIA's multilingual
    /// FLEURS scoring pipeline, which ITNs digits in the reference so the
    /// model's spelled-out output isn't penalized.
    public static func fleursToSpellOutLocale(_ fleursCode: String) -> Locale? {
        switch fleursCode {
        case "fr_fr": return Locale(identifier: "fr_FR")
        case "de_de": return Locale(identifier: "de_DE")
        case "es_419": return Locale(identifier: "es_419")
        case "it_it": return Locale(identifier: "it_IT")
        case "pt_br": return Locale(identifier: "pt_BR")
        default: return nil
        }
    }

    public func run() async throws -> [LanguageResult] {
        logger.info("Starting Nemotron Multilingual FLEURS Benchmark")
        logger.info(String(repeating: "=", count: 50))

        // Reuse the existing FLEURSBenchmark to download / load samples so
        // the cache directory layout is shared.
        let downloadConfig = FLEURSBenchmark.FLEURSConfig(
            languages: config.languages,
            samplesPerLanguage: config.samplesPerLanguage,
            outputFile: config.outputFile,
            cacheDir: config.cacheDir,
            debugMode: config.debugMode,
            datasetRepo: config.datasetRepo
        )
        let downloader = FLEURSBenchmark(config: downloadConfig)
        try await downloader.downloadFLEURS(languages: config.languages)
        let samples = try downloader.loadFLEURSSamples(languages: config.languages)
        if samples.isEmpty {
            logger.warning("No samples loaded. Aborting.")
            return []
        }

        logger.info("Loaded \(samples.count) samples across \(config.languages.count) languages")

        // Load the multilingual ASR manager once.
        let manager = StreamingNemotronMultilingualAsrManager()
        try await manager.loadModels(from: config.modelDir)
        if config.forcedPrefix {
            await manager.setForcedPrefix(true)
            logger.info("Forced-prefix decoding enabled (Whisper-style hard language lock)")
        }

        // Optional per-sample JSONL dump for normalizer debugging.
        var dumpHandle: FileHandle?
        if let dumpPath = config.dumpSamplesPath {
            let url = URL(fileURLWithPath: dumpPath)
            FileManager.default.createFile(atPath: url.path, contents: nil)
            dumpHandle = try? FileHandle(forWritingTo: url)
            logger.info("Per-sample dump: \(url.path)")
        }
        defer { try? dumpHandle?.close() }

        var results: [LanguageResult] = []
        let groups = Dictionary(grouping: samples, by: { $0.language })
        // Preserve user-specified language order in output.
        for lang in config.languages {
            guard let langSamples = groups[lang] else { continue }
            let promptLang = Self.fleursToMultilingualLanguage(lang)
            await manager.setLanguage(promptLang)
            await manager.reset()

            let result = try await runLanguage(
                manager: manager,
                language: lang,
                promptLanguageCode: promptLang,
                samples: langSamples,
                dumpHandle: dumpHandle
            )
            results.append(result)

            let skippedInfo = result.samplesSkipped > 0 ? ", \(result.samplesSkipped) skipped" : ""
            logger.info(
                "\(lang) [\(promptLang)]: WER=\(String(format: "%.1f", result.wer * 100))%, CER=\(String(format: "%.1f", result.cer * 100))%, RTFx=\(String(format: "%.1f", result.rtfx))x (\(result.samplesProcessed) processed\(skippedInfo))"
            )
        }

        return results
    }

    private func runLanguage(
        manager: StreamingNemotronMultilingualAsrManager,
        language: String,
        promptLanguageCode: String,
        samples: [FLEURSBenchmark.FLEURSSample],
        dumpHandle: FileHandle?
    ) async throws -> LanguageResult {
        var totalWER = 0.0
        var totalCER = 0.0
        var totalDuration = 0.0
        var totalProcessingTime = 0.0
        var processed = 0
        var skipped = 0

        let audioConverter = AudioConverter()

        for sample in samples {
            guard FileManager.default.fileExists(atPath: sample.audioPath) else {
                logger.warning("Audio missing: \(sample.audioPath)")
                skipped += 1
                continue
            }

            let audioURL = URL(fileURLWithPath: sample.audioPath)
            let audioSamples: [Float]
            do {
                audioSamples = try audioConverter.resampleAudioFile(path: sample.audioPath)
            } catch {
                logger.warning("Resample failed for \(sample.sampleId): \(error.localizedDescription)")
                skipped += 1
                continue
            }
            let audioDuration = Double(audioSamples.count) / 16000.0

            do {
                let audioFile = try AVAudioFile(forReading: audioURL)
                guard
                    let buffer = AVAudioPCMBuffer(
                        pcmFormat: audioFile.processingFormat,
                        frameCapacity: AVAudioFrameCount(audioFile.length)
                    )
                else {
                    logger.warning("Buffer alloc failed for \(sample.sampleId)")
                    skipped += 1
                    continue
                }
                try audioFile.read(into: buffer)

                let startTime = Date()
                _ = try await manager.process(audioBuffer: buffer)
                let hypothesis = try await manager.finish()
                let processingTime = Date().timeIntervalSince(startTime)

                if !sample.transcription.isEmpty {
                    // For CJK / no-space scripts, FLEURS word-tokenized WER
                    // is meaningless (hypothesis and reference disagree on
                    // segmentation). Route through character-level scoring
                    // so the reported "WER" matches the community standard
                    // (ESPnet / Whisper paper) for these languages.
                    let metrics:
                        (
                            wer: Double, cer: Double, insertions: Int, deletions: Int, substitutions: Int,
                            totalWords: Int, totalCharacters: Int
                        )
                    if WERCalculator.isCJKLanguage(language) {
                        metrics = WERCalculator.calculateCJKMetrics(
                            hypothesis: hypothesis,
                            reference: sample.transcription
                        )
                    } else if language.lowercased().hasPrefix("en") {
                        // English: apply the full HF/Whisper EnglishTextNormalizer
                        // equivalent (contraction expansion, number folding,
                        // British→American, abbreviations) — matches NVIDIA's
                        // pipeline for English FLEURS scoring.
                        metrics = WERCalculator.calculateWERAndCER(
                            hypothesis: hypothesis,
                            reference: sample.transcription
                        )
                    } else {
                        // Non-English Latin-script langs (fr/de/es/it/pt/...):
                        // apply the BasicTextNormalizer-equivalent (lowercase,
                        // NFKD, strip punctuation/symbols, keep diacritics)
                        // plus an ITN pass (digits → spelled-out via
                        // NumberFormatter) so the reference's literal "1976"
                        // is comparable to the model's "mille neuf cent
                        // soixante seize". This matches NeMo / NVIDIA's
                        // multilingual leaderboard scoring; without ITN, the
                        // ~22-25% of FLEURS samples that contain digits in
                        // the reference get heavily penalized.
                        let locale = Self.fleursToSpellOutLocale(language)
                        metrics = WERCalculator.calculateBasicWERAndCER(
                            hypothesis: hypothesis,
                            reference: sample.transcription,
                            spellOutLocale: locale
                        )
                    }
                    totalWER += metrics.wer
                    totalCER += metrics.cer

                    // Per-sample dump: capture raw + both-normalizer variants
                    // + per-sample WER under each so we can diagnose why the
                    // basic normalizer raises WER on non-English vs the
                    // English normalizer.
                    if let handle = dumpHandle {
                        let engMetrics = WERCalculator.calculateWERAndCER(
                            hypothesis: hypothesis,
                            reference: sample.transcription
                        )
                        let basicMetrics = WERCalculator.calculateBasicWERAndCER(
                            hypothesis: hypothesis,
                            reference: sample.transcription
                        )
                        let spellLocale = Self.fleursToSpellOutLocale(language)
                        let basicItnMetrics = WERCalculator.calculateBasicWERAndCER(
                            hypothesis: hypothesis,
                            reference: sample.transcription,
                            spellOutLocale: spellLocale
                        )
                        let row: [String: Any] = [
                            "sampleId": sample.sampleId,
                            "language": language,
                            "hyp_raw": hypothesis,
                            "ref_raw": sample.transcription,
                            "hyp_eng": TextNormalizer.normalize(hypothesis),
                            "ref_eng": TextNormalizer.normalize(sample.transcription),
                            "hyp_basic": TextNormalizer.basicNormalize(hypothesis),
                            "ref_basic": TextNormalizer.basicNormalize(sample.transcription),
                            "hyp_basic_itn": TextNormalizer.basicNormalize(
                                hypothesis, spellOutLocale: spellLocale),
                            "ref_basic_itn": TextNormalizer.basicNormalize(
                                sample.transcription, spellOutLocale: spellLocale),
                            "wer_eng": engMetrics.wer,
                            "wer_basic": basicMetrics.wer,
                            "wer_basic_itn": basicItnMetrics.wer,
                            "ins_eng": engMetrics.insertions,
                            "del_eng": engMetrics.deletions,
                            "sub_eng": engMetrics.substitutions,
                            "ins_basic": basicMetrics.insertions,
                            "del_basic": basicMetrics.deletions,
                            "sub_basic": basicMetrics.substitutions,
                            "ins_basic_itn": basicItnMetrics.insertions,
                            "del_basic_itn": basicItnMetrics.deletions,
                            "sub_basic_itn": basicItnMetrics.substitutions,
                        ]
                        if let data = try? JSONSerialization.data(withJSONObject: row, options: []) {
                            handle.write(data)
                            handle.write(Data([0x0A]))  // newline
                        }
                    }
                }

                totalDuration += audioDuration
                totalProcessingTime += processingTime
                processed += 1

                if config.debugMode {
                    let detected = await manager.detectedLanguage() ?? "(none)"
                    logger.debug("  [\(sample.sampleId)] detected=\(detected)")
                    logger.debug("    Hypothesis: \(hypothesis)")
                    if !sample.transcription.isEmpty {
                        logger.debug("    Reference:  \(sample.transcription)")
                    }
                }

                // Reset session state between samples; keep language setting.
                await manager.reset()

            } catch {
                logger.warning("Transcription error for \(sample.sampleId): \(error.localizedDescription)")
                skipped += 1
                // Try to keep going with a fresh state.
                await manager.reset()
            }
        }

        guard processed > 0 else {
            throw ASRError.processingFailed("Benchmark failed for \(language): no samples processed")
        }

        let avgWER = totalWER / Double(processed)
        let avgCER = totalCER / Double(processed)
        let rtfx = totalProcessingTime > 0 ? totalDuration / totalProcessingTime : 0.0

        return LanguageResult(
            language: language,
            promptLanguageCode: promptLanguageCode,
            wer: avgWER,
            cer: avgCER,
            rtfx: rtfx,
            samplesProcessed: processed,
            samplesSkipped: skipped,
            totalDuration: totalDuration,
            processingTime: totalProcessingTime
        )
    }

    public func saveResults(_ results: [LanguageResult], to outputPath: String) throws {
        func sanitize(_ value: Double) -> Double {
            value.isNaN || value.isInfinite ? 0.0 : value
        }
        let output: [String: Any] = [
            "benchmark": "Nemotron Multilingual FLEURS",
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "config": [
                "languages": config.languages,
                "samplesPerLanguage": config.samplesPerLanguage,
                "modelDir": config.modelDir.path,
            ],
            "results": results.map { r in
                [
                    "language": r.language,
                    "promptLanguageCode": r.promptLanguageCode,
                    "wer": sanitize(r.wer),
                    "cer": sanitize(r.cer),
                    "rtfx": sanitize(r.rtfx),
                    "samplesProcessed": r.samplesProcessed,
                    "samplesSkipped": r.samplesSkipped,
                    "totalDuration": sanitize(r.totalDuration),
                    "processingTime": sanitize(r.processingTime),
                ]
            },
            "summary": [
                "averageWER": sanitize(results.reduce(0.0) { $0 + $1.wer } / Double(max(results.count, 1))),
                "averageCER": sanitize(results.reduce(0.0) { $0 + $1.cer } / Double(max(results.count, 1))),
                "averageRTFx": sanitize(results.reduce(0.0) { $0 + $1.rtfx } / Double(max(results.count, 1))),
                "totalSamples": results.reduce(0) { $0 + $1.samplesProcessed },
                "totalSkipped": results.reduce(0) { $0 + $1.samplesSkipped },
                "totalDuration": sanitize(results.reduce(0.0) { $0 + $1.totalDuration }),
                "totalProcessingTime": sanitize(results.reduce(0.0) { $0 + $1.processingTime }),
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: outputPath))
    }
}

extension NemotronMultilingualFleursBenchmark {

    public static func runCLI(arguments: [String]) async {
        let logger = AppLogger(category: "NemotronMultilingualFleurs")

        // Defaults: n=100 samples, 5 languages spread across the multilingual model.
        var languages: [String] = ["en_us", "fr_fr", "de_de", "es_419", "ja_jp"]
        var samplesPerLanguage = 100
        var outputFile = "nemotron_multilingual_fleurs_results.json"
        var cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/FluidAudio/FLEURS-full").path
        var datasetRepo = "FluidInference/fleurs-full"
        var modelDir: URL?
        var debugMode = false
        var forcedPrefix = false
        var dumpSamplesPath: String?

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--languages":
                if i + 1 < arguments.count {
                    let arg = arguments[i + 1]
                    languages = arg.split(separator: ",").map(String.init)
                    i += 1
                }
            case "--samples":
                if i + 1 < arguments.count {
                    if arguments[i + 1].lowercased() == "all" {
                        samplesPerLanguage = Int.max
                    } else if let v = Int(arguments[i + 1]) {
                        samplesPerLanguage = v
                    }
                    i += 1
                }
            case "--output":
                if i + 1 < arguments.count {
                    outputFile = arguments[i + 1]
                    i += 1
                }
            case "--cache-dir":
                if i + 1 < arguments.count {
                    cacheDir = arguments[i + 1]
                    i += 1
                }
            case "--dataset-repo":
                if i + 1 < arguments.count {
                    datasetRepo = arguments[i + 1]
                    i += 1
                }
            case "--model-dir", "-m":
                if i + 1 < arguments.count {
                    modelDir = URL(fileURLWithPath: arguments[i + 1])
                    i += 1
                }
            case "--debug":
                debugMode = true
            case "--forced-prefix":
                forcedPrefix = true
            case "--dump-samples":
                if i + 1 < arguments.count {
                    dumpSamplesPath = arguments[i + 1]
                    i += 1
                }
            case "--help", "-h":
                printUsage()
                return
            default:
                logger.warning("Unknown option: \(arguments[i])")
            }
            i += 1
        }

        guard let modelDir = modelDir else {
            logger.error("Missing --model-dir. The multilingual model is not auto-downloaded.")
            printUsage()
            return
        }

        logger.info("Nemotron Multilingual FLEURS Benchmark")
        logger.info(String(repeating: "=", count: 50))
        logger.info("Languages: \(languages.joined(separator: ", "))")
        logger.info("Samples per language: \(samplesPerLanguage == Int.max ? "all" : String(samplesPerLanguage))")
        logger.info("Model dir: \(modelDir.path)")
        logger.info("Dataset repo: \(datasetRepo)")
        logger.info("Cache dir: \(cacheDir)")
        logger.info("Output: \(outputFile)")

        let config = Config(
            languages: languages,
            samplesPerLanguage: samplesPerLanguage,
            outputFile: outputFile,
            cacheDir: cacheDir,
            modelDir: modelDir,
            debugMode: debugMode,
            datasetRepo: datasetRepo,
            forcedPrefix: forcedPrefix,
            dumpSamplesPath: dumpSamplesPath
        )

        let benchmark = NemotronMultilingualFleursBenchmark(config: config)

        do {
            let results = try await benchmark.run()
            try benchmark.saveResults(results, to: outputFile)
            logger.info("Results saved to \(outputFile)")

            // Print summary table
            print("")
            print(
                "Language".padding(toLength: 12, withPad: " ", startingAt: 0) + " | "
                    + "Prompt".padding(toLength: 8, withPad: " ", startingAt: 0) + " | "
                    + "WER%".padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                    + "CER%".padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                    + "RTFx".padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                    + "Duration".padding(toLength: 9, withPad: " ", startingAt: 0) + " | "
                    + "Processed".padding(toLength: 9, withPad: " ", startingAt: 0) + " | "
                    + "Skipped"
            )
            print(String(repeating: "-", count: 80))

            for r in results {
                let werStr = String(format: "%.1f", r.wer * 100)
                let cerStr = String(format: "%.1f", r.cer * 100)
                let rtfxStr = String(format: "%.1f", r.rtfx)
                let durStr = String(format: "%.1fs", r.totalDuration)
                let procStr = String(r.samplesProcessed)
                let skipStr = r.samplesSkipped > 0 ? String(r.samplesSkipped) : "-"
                print(
                    r.language.padding(toLength: 12, withPad: " ", startingAt: 0) + " | "
                        + r.promptLanguageCode.padding(toLength: 8, withPad: " ", startingAt: 0) + " | "
                        + werStr.padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                        + cerStr.padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                        + rtfxStr.padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                        + durStr.padding(toLength: 9, withPad: " ", startingAt: 0) + " | "
                        + procStr.padding(toLength: 9, withPad: " ", startingAt: 0) + " | "
                        + skipStr
                )
            }

            if !results.isEmpty {
                let avgWER = results.reduce(0.0) { $0 + $1.wer } / Double(results.count)
                let avgCER = results.reduce(0.0) { $0 + $1.cer } / Double(results.count)
                let avgRTFx = results.reduce(0.0) { $0 + $1.rtfx } / Double(results.count)
                print(String(repeating: "-", count: 80))
                print(
                    "AVERAGE".padding(toLength: 12, withPad: " ", startingAt: 0) + " | "
                        + "—".padding(toLength: 8, withPad: " ", startingAt: 0) + " | "
                        + String(format: "%.1f", avgWER * 100).padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                        + String(format: "%.1f", avgCER * 100).padding(toLength: 6, withPad: " ", startingAt: 0) + " | "
                        + String(format: "%.1f", avgRTFx).padding(toLength: 6, withPad: " ", startingAt: 0)
                )
            }
        } catch {
            logger.error("Benchmark failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    private static func printUsage() {
        let logger = AppLogger(category: "NemotronMultilingualFleurs")
        logger.info(
            """

            Nemotron Multilingual FLEURS Benchmark Usage:
                fluidaudio nemotron-multilingual-benchmark --model-dir <path> [options]

            Required:
                --model-dir, -m <path>   Path to multilingual model directory
                                         (must contain metadata.json, tokenizer.json,
                                         encoder.mlmodelc or encoder.mlpackage, etc.)

            Options:
                --languages <list>       Comma-separated FLEURS codes (default: en_us,fr_fr,de_de,es_419,ja_jp)
                --samples <n|all>        Samples per language (default: 100)
                --output <file>          Output JSON file (default: nemotron_multilingual_fleurs_results.json)
                --cache-dir <path>       FLEURS dataset cache (default: ~/Library/Application Support/FluidAudio/FLEURS)
                --dump-samples <path>    Write per-sample JSONL with raw + English/basic
                                         normalized hyp/ref + per-sample WER under each
                                         (for normalizer debugging)
                --debug                  Verbose logging
                --help, -h               Show this help

            Examples:
                # Default run: 5 languages × 100 samples
                fluidaudio nemotron-multilingual-benchmark --model-dir ~/my-multilingual-model

                # Custom language mix
                fluidaudio nemotron-multilingual-benchmark \\
                    --model-dir ~/my-multilingual-model \\
                    --languages en_us,zh_cn,ja_jp \\
                    --samples 50

            """
        )
    }
}

#endif
