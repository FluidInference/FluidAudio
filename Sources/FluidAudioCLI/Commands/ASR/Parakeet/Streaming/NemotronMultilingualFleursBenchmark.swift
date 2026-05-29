#if os(macOS)
import AVFoundation
import CoreML
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
        /// Which benchmark dataset to evaluate against (FLEURS, MCV-17, MLS).
        /// Default `.fleurs` preserves backward compatibility with existing
        /// invocations.
        var dataset: MultilingualBenchmarkDataset = .fleurs
        /// HuggingFace FLEURS dataset repo override. Only consulted when
        /// `dataset == .fleurs`. Defaults to `FluidInference/fleurs-full`
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
        /// Optional prompt-code override (e.g. "es-US", "pt-PT"). When set,
        /// bypasses the FLEURS-code → prompt-code mapping and feeds this code
        /// directly to `manager.setLanguage(_:)`. Used for regional-prompt A/Bs
        /// (e.g. running MLS pt with "pt-PT" instead of the auto-derived "pt-BR").
        var promptOverride: String? = nil
        /// Optional compute-units override for the ASR manager's MLModelConfiguration.
        /// Default (nil) uses MLModelConfigurationUtils.defaultConfiguration() = .cpuAndNeuralEngine.
        /// Use `.cpuAndGPU` for INT4 builds (ANE compilation hangs on macOS 26.5 with int4 affine_dequantize).
        var computeUnits: MLComputeUnits? = nil
        /// LibriSpeech subset, only used when `dataset == .librispeech`.
        /// One of "test-clean" (default), "test-other", "dev-clean", "dev-other".
        var librispeechSubset: String = "test-clean"
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

        // Download + load samples for the configured dataset. All three
        // loaders normalize to the LibriSpeech-style on-disk layout
        // (`<cache>/<lang>/<id>.{wav|mp3|flac}` + `<lang>.trans.txt`) so we
        // can reuse `FLEURSBenchmark.loadFLEURSSamples` to produce the
        // `[FLEURSSample]` consumed below regardless of dataset.
        let samples: [FLEURSBenchmark.FLEURSSample]
        switch config.dataset {
        case .fleurs:
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
            samples = try downloader.loadFLEURSSamples(languages: config.languages)
        case .mcv:
            try await HFDatasetsServerDownloader.downloadMCV(
                languages: config.languages,
                cacheRoot: URL(fileURLWithPath: config.cacheDir),
                samplesPerLanguage: config.samplesPerLanguage
            )
            samples = try loadHFDatasetsServerSamples(
                cacheRoot: URL(fileURLWithPath: config.cacheDir),
                languages: config.languages,
                audioExtension: MultilingualBenchmarkDataset.mcv.audioExtension
            )
        case .mls:
            try await HFDatasetsServerDownloader.downloadMLS(
                languages: config.languages,
                cacheRoot: URL(fileURLWithPath: config.cacheDir),
                samplesPerLanguage: config.samplesPerLanguage
            )
            samples = try loadHFDatasetsServerSamples(
                cacheRoot: URL(fileURLWithPath: config.cacheDir),
                languages: config.languages,
                audioExtension: MultilingualBenchmarkDataset.mls.audioExtension
            )
        case .librispeech:
            samples = try loadLibriSpeechSamples(
                cacheRoot: URL(fileURLWithPath: config.cacheDir),
                subset: config.librispeechSubset,
                samplesPerLanguage: config.samplesPerLanguage
            )
        case .earnings22:
            samples = try loadEarnings22Samples(
                cacheRoot: URL(fileURLWithPath: config.cacheDir),
                samplesPerLanguage: config.samplesPerLanguage
            )
        }
        if samples.isEmpty {
            logger.warning("No samples loaded. Aborting.")
            return []
        }

        logger.info("Loaded \(samples.count) samples across \(config.languages.count) languages")

        // Load the multilingual ASR manager once.
        var mlConfig: MLModelConfiguration? = nil
        if let units = config.computeUnits {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = units
            mlConfig = cfg
            logger.info("Forcing computeUnits = \(units)")
        }
        let manager = StreamingNemotronMultilingualAsrManager(configuration: mlConfig)
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
            let promptLang = config.promptOverride ?? Self.fleursToMultilingualLanguage(lang)
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
                // Predecoded-PCM bench mode: `audioSamples` is already
                // resampled to 16 kHz by `audioConverter.resampleAudioFile`
                // above. Feed it directly via `process(samples:)` to bypass
                // the redundant AVAudioFile open + AVAudioPCMBuffer alloc +
                // second `resampleBuffer` resample inside
                // `process(audioBuffer:)`. Isolates model wall-time from
                // file-IO/resample harness overhead.
                let startTime = Date()
                _ = try await manager.process(samples: audioSamples)
                let hypothesis = try await manager.finish()
                let processingTime = Date().timeIntervalSince(startTime)
                // Capture diagnostic stats from this finish() call before any
                // subsequent reset() clears them.
                let decodeStats = await manager.lastDecodeStats()

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
                            "audio_duration": audioDuration,
                            "accumulated_token_count": decodeStats.tokenCount,
                            "detected_language": decodeStats.detectedLanguage ?? NSNull(),
                            "processed_chunks": decodeStats.processedChunks,
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

    /// Load samples from the Earnings22 KWS dataset (argmaxinc/contextual-earnings22).
    /// Layout: `<cacheRoot>/test-dataset/<id>_chunk<N>.wav` plus
    /// `<id>_chunk<N>.text.txt` siblings. Populate via
    /// `fluidaudio download --dataset earnings22-kws`.
    /// Samples are returned in natural-sort order (by call id, then chunk).
    private func loadEarnings22Samples(
        cacheRoot: URL,
        samplesPerLanguage: Int
    ) throws -> [FLEURSBenchmark.FLEURSSample] {
        let dataDir = cacheRoot.appendingPathComponent("test-dataset")
        guard FileManager.default.fileExists(atPath: dataDir.path) else {
            logger.warning(
                "Earnings22 not found at \(dataDir.path). Run "
                    + "`fluidaudio download --dataset earnings22-kws` first."
            )
            return []
        }
        let fm = FileManager.default
        let wavs = ((try? fm.contentsOfDirectory(at: dataDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension.lowercased() == "wav" }
            .sorted { a, b in
                // Natural sort: by call id then by chunk index. File names
                // look like "4483857_chunk_0.wav" or "4471961_chunk91.wav"
                // (some have the underscore between "chunk" and the number,
                // some don't). Sort by the (id, chunkIdx) tuple.
                func parse(_ url: URL) -> (String, Int) {
                    let stem = url.deletingPathExtension().lastPathComponent
                    let parts = stem.split(separator: "_", omittingEmptySubsequences: false).map(String.init)
                    let id = parts.first ?? stem
                    let idx = Int(parts.last ?? "0") ?? 0
                    return (id, idx)
                }
                let pa = parse(a)
                let pb = parse(b)
                if pa.0 != pb.0 { return pa.0 < pb.0 }
                return pa.1 < pb.1
            }
        let take = samplesPerLanguage == Int.max ? wavs.count : min(samplesPerLanguage, wavs.count)
        var out: [FLEURSBenchmark.FLEURSSample] = []
        for wav in wavs.prefix(take) {
            let id = wav.deletingPathExtension().lastPathComponent
            let refURL = wav.deletingPathExtension().appendingPathExtension("text.txt")
            let reference = (try? String(contentsOf: refURL).trimmingCharacters(in: .whitespacesAndNewlines)) ?? ""
            if reference.isEmpty {
                // Skip chunks whose reference is missing; the KWS variant
                // omits the .text.txt for some chunks.
                continue
            }
            out.append(
                FLEURSBenchmark.FLEURSSample(
                    audioPath: wav.path,
                    transcription: reference,
                    language: "en_us",
                    sampleId: id
                )
            )
        }
        logger.info(
            "Loaded \(out.count) Earnings22 samples"
                + (out.count < wavs.count ? " (limited by --samples or missing refs)" : "")
        )
        return out
    }

    /// Load samples from an on-disk LibriSpeech subset (test-clean by default).
    /// Layout: `<cacheRoot>/<subset>/<speaker>/<chapter>/<id>.flac` and
    /// `<speaker>-<chapter>.trans.txt` in the same directory. Use the
    /// `download --dataset librispeech-test-clean` command to populate this.
    private func loadLibriSpeechSamples(
        cacheRoot: URL,
        subset: String,
        samplesPerLanguage: Int
    ) throws -> [FLEURSBenchmark.FLEURSSample] {
        let subsetDir = cacheRoot.appendingPathComponent(subset)
        guard FileManager.default.fileExists(atPath: subsetDir.path) else {
            logger.warning(
                "LibriSpeech subset not found at \(subsetDir.path). Run "
                    + "`fluidaudio download --dataset librispeech-\(subset)` first."
            )
            return []
        }
        let fm = FileManager.default
        var transcripts: [String: String] = [:]
        var flacs: [URL] = []
        if let walker = fm.enumerator(at: subsetDir, includingPropertiesForKeys: nil) {
            for case let url as URL in walker {
                let ext = url.pathExtension.lowercased()
                if ext == "flac" {
                    flacs.append(url)
                } else if url.lastPathComponent.hasSuffix(".trans.txt") {
                    let content = (try? String(contentsOf: url)) ?? ""
                    for line in content.components(separatedBy: .newlines) where !line.isEmpty {
                        let parts = line.split(separator: " ", maxSplits: 1)
                        if parts.count == 2 {
                            transcripts[String(parts[0])] = String(parts[1])
                        }
                    }
                }
            }
        }
        flacs.sort { $0.lastPathComponent < $1.lastPathComponent }
        let take = samplesPerLanguage == Int.max ? flacs.count : min(samplesPerLanguage, flacs.count)
        var out: [FLEURSBenchmark.FLEURSSample] = []
        for url in flacs.prefix(take) {
            let id = url.deletingPathExtension().lastPathComponent
            out.append(
                FLEURSBenchmark.FLEURSSample(
                    audioPath: url.path,
                    transcription: transcripts[id] ?? "",
                    language: "en_us",
                    sampleId: id
                )
            )
        }
        logger.info(
            "Loaded \(out.count) LibriSpeech \(subset) samples"
                + (out.count < flacs.count ? " (limited by --samples)" : "")
        )
        return out
    }

    /// Load samples produced by `HFDatasetsServerDownloader` (MCV / MLS).
    /// Layout mirrors FLEURS: per-language directory containing audio files
    /// and a `<lang>.trans.txt` LibriSpeech-style transcript file.
    private func loadHFDatasetsServerSamples(
        cacheRoot: URL,
        languages: [String],
        audioExtension: String
    ) throws -> [FLEURSBenchmark.FLEURSSample] {
        var out: [FLEURSBenchmark.FLEURSSample] = []
        for lang in languages {
            let langDir = cacheRoot.appendingPathComponent(lang)
            guard FileManager.default.fileExists(atPath: langDir.path) else {
                logger.warning("No data found for \(lang). Skipping.")
                continue
            }
            let transFile = langDir.appendingPathComponent("\(lang).trans.txt")
            var transcripts: [String: String] = [:]
            if FileManager.default.fileExists(atPath: transFile.path) {
                let content = (try? String(contentsOf: transFile)) ?? ""
                for line in content.components(separatedBy: .newlines) where !line.isEmpty {
                    let parts = line.split(separator: " ", maxSplits: 1)
                    if parts.count >= 1 {
                        let id = String(parts[0])
                        transcripts[id] = parts.count > 1 ? String(parts[1]) : ""
                    }
                }
            }
            let files =
                (try? FileManager.default.contentsOfDirectory(at: langDir, includingPropertiesForKeys: nil)) ?? []
            let audioFiles =
                files
                .filter { $0.pathExtension.lowercased() == audioExtension.lowercased() }
                .sorted { $0.lastPathComponent < $1.lastPathComponent }
                .prefix(config.samplesPerLanguage == Int.max ? Int.max : config.samplesPerLanguage)
            for url in audioFiles {
                let id = url.deletingPathExtension().lastPathComponent
                out.append(
                    FLEURSBenchmark.FLEURSSample(
                        audioPath: url.path,
                        transcription: transcripts[id] ?? "",
                        language: lang,
                        sampleId: id
                    )
                )
            }
            if !audioFiles.isEmpty {
                logger.info("Loaded \(audioFiles.count) samples for \(lang)")
            }
        }
        return out
    }

    public func saveResults(_ results: [LanguageResult], to outputPath: String) throws {
        func sanitize(_ value: Double) -> Double {
            value.isNaN || value.isInfinite ? 0.0 : value
        }
        let output: [String: Any] = [
            "benchmark": "Nemotron Multilingual \(config.dataset.rawValue.uppercased())",
            "dataset": config.dataset.rawValue,
            "timestamp": ISO8601DateFormatter().string(from: Date()),
            "config": [
                "dataset": config.dataset.rawValue,
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
        var samplesExplicitlySet = false
        var outputFile: String?
        var cacheDir: String?
        var dataset: MultilingualBenchmarkDataset = .fleurs
        var datasetRepo = "FluidInference/fleurs-full"
        var modelDir: URL?
        var debugMode = false
        var forcedPrefix = false
        var dumpSamplesPath: String?
        var promptOverride: String?
        var computeUnits: MLComputeUnits?
        var librispeechSubset = "test-clean"
        var autoDownload = false
        var chunkMs = 2240

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
                    samplesExplicitlySet = true
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
            case "--dataset":
                if i + 1 < arguments.count {
                    let raw = arguments[i + 1].lowercased()
                    if let parsed = MultilingualBenchmarkDataset(rawValue: raw) {
                        dataset = parsed
                    } else {
                        logger.error(
                            "Unknown --dataset value '\(raw)'. Expected one of: fleurs, mcv, mls, librispeech, earnings22."
                        )
                        return
                    }
                    i += 1
                }
            case "--librispeech-subset":
                if i + 1 < arguments.count {
                    librispeechSubset = arguments[i + 1]
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
            case "--prompt":
                if i + 1 < arguments.count {
                    promptOverride = arguments[i + 1]
                    i += 1
                }
            case "--compute-units":
                if i + 1 < arguments.count {
                    switch arguments[i + 1].lowercased() {
                    case "cpu", "cpuonly": computeUnits = .cpuOnly
                    case "gpu", "cpuandgpu", "cpu+gpu": computeUnits = .cpuAndGPU
                    case "ane", "cpuandneuralengine", "cpu+ane": computeUnits = .cpuAndNeuralEngine
                    case "all": computeUnits = .all
                    default:
                        logger.warning("Unknown --compute-units value '\(arguments[i + 1])'. Using default.")
                    }
                    i += 1
                }
            case "--auto-download":
                autoDownload = true
            case "--chunk-ms":
                if i + 1 < arguments.count {
                    if let v = Int(arguments[i + 1]) { chunkMs = v }
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

        // Resolve the model directory: either an explicit local --model-dir, or
        // --auto-download to fetch the per-language variant (`<lang>/<chunkMs>ms`,
        // .mlmodelc only) from the HuggingFace repo. Auto-download is
        // single-language only — each per-language ship is a separate variant
        // directory, so a multi-language run must point at a local --model-dir.
        if modelDir == nil && autoDownload {
            guard languages.count == 1 else {
                logger.error(
                    "--auto-download supports a single language at a time (got \(languages.count): "
                        + "\(languages.joined(separator: ", "))). Use --model-dir for multi-language runs.")
                return
            }
            do {
                logger.info("Auto-downloading variant \(languages[0])/\(chunkMs)ms from HuggingFace...")
                modelDir = try await StreamingNemotronMultilingualAsrManager.downloadVariant(
                    languageCode: languages[0], chunkMs: chunkMs)
            } catch {
                logger.error("Auto-download failed: \(error)")
                return
            }
        }

        guard let modelDir = modelDir else {
            logger.error("Missing --model-dir (or pass --auto-download). The model is not auto-downloaded by default.")
            printUsage()
            return
        }

        // Default samples behavior:
        // - FLEURS: 100 (matches existing behavior)
        // - MCV / MLS: all (test splits are larger; explicit opt-in to a
        //   small slice via `--samples N`).
        if !samplesExplicitlySet && dataset != .fleurs {
            samplesPerLanguage = Int.max
        }
        let resolvedCacheDir =
            cacheDir
            ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/FluidAudio/\(dataset.cacheSubdir)").path
        let resolvedOutput =
            outputFile ?? "nemotron_multilingual_\(dataset.rawValue)_results.json"

        logger.info("Nemotron Multilingual \(dataset.rawValue.uppercased()) Benchmark")
        logger.info(String(repeating: "=", count: 50))
        logger.info("Dataset: \(dataset.rawValue) (\(dataset.hfRepo))")
        logger.info("Languages: \(languages.joined(separator: ", "))")
        logger.info("Samples per language: \(samplesPerLanguage == Int.max ? "all" : String(samplesPerLanguage))")
        logger.info("Model dir: \(modelDir.path)")
        if dataset == .fleurs {
            logger.info("Dataset repo: \(datasetRepo)")
        }
        logger.info("Cache dir: \(resolvedCacheDir)")
        logger.info("Output: \(resolvedOutput)")

        let config = Config(
            languages: languages,
            samplesPerLanguage: samplesPerLanguage,
            outputFile: resolvedOutput,
            cacheDir: resolvedCacheDir,
            modelDir: modelDir,
            debugMode: debugMode,
            dataset: dataset,
            datasetRepo: datasetRepo,
            forcedPrefix: forcedPrefix,
            dumpSamplesPath: dumpSamplesPath,
            promptOverride: promptOverride,
            computeUnits: computeUnits,
            librispeechSubset: librispeechSubset
        )

        let benchmark = NemotronMultilingualFleursBenchmark(config: config)

        do {
            let results = try await benchmark.run()
            try benchmark.saveResults(results, to: resolvedOutput)
            logger.info("Results saved to \(resolvedOutput)")

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

                        Nemotron Multilingual Benchmark Usage:
                            fluidaudio nemotron-multilingual-benchmark --model-dir <path> [options]

                        Required:
                            --model-dir, -m <path>   Path to multilingual model directory
                                                     (must contain metadata.json, tokenizer.json,
                                                     encoder.mlmodelc or encoder.mlpackage, etc.)

                        Options:
                            --dataset <name>         Benchmark dataset: fleurs (default) | mcv | mls
                                                     - fleurs: FLEURS (any supported language)
                                                     - mcv:    Mozilla Common Voice v17.0
                                                               (en/es/fr/it/pt; gated, needs HF_TOKEN)
                                                     - mls:    Multilingual LibriSpeech
                                                               (en/es/fr/it/pt; gated, needs HF_TOKEN)
                                                     - librispeech: LibriSpeech English (test-clean default)
                                                               (no HF token needed; auto-downloaded by
                                                               `fluidaudio download --dataset librispeech-test-clean`)
                                                     - earnings22: Earnings22 financial earnings calls
                                                               (real multi-speaker; auto-downloaded by
                                                               `fluidaudio download --dataset earnings22-kws`)
                            --languages <list>       Comma-separated FLEURS codes (default: en_us,fr_fr,de_de,es_419,ja_jp)
                            --samples <n|all>        Samples per language. Default: 100 for FLEURS, "all" for MCV/MLS.
                            --output <file>          Output JSON file (default: nemotron_multilingual_<dataset>_results.json)
                            --cache-dir <path>       Dataset cache (default: ~/Library/Application Support/FluidAudio/<DatasetDir>)
                            --dataset-repo <repo>    Override FLEURS HF repo (default: FluidInference/fleurs-full)
                                                     Ignored for mcv / mls / librispeech.
                            --librispeech-subset <name>  LibriSpeech subset (default: test-clean).
                                                         One of: test-clean, test-other, dev-clean, dev-other.
                                                         Only used when --dataset librispeech.
                            --dump-samples <path>    Write per-sample JSONL with raw + English/basic
                                                     normalized hyp/ref + per-sample WER under each
                                                     (for normalizer debugging)
                            --debug                  Verbose logging
                            --help, -h               Show this help

                        Examples:
                            # FLEURS, 5 languages × 100 samples (default)
                            fluidaudio nemotron-multilingual-benchmark --model-dir ~/my-multilingual-model

                            # MCV-17 cross-validation, all 5 European-Latin languages, full test split
                            HF_TOKEN=hf_... fluidaudio nemotron-multilingual-benchmark \\
                                --model-dir ~/my-multilingual-model \\
                                --dataset mcv \\
                                --languages en_us,fr_fr,es_419,it_it,pt_br

                            # MLS quick smoke test (10 samples / lang)
                            HF_TOKEN=hf_... fluidaudio nemotron-multilingual-benchmark \\
                                --model-dir ~/my-multilingual-model \\
                                --dataset mls \\
                                --languages en_us \\
                                --samples 10

            """
        )
    }
}

#endif
