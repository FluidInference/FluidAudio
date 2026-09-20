#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// `enhance-benchmark`: near-end word recall / far-end leakage of LocalVQE on the
/// Microsoft AEC-Challenge synthetic set (mic + loopback + clean near-end triples).
///
/// The in-repo Parakeet ASR transcribes the clean near-end clip (reference
/// words) and the loopback clip (far-end words), then the unprocessed mic and
/// each enhanced output. Near-end recall is the fraction of reference words the
/// hypothesis keeps (1 - (deletions + substitutions) / N); far-end leakage is
/// the fraction of far-end words that show up in the hypothesis without being
/// near-end words. WER against the clean-near-end transcript is reported too.
enum EnhanceBenchmarkCommand {
    private static let logger = AppLogger(category: "EnhanceBenchmark")

    static let datasetRepo = "FluidInference/aec-challenge-synthetic-mini"
    static let datasetArchive = "aec-synthetic-mini.tar.gz"
    static let datasetFolder = "aec-synthetic-mini"
    static let datasetRevision = "1f3714b5a3f98cedef1bbb017f21bbd7ae688596"
    static let datasetArchiveSHA256 = "45ff5d7acfce499558c25a0eace45eb819cec8aa76420fe733de7ee116ae548d"
    static let datasetMetadataSHA256 = "865aff8e66eb682c292f42a9d747d931f3a2f71f18e16fec53aea80dbdc2eacc"
    static let expectedDatasetExamples = 200

    private struct Options {
        var datasetDir: String?
        var maxFiles: Int?
        var variants: [LocalVqeVariant] = [.v13, .v12]
        var chunk: LocalVqeChunk = .batch256ms
        var computeUnits: MLComputeUnits = .cpuOnly
        var includeNoReference = false
        var outputPath: String?
    }

    private struct ConditionTotals {
        var files = 0
        var refWords = 0
        var hits = 0
        var errors = 0
        var farWords = 0
        var leaked = 0
        var enhanceSeconds = 0.0
        var audioSeconds = 0.0

        var recall: Double { refWords == 0 ? 0 : Double(hits) / Double(refWords) }
        var wer: Double { refWords == 0 ? 0 : Double(errors) / Double(refWords) }
        var leakage: Double { farWords == 0 ? 0 : Double(leaked) / Double(farWords) }
        var rtfx: Double { enhanceSeconds <= 0 ? 0 : audioSeconds / enhanceSeconds }
    }

    static func run(arguments: [String]) async {
        var options = Options()
        var index = 0
        while index < arguments.count {
            let arg = arguments[index]
            switch arg {
            case "--help", "-h":
                printUsage()
                exit(0)
            case "--dataset-dir":
                options.datasetDir = requiredValue(arguments, &index)
            case "--max-files":
                guard let raw = next(arguments, &index), let maxFiles = Int(raw), maxFiles > 0 else {
                    logger.error("--max-files must be a positive integer")
                    exit(1)
                }
                options.maxFiles = maxFiles
            case "--variants":
                let raw = (next(arguments, &index) ?? "").split(separator: ",", omittingEmptySubsequences: false)
                    .map(String.init)
                let parsed = raw.compactMap(LocalVqeVariant.init(rawValue:))
                guard parsed.count == raw.count, !parsed.isEmpty,
                    Set(parsed.map(\.rawValue)).count == parsed.count
                else {
                    logger.error("--variants must be a comma list of \(LocalVqeVariant.allCases.map(\.rawValue))")
                    exit(1)
                }
                options.variants = parsed
            case "--chunk":
                guard let raw = next(arguments, &index), let c = LocalVqeChunk(rawValue: raw) else {
                    logger.error("--chunk must be one of \(LocalVqeChunk.allCases.map(\.rawValue))")
                    exit(1)
                }
                options.chunk = c
            case "--compute-units":
                switch next(arguments, &index)?.lowercased() {
                case "cpu-only", "cpu": options.computeUnits = .cpuOnly
                case "gpu", "cpu-and-gpu": options.computeUnits = .cpuAndGPU
                case "ane", "cpu-and-ne": options.computeUnits = .cpuAndNeuralEngine
                case "all": options.computeUnits = .all
                default:
                    logger.error("--compute-units must be cpu-only | gpu | ane | all")
                    exit(1)
                }
            case "--no-reference":
                options.includeNoReference = true
            case "--output":
                options.outputPath = requiredValue(arguments, &index)
            default:
                logger.error("Unknown option: \(arg)")
                exit(1)
            }
            index += 1
        }

        do {
            let datasetDir = try await resolveDataset(options.datasetDir)
            var examples = try EnhanceBenchmarkDataset.loadExamples(from: datasetDir)
            if let maxFiles = options.maxFiles { examples = Array(examples.prefix(maxFiles)) }
            guard !examples.isEmpty else {
                logger.error("No examples found in \(datasetDir.path)")
                exit(1)
            }
            report("Dataset: \(datasetDir.path) (\(examples.count) examples)")
            let startedAt = Date()
            let audioHashes = options.outputPath == nil ? [:] : try EnhanceBenchmarkProvenance.audioFiles(examples)

            let asr = AsrManager()
            let asrConfiguration = MLModelConfigurationUtils.defaultConfiguration(computeUnits: .cpuOnly)
            try await asr.loadModels(
                try await AsrModels.downloadAndLoad(configuration: asrConfiguration, version: .v3))
            report("ASR: Parakeet TDT v3 int8 loaded (CPU-only)")

            var conditions: [(name: String, manager: LocalVqeManager?, useReference: Bool)] = [
                ("unprocessed", nil, true)
            ]
            for variant in options.variants {
                let config = LocalVqeConfig(variant: variant, chunk: options.chunk, computeUnits: options.computeUnits)
                let manager = try await LocalVqeManager(config: config)
                conditions.append(("localvqe-\(variant.rawValue)", manager, true))
                if options.includeNoReference {
                    conditions.append(("localvqe-\(variant.rawValue)-noref", manager, false))
                }
            }
            report("Conditions: \(conditions.map(\.name).joined(separator: ", "))")
            let modelHashes =
                options.outputPath == nil
                ? [:]
                : try EnhanceBenchmarkProvenance.modelFiles(
                    variants: options.variants, chunk: options.chunk)

            let converter = AudioConverter()
            var totals = [String: ConditionTotals]()
            var bySer = [String: [String: ConditionTotals]]()  // bucket -> condition -> totals
            var rows: [[String: Any]] = []
            var emptyReferenceFileIDs: [String] = []

            for (i, example) in examples.enumerated() {
                let mic = try converter.resampleAudioFile(example.mic)
                let lpb = try converter.resampleAudioFile(example.lpb)
                let clean = try converter.resampleAudioFile(example.clean)
                try validateAudio(mic: mic, reference: lpb, clean: clean, fileID: example.fileID)

                let refWords = words(try await transcribe(asr, clean))
                if refWords.isEmpty {
                    // No reference words to recall: the ASR produced nothing on the
                    // clean near-end clip. Excluded from every metric and counted.
                    emptyReferenceFileIDs.append(example.fileID)
                    continue
                }
                let farWords = words(try await transcribe(asr, lpb))
                let bucket = serBucket(example.ser)
                var row: [String: Any] = [
                    "fileid": example.fileID, "ser": example.ser, "ref_words": refWords.count,
                    "far_words": farWords.count, "reference": refWords.joined(separator: " "),
                    "far_reference": farWords.joined(separator: " "),
                    "is_farend_noisy": example.farendNoisy,
                    "is_nearend_noisy": example.nearendNoisy,
                    "audio_seconds": Double(mic.count) / Double(LocalVqeManager.sampleRate),
                ]

                for condition in conditions {
                    var enhanced = mic
                    var enhanceSeconds = 0.0
                    if let manager = condition.manager {
                        let reference = condition.useReference ? lpb : [Float](repeating: 0, count: mic.count)
                        let start = ContinuousClock.now
                        enhanced = try await manager.process(mic: mic, reference: reference)
                        let elapsed = start.duration(to: .now).components
                        enhanceSeconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                    }
                    guard enhanced.count == mic.count, enhanced.allSatisfy(\.isFinite) else {
                        throw LocalVqeError.modelProcessingFailed(
                            "Invalid \(condition.name) output for fileid \(example.fileID)")
                    }
                    let hypWords = words(try await transcribe(asr, enhanced))
                    let m = score(hypothesis: hypWords, reference: refWords, farEnd: farWords)

                    var t = totals[condition.name, default: ConditionTotals()]
                    t.files += 1
                    t.refWords += refWords.count
                    t.hits += m.hits
                    t.errors += m.errors
                    t.farWords += farWords.count
                    t.leaked += m.leaked
                    t.enhanceSeconds += enhanceSeconds
                    t.audioSeconds += Double(mic.count) / Double(LocalVqeManager.sampleRate)
                    totals[condition.name] = t
                    var b = bySer[bucket, default: [:]][condition.name, default: ConditionTotals()]
                    b.files += 1
                    b.refWords += refWords.count
                    b.hits += m.hits
                    b.errors += m.errors
                    b.farWords += farWords.count
                    b.leaked += m.leaked
                    bySer[bucket, default: [:]][condition.name] = b

                    row["\(condition.name)_recall"] = refWords.isEmpty ? 0 : Double(m.hits) / Double(refWords.count)
                    row["\(condition.name)_wer"] = refWords.isEmpty ? 0 : Double(m.errors) / Double(refWords.count)
                    row["\(condition.name)_leaked"] = m.leaked
                    row["\(condition.name)_hyp"] = hypWords.joined(separator: " ")
                    row["\(condition.name)_hits"] = m.hits
                    row["\(condition.name)_errors"] = m.errors
                    row["\(condition.name)_insertions"] = m.insertions
                    row["\(condition.name)_deletions"] = m.deletions
                    row["\(condition.name)_substitutions"] = m.substitutions
                    row["\(condition.name)_enhancement_seconds"] = enhanceSeconds
                }
                rows.append(row)

                if (i + 1) % 10 == 0 || i + 1 == examples.count {
                    let parts = conditions.map { c -> String in
                        let t = totals[c.name] ?? ConditionTotals()
                        return String(format: "%@ R=%.1f%% L=%.1f%%", c.name, t.recall * 100, t.leakage * 100)
                    }
                    report("[\(i + 1)/\(examples.count)] " + parts.joined(separator: " | "))
                }
            }

            guard !rows.isEmpty else {
                throw LocalVqeError.modelProcessingFailed(
                    "No examples were scored: all \(examples.count) clean-reference transcripts were empty")
            }
            report("")
            if !emptyReferenceFileIDs.isEmpty {
                report(
                    "Excluded \(emptyReferenceFileIDs.count) examples whose clean near-end transcript was empty: "
                        + emptyReferenceFileIDs.joined(separator: ", "))
            }
            report(row(["condition", "files", "recall", "WER", "leakage", "RTFx"]))
            for condition in conditions {
                let t = totals[condition.name] ?? ConditionTotals()
                report(
                    row([
                        condition.name, "\(t.files)", pct(t.recall), pct(t.wer), pct(t.leakage),
                        condition.manager == nil ? "-" : String(format: "%.1fx", t.rtfx),
                    ]))
            }
            for bucket in ["ser<=0", "ser>0", "ser=?"] {
                guard let perCondition = bySer[bucket] else { continue }
                report("")
                report("SER bucket \(bucket):")
                for condition in conditions {
                    let t = perCondition[condition.name] ?? ConditionTotals()
                    report(row(["  " + condition.name, "\(t.files)", pct(t.recall), pct(t.wer), pct(t.leakage), ""]))
                }
            }

            if let outputPath = options.outputPath {
                var summary: [String: Any] = [:]
                for (name, t) in totals {
                    summary[name] = [
                        "files": t.files, "recall": t.recall, "wer": t.wer, "leakage": t.leakage, "rtfx": t.rtfx,
                        "reference_words": t.refWords, "hits": t.hits, "errors": t.errors,
                        "far_end_words": t.farWords, "leaked_words": t.leaked,
                        "audio_seconds": t.audioSeconds, "enhancement_seconds": t.enhanceSeconds,
                    ]
                }
                let payload: [String: Any] = [
                    "schema_version": 2,
                    "protocol": "localvqe-asr-v2",
                    "started_at": ISO8601DateFormatter().string(from: startedAt),
                    "completed_at": ISO8601DateFormatter().string(from: Date()),
                    "conditions": conditions.map(\.name),
                    "configuration": [
                        "asr": "parakeet-tdt-v3-int8", "asr_compute_units": "cpu-only",
                        "enhancement_compute_units": options.computeUnits.rawValue,
                        "normalization": "TextNormalizer.normalize", "sample_rate": LocalVqeManager.sampleRate,
                        "aggregation": "micro", "empty_reference_policy": "exclude-from-all-conditions",
                    ],
                    "model_files_sha256": modelHashes,
                    "environment": [
                        "os": ProcessInfo.processInfo.operatingSystemVersionString,
                        "processor_count": ProcessInfo.processInfo.processorCount,
                        "source_revision": ProcessInfo.processInfo.environment["GITHUB_SHA"] ?? "unrecorded",
                    ],
                    "dataset": [
                        "path": datasetDir.path,
                        "repository": options.datasetDir == nil ? datasetRepo : "custom",
                        "revision": options.datasetDir == nil ? datasetRevision : "custom",
                        "archive_sha256": options.datasetDir == nil ? datasetArchiveSHA256 : NSNull(),
                        "metadata_sha256": try EnhanceBenchmarkDataset.sha256(
                            of: datasetDir.appendingPathComponent("meta.csv")),
                        "selection_order": "numeric fileid",
                        "selected_fileids": examples.map(\.fileID),
                        "audio_files_sha256": audioHashes,
                    ],
                    "chunk": options.chunk.rawValue, "summary": summary, "files": rows,
                    "excluded_empty_reference_fileids": emptyReferenceFileIDs,
                ]
                let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: URL(fileURLWithPath: outputPath), options: .atomic)
                report("Wrote \(outputPath)")
            }
        } catch {
            logger.error("enhance-benchmark failed: \(error)")
            exit(1)
        }
    }

    // MARK: - Scoring

    private static func transcribe(_ asr: AsrManager, _ samples: [Float]) async throws -> String {
        var state = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
        return try await asr.transcribe(samples, decoderState: &state).text
    }

    private static func words(_ text: String) -> [String] {
        TextNormalizer.normalize(text).split(whereSeparator: { $0.isWhitespace }).map(String.init)
    }

    /// hits = reference words kept by the hypothesis (N - deletions - substitutions);
    /// errors = S + D + I; leaked = far-end words present in the hypothesis beyond
    /// what the near-end reference accounts for (multiset).
    static func score(
        hypothesis: [String], reference: [String], farEnd: [String]
    ) -> (hits: Int, errors: Int, leaked: Int, insertions: Int, deletions: Int, substitutions: Int) {
        let m = WERCalculator.calculateWordMetrics(hypothesis: hypothesis, reference: reference)
        let hits = max(0, m.totalWords - m.deletions - m.substitutions)
        let errors = m.insertions + m.deletions + m.substitutions

        var spare = [String: Int]()
        for w in hypothesis { spare[w, default: 0] += 1 }
        for w in reference where spare[w, default: 0] > 0 { spare[w, default: 0] -= 1 }
        var leaked = 0
        for w in farEnd where spare[w, default: 0] > 0 {
            spare[w, default: 0] -= 1
            leaked += 1
        }
        return (hits, errors, leaked, m.insertions, m.deletions, m.substitutions)
    }

    static func validateAudio(mic: [Float], reference: [Float], clean: [Float], fileID: String) throws {
        guard !mic.isEmpty, mic.count == reference.count, mic.count == clean.count else {
            throw LocalVqeError.modelProcessingFailed("Empty or unequal audio lengths for fileid \(fileID)")
        }
        guard mic.allSatisfy(\.isFinite), reference.allSatisfy(\.isFinite), clean.allSatisfy(\.isFinite) else {
            throw LocalVqeError.modelProcessingFailed("Non-finite audio samples for fileid \(fileID)")
        }
    }

    private static func pct(_ value: Double) -> String {
        String(format: "%.2f%%", value * 100)
    }

    /// Fixed-width table row: first column left-aligned, the rest right-aligned.
    private static func row(_ cells: [String]) -> String {
        let widths = [24, 6, 9, 9, 9, 8]
        return cells.enumerated().map { i, cell in
            let w = widths[min(i, widths.count - 1)]
            let pad = String(repeating: " ", count: max(0, w - cell.count))
            return i == 0 ? cell + pad : pad + cell
        }.joined(separator: " ")
    }

    private static func serBucket(_ ser: Int?) -> String {
        guard let ser else { return "ser=?" }
        return ser <= 0 ? "ser<=0" : "ser>0"
    }

    // MARK: - Dataset

    private static func resolveDataset(_ override: String?) async throws -> URL {
        if let override {
            return URL(fileURLWithPath: override)
        }
        let base = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/FluidAudio/Datasets", isDirectory: true)
        let dir = base.appendingPathComponent(datasetFolder, isDirectory: true)
        if FileManager.default.fileExists(atPath: dir.appendingPathComponent("meta.csv").path) {
            try validatePinnedDataset(dir)
            return dir
        }
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let url = try ModelRegistry.resolveDataset(datasetRepo, datasetArchive, revision: datasetRevision)
        report("Downloading \(url.absoluteString)")
        let (tmp, response) = try await URLSession.shared.download(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw LocalVqeError.modelProcessingFailed("dataset download failed: \(response)")
        }
        let actualArchiveHash = try EnhanceBenchmarkDataset.sha256(of: tmp)
        guard actualArchiveHash == datasetArchiveSHA256 else {
            throw LocalVqeError.modelProcessingFailed(
                "dataset archive checksum mismatch: expected \(datasetArchiveSHA256), got \(actualArchiveHash)")
        }
        let archive = base.appendingPathComponent(datasetArchive)
        try? FileManager.default.removeItem(at: archive)
        try FileManager.default.moveItem(at: tmp, to: archive)
        let tar = Process()
        tar.executableURL = URL(fileURLWithPath: "/usr/bin/tar")
        tar.arguments = ["-xzf", archive.path, "-C", base.path]
        try tar.run()
        tar.waitUntilExit()
        try? FileManager.default.removeItem(at: archive)
        guard tar.terminationStatus == 0,
            FileManager.default.fileExists(atPath: dir.appendingPathComponent("meta.csv").path)
        else {
            throw LocalVqeError.modelProcessingFailed("dataset extraction failed (tar status \(tar.terminationStatus))")
        }
        try validatePinnedDataset(dir)
        return dir
    }

    private static func validatePinnedDataset(_ dir: URL) throws {
        let metaURL = dir.appendingPathComponent("meta.csv")
        let actualMetadataHash = try EnhanceBenchmarkDataset.sha256(of: metaURL)
        guard actualMetadataHash == datasetMetadataSHA256 else {
            throw LocalVqeError.modelProcessingFailed(
                "dataset metadata checksum mismatch: expected \(datasetMetadataSHA256), got \(actualMetadataHash)")
        }
        let examples = try EnhanceBenchmarkDataset.loadExamples(from: dir)
        guard examples.count == expectedDatasetExamples else {
            throw LocalVqeError.modelProcessingFailed(
                "dataset contains \(examples.count) examples; expected \(expectedDatasetExamples)")
        }
    }

    private static func next(_ arguments: [String], _ index: inout Int) -> String? {
        guard index + 1 < arguments.count, !arguments[index + 1].hasPrefix("--") else { return nil }
        index += 1
        return arguments[index]
    }

    private static func requiredValue(_ arguments: [String], _ index: inout Int) -> String {
        let option = arguments[index]
        guard let value = next(arguments, &index), !value.isEmpty else {
            logger.error("\(option) requires a value")
            exit(1)
        }
        return value
    }

    private static func report(_ line: String) {
        print(line)
        logger.info("\(line)")
    }

    private static func printUsage() {
        // print, not the logger: usage must show in release builds too.
        print(
            """
            Usage: fluidaudiocli enhance-benchmark [options]

            Scores LocalVQE on the Microsoft AEC-Challenge synthetic set (mic + loopback + clean near-end):
            near-end word recall, WER vs the clean-near-end transcript, and far-end word leakage, all
            measured with the in-repo Parakeet TDT v3 ASR.

            Options:
                --dataset-dir <dir>      Directory with fileid_*_{mic,lpb,clean}.wav + meta.csv
                                         (default: auto-download \(datasetRepo)).
                --max-files <n>          Score only the first n examples (numeric fileid order).
                --variants <list>        Comma list of v1.3,v1.2 (default both).
                --chunk <256ms|16ms>     Chunk export to benchmark (default 256ms).
                --compute-units <cpu-only|gpu|ane|all>
                --no-reference           Also score each variant with a silent far end (NS-only mode).
                --output <file.json>     Write per-file and summary results.
            """
        )
    }
}
#endif
