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

    private struct Options {
        var datasetDir: String?
        var maxFiles: Int?
        var variants: [LocalVqeVariant] = [.v13, .v12]
        var chunk: LocalVqeChunk = .batch256ms
        var computeUnits: MLComputeUnits = .cpuOnly
        var includeNoReference = false
        var outputPath: String?
    }

    private struct Example {
        let fileID: String
        let mic: URL
        let lpb: URL
        let clean: URL
        let ser: Int?
        let nearendNoisy: Bool
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
                options.datasetDir = next(arguments, &index)
            case "--max-files":
                options.maxFiles = Int(next(arguments, &index) ?? "")
            case "--variants":
                let raw = (next(arguments, &index) ?? "").split(separator: ",").map(String.init)
                let parsed = raw.compactMap(LocalVqeVariant.init(rawValue:))
                guard parsed.count == raw.count, !parsed.isEmpty else {
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
                options.outputPath = next(arguments, &index)
            default:
                logger.warning("Unknown option: \(arg)")
            }
            index += 1
        }

        do {
            let datasetDir = try await resolveDataset(options.datasetDir)
            var examples = try loadExamples(from: datasetDir)
            if let maxFiles = options.maxFiles { examples = Array(examples.prefix(maxFiles)) }
            guard !examples.isEmpty else {
                logger.error("No examples found in \(datasetDir.path)")
                exit(1)
            }
            report("Dataset: \(datasetDir.path) (\(examples.count) examples)")

            let asr = AsrManager()
            try await asr.loadModels(try await AsrModels.downloadAndLoad())
            report("ASR: Parakeet TDT v3 loaded")

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

            let converter = AudioConverter()
            var totals = [String: ConditionTotals]()
            var bySer = [String: [String: ConditionTotals]]()  // bucket -> condition -> totals
            var rows: [[String: Any]] = []

            for (i, example) in examples.enumerated() {
                let mic = try converter.resampleAudioFile(example.mic)
                var lpb = try converter.resampleAudioFile(example.lpb)
                let clean = try converter.resampleAudioFile(example.clean)
                if lpb.count < mic.count {
                    lpb.append(contentsOf: [Float](repeating: 0, count: mic.count - lpb.count))
                } else if lpb.count > mic.count {
                    lpb.removeLast(lpb.count - mic.count)
                }

                let refWords = words(try await transcribe(asr, clean))
                let farWords = words(try await transcribe(asr, lpb))
                let bucket = serBucket(example.ser)
                var row: [String: Any] = [
                    "fileid": example.fileID, "ser": example.ser as Any, "ref_words": refWords.count,
                    "far_words": farWords.count, "reference": refWords.joined(separator: " "),
                ]

                for condition in conditions {
                    var enhanced = mic
                    var enhanceSeconds = 0.0
                    if let manager = condition.manager {
                        let reference = condition.useReference ? lpb : [Float](repeating: 0, count: mic.count)
                        let start = Date()
                        enhanced = try await manager.process(mic: mic, reference: reference)
                        enhanceSeconds = Date().timeIntervalSince(start)
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

            report("")
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
                    ]
                }
                let payload: [String: Any] = [
                    "dataset": datasetDir.path, "chunk": options.chunk.rawValue, "summary": summary, "files": rows,
                ]
                let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: URL(fileURLWithPath: outputPath))
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
    private static func score(
        hypothesis: [String], reference: [String], farEnd: [String]
    ) -> (hits: Int, errors: Int, leaked: Int) {
        let m = WERCalculator.calculateWERMetrics(
            hypothesis: hypothesis.joined(separator: " "), reference: reference.joined(separator: " "))
        let hits = max(0, m.totalWords - m.deletions - m.substitutions)
        let errors = m.insertions + m.deletions + m.substitutions

        var spare = [String: Int]()
        for w in hypothesis { spare[w, default: 0] += 1 }
        for w in reference where spare[w, default: 0] > 0 { spare[w]! -= 1 }
        var leaked = 0
        for w in farEnd where spare[w, default: 0] > 0 {
            spare[w]! -= 1
            leaked += 1
        }
        return (hits, errors, leaked)
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
            return dir
        }
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        let url = try ModelRegistry.resolveDataset(datasetRepo, datasetArchive)
        report("Downloading \(url.absoluteString)")
        let (tmp, response) = try await URLSession.shared.download(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw LocalVqeError.modelProcessingFailed("dataset download failed: \(response)")
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
        return dir
    }

    private static func loadExamples(from dir: URL) throws -> [Example] {
        let metaURL = dir.appendingPathComponent("meta.csv")
        let text = try String(contentsOf: metaURL, encoding: .utf8)
        var lines = text.split(whereSeparator: { $0 == "\n" || $0 == "\r\n" }).map(String.init)
        guard !lines.isEmpty else { return [] }
        let header = lines.removeFirst().split(separator: ",").map { String($0).trimmingCharacters(in: .whitespaces) }
        func column(_ name: String, _ fields: [String]) -> String? {
            guard let i = header.firstIndex(of: name), i < fields.count else { return nil }
            let v = fields[i].trimmingCharacters(in: .whitespaces)
            return v.isEmpty ? nil : v
        }
        var examples: [Example] = []
        for line in lines {
            let fields = line.split(separator: ",", omittingEmptySubsequences: false).map(String.init)
            guard let fileID = column("fileid", fields) else { continue }
            let stem = "fileid_\(fileID)"
            let mic = dir.appendingPathComponent("\(stem)_mic.wav")
            let lpb = dir.appendingPathComponent("\(stem)_lpb.wav")
            let clean = dir.appendingPathComponent("\(stem)_clean.wav")
            guard [mic, lpb, clean].allSatisfy({ FileManager.default.fileExists(atPath: $0.path) }) else { continue }
            examples.append(
                Example(
                    fileID: fileID, mic: mic, lpb: lpb, clean: clean,
                    ser: column("ser", fields).flatMap(Int.init),
                    nearendNoisy: column("is_nearend_noisy", fields) == "1"))
        }
        return examples.sorted { ($0.ser ?? 0, $0.fileID) < ($1.ser ?? 0, $1.fileID) }
    }

    private static func next(_ arguments: [String], _ index: inout Int) -> String? {
        guard index + 1 < arguments.count else { return nil }
        index += 1
        return arguments[index]
    }

    private static func report(_ line: String) {
        print(line)
        logger.info("\(line)")
    }

    private static func printUsage() {
        logger.info(
            """
            Usage: fluidaudiocli enhance-benchmark [options]

            Scores LocalVQE on the Microsoft AEC-Challenge synthetic set (mic + loopback + clean near-end):
            near-end word recall, WER vs the clean-near-end transcript, and far-end word leakage, all
            measured with the in-repo Parakeet TDT v3 ASR.

            Options:
                --dataset-dir <dir>      Directory with fileid_*_{mic,lpb,clean}.wav + meta.csv
                                         (default: auto-download \(datasetRepo)).
                --max-files <n>          Score only the first n examples (sorted by SER).
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
