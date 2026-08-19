#if os(macOS)
import AVFoundation
import FluidAudio
import Foundation

/// Paired baseline-vs-biased benchmark for Nemotron decode-time custom
/// vocabulary (issue #841) on the earnings22-kws chunk dataset (the NeMo
/// CTC-WS keyword-spotting rig: per-chunk `.wav` + `.text.txt` reference +
/// `.dictionary.txt` keywords).
///
/// Each file is transcribed twice by the same manager — once with an empty
/// vocabulary, once with the chunk's dictionary terms — and scored with the
/// same presence-based TP/FP/FN scheme `ctc-earnings-benchmark` uses, so the
/// two vocabulary paths are comparable.
public enum NemotronVocabBenchmark {

    private struct FileResult {
        let fileId: String
        let audioSeconds: Double
        var wer: [String: Double] = [:]
        var tp: [String: Int] = [:]
        var fp: [String: Int] = [:]
        var fn: [String: Int] = [:]
        var seconds: [String: Double] = [:]
        var hypothesis: [String: String] = [:]
    }

    private static let conditions = ["baseline", "biased"]

    public static func runCLI(arguments: [String]) async {
        var dataDir =
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/earnings22-kws/test-dataset").path
        var modelDir: String? = nil
        var maxFiles = 50
        var language = "en-US"
        var weightOverride: Float? = nil
        var outputFile: String? = nil

        var i = 0
        while i < arguments.count {
            switch arguments[i] {
            case "--data-dir":
                i += 1
                if i < arguments.count { dataDir = arguments[i] }
            case "--model-dir", "-m":
                i += 1
                if i < arguments.count { modelDir = arguments[i] }
            case "--max-files":
                i += 1
                if i < arguments.count, let n = Int(arguments[i]) { maxFiles = n }
            case "--language", "-l":
                i += 1
                if i < arguments.count { language = arguments[i] }
            case "--weight":
                i += 1
                if i < arguments.count, let w = Float(arguments[i]) { weightOverride = w }
            case "--output", "-o":
                i += 1
                if i < arguments.count { outputFile = arguments[i] }
            case "--help", "-h":
                printUsage()
                return
            default:
                print("Unknown argument: \(arguments[i])")
            }
            i += 1
        }

        let dataURL = URL(fileURLWithPath: dataDir)
        guard FileManager.default.fileExists(atPath: dataURL.path) else {
            print("Data directory not found: \(dataDir)")
            print("Download with: fluidaudiocli download --dataset earnings22-kws")
            return
        }

        do {
            let resolvedModelDir: URL
            if let modelDir {
                resolvedModelDir = URL(fileURLWithPath: modelDir)
            } else {
                resolvedModelDir =
                    try await StreamingNemotronMultilingualAsrManager
                    .downloadVariant(languageCode: language, chunkMs: 2240)
            }

            let manager = StreamingNemotronMultilingualAsrManager()
            try await manager.loadModels(from: resolvedModelDir)
            await manager.setLanguage(language)

            // File ids = every .wav with a reference and a dictionary beside it.
            let allFiles = try FileManager.default.contentsOfDirectory(atPath: dataURL.path)
            let fileIds = allFiles.filter { $0.hasSuffix(".wav") }
                .map { String($0.dropLast(4)) }
                .filter { id in
                    allFiles.contains("\(id).text.txt") && allFiles.contains("\(id).dictionary.txt")
                }
                .sorted()
                .prefix(maxFiles)

            print("Nemotron custom-vocabulary benchmark (issue #841)")
            print("Model: \(resolvedModelDir.path)")
            print("Files: \(fileIds.count) (of \(allFiles.filter { $0.hasSuffix(".wav") }.count) available)")
            print("Boost: \(weightOverride.map { String($0) } ?? "default (4.5)")\n")

            var results: [FileResult] = []
            let converter = AudioConverter()

            for (index, fileId) in fileIds.enumerated() {
                let wavURL = dataURL.appendingPathComponent("\(fileId).wav")
                let reference =
                    (try? String(
                        contentsOf: dataURL.appendingPathComponent("\(fileId).text.txt"),
                        encoding: .utf8)) ?? ""
                let dictWords =
                    ((try? String(
                        contentsOf: dataURL.appendingPathComponent("\(fileId).dictionary.txt"),
                        encoding: .utf8)) ?? "")
                    .components(separatedBy: .newlines)
                    .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                    .filter { !$0.isEmpty }
                guard !reference.isEmpty, !dictWords.isEmpty else { continue }

                let audioFile = try AVAudioFile(forReading: wavURL)
                guard
                    let buffer = AVAudioPCMBuffer(
                        pcmFormat: audioFile.processingFormat,
                        frameCapacity: AVAudioFrameCount(audioFile.length))
                else { continue }
                try audioFile.read(into: buffer)
                let samples = try converter.resampleBuffer(buffer)
                let audioSeconds = Double(audioFile.length) / audioFile.processingFormat.sampleRate

                let terms = dictWords.map { CustomVocabularyTerm(text: $0, weight: weightOverride) }
                var result = FileResult(fileId: fileId, audioSeconds: audioSeconds)

                for condition in conditions {
                    await manager.setCustomVocabulary(condition == "biased" ? terms : [])
                    let start = Date()
                    _ = try await manager.process(samples: samples)
                    let hypothesis = try await manager.finish()
                    result.seconds[condition] = Date().timeIntervalSince(start)
                    await manager.reset()

                    let metrics = WERCalculator.calculateWERMetrics(
                        hypothesis: hypothesis, reference: reference)
                    result.wer[condition] = metrics.wer
                    result.hypothesis[condition] = hypothesis

                    let refLower = TextNormalizer.normalize(reference).lowercased()
                    let hypLower = TextNormalizer.normalize(hypothesis).lowercased()
                    var tp = 0
                    var fp = 0
                    var fn = 0
                    for word in dictWords {
                        let inRef = containsWholeWord(refLower, word)
                        let inHyp = containsWholeWord(hypLower, word)
                        if inRef && inHyp {
                            tp += 1
                        } else if inHyp {
                            fp += 1
                        } else if inRef {
                            fn += 1
                        }
                    }
                    result.tp[condition] = tp
                    result.fp[condition] = fp
                    result.fn[condition] = fn
                }
                results.append(result)

                let bWer = (result.wer["baseline"] ?? 0) * 100
                let vWer = (result.wer["biased"] ?? 0) * 100
                let bRecall = recallString(result, "baseline")
                let vRecall = recallString(result, "biased")
                print(
                    String(
                        format: "[%3d/%d] %@ WER %5.1f%% -> %5.1f%%  dict %@ -> %@",
                        index + 1, fileIds.count,
                        fileId.padding(toLength: 24, withPad: " ", startingAt: 0),
                        bWer, vWer, bRecall, vRecall))
            }

            printSummary(results, weightOverride: weightOverride)
            if let outputFile {
                try writeJSON(results, to: outputFile, weightOverride: weightOverride)
                print("\nResults written to \(outputFile)")
            }
        } catch {
            print("Benchmark failed: \(error)")
        }
    }

    private static func containsWholeWord(_ text: String, _ word: String) -> Bool {
        let pattern = "\\b\(NSRegularExpression.escapedPattern(for: word.lowercased()))\\b"
        guard let regex = try? NSRegularExpression(pattern: pattern) else {
            return text.contains(word.lowercased())
        }
        return regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text)) != nil
    }

    private static func recallString(_ result: FileResult, _ condition: String) -> String {
        let tp = result.tp[condition] ?? 0
        let fn = result.fn[condition] ?? 0
        return "\(tp)/\(tp + fn)"
    }

    private static func printSummary(_ results: [FileResult], weightOverride: Float?) {
        guard !results.isEmpty else {
            print("No files processed")
            return
        }
        print("\n" + String(repeating: "=", count: 72))
        print("NEMOTRON CUSTOM VOCABULARY BENCHMARK (earnings22-kws)")
        print(String(repeating: "=", count: 72))
        print("Files: \(results.count)   Boost: \(weightOverride.map { String($0) } ?? "default (4.5)")")
        let audio = results.reduce(0.0) { $0 + $1.audioSeconds }
        print(String(format: "Audio: %.1fs", audio))
        for condition in conditions {
            let avgWer =
                results.reduce(0.0) { $0 + ($1.wer[condition] ?? 0) } / Double(results.count) * 100
            let tp = results.reduce(0) { $0 + ($1.tp[condition] ?? 0) }
            let fp = results.reduce(0) { $0 + ($1.fp[condition] ?? 0) }
            let fn = results.reduce(0) { $0 + ($1.fn[condition] ?? 0) }
            let recall = tp + fn > 0 ? Double(tp) / Double(tp + fn) * 100 : 0
            let precision = tp + fp > 0 ? Double(tp) / Double(tp + fp) * 100 : 0
            let time = results.reduce(0.0) { $0 + ($1.seconds[condition] ?? 0) }
            let rtfx = time > 0 ? audio / time : 0
            print(
                String(
                    format: "%@  WER %6.2f%%   recall %5.1f%% (TP=%d FN=%d)   precision %5.1f%% (FP=%d)   RTFx %.1fx",
                    condition.padding(toLength: 9, withPad: " ", startingAt: 0),
                    avgWer, recall, tp, fn, precision, fp, rtfx))
        }
    }

    private static func writeJSON(
        _ results: [FileResult], to path: String, weightOverride: Float?
    ) throws {
        let files: [[String: Any]] = results.map { result in
            var entry: [String: Any] = [
                "fileId": result.fileId,
                "audioSeconds": result.audioSeconds,
            ]
            for condition in conditions {
                entry[condition] = [
                    "wer": result.wer[condition] ?? 0,
                    "tp": result.tp[condition] ?? 0,
                    "fp": result.fp[condition] ?? 0,
                    "fn": result.fn[condition] ?? 0,
                    "seconds": result.seconds[condition] ?? 0,
                    "hypothesis": result.hypothesis[condition] ?? "",
                ]
            }
            return entry
        }
        let payload: [String: Any] = [
            "benchmark": "nemotron-vocab-benchmark",
            "boost": weightOverride.map { Double($0) } ?? 4.5,
            "files": files,
        ]
        let data = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }

    private static func printUsage() {
        print(
            """
            Nemotron decode-time custom vocabulary benchmark (issue #841)

            Runs each earnings22-kws chunk twice — with and without the chunk's
            dictionary terms — and reports paired WER / vocab recall / precision.

            Usage: fluidaudio nemotron-vocab-benchmark [options]

            Options:
                --data-dir <path>    earnings22-kws test-dataset directory
                                     (default: app-support copy)
                --model-dir <path>   Nemotron multilingual model directory
                                     (default: auto-download 2240ms variant)
                --max-files <int>    Number of chunks to run (default: 50)
                --language <code>    Language hint (default: en-US)
                --weight <float>     Per-term weight override in (0, 6.0]
                                     (default boost 4.5; values above 6.0
                                     fall back to the default)
                --output, -o <path>  Write per-file JSON results
            """
        )
    }
}
#endif
