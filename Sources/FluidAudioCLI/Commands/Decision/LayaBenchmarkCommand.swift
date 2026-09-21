#if os(macOS)
import FluidAudio
import Foundation

/// `laya-benchmark`: accuracy and latency of the Core ML laya buckets on laya's published suites.
///
/// Reads `suites.jsonl` written by the Mobius `benchmark.py` (one question per line with the exact
/// serialized state and gold label) and, optionally, `reference-rows.jsonl` with the PyTorch answers
/// for the same rows, so the report shows accuracy per suite, agreement with the reference, and
/// per-question latency on device.
struct LayaBenchmarkCommand {
    private static let logger = AppLogger(category: "LayaBenchmark")

    static func run(arguments: [String]) async {
        if arguments.contains("--help") || arguments.contains("-h") {
            printUsage()
            return
        }
        do {
            try await execute(arguments: arguments)
        } catch {
            logger.error("laya-benchmark failed: \(error.localizedDescription)")
            exit(1)
        }
    }

    private struct Row: Decodable {
        let suite: String
        let index: Int
        let state: String
        let type: String
        let instructions: String
        let options: [[String?]]
        let gold: Int

        var question: LayaQuestion {
            switch type {
            case "choice":
                return .choice(
                    instructions,
                    options: options.map { LayaQuestion.Choice($0[0] ?? "", description: $0.count > 1 ? $0[1] : nil) })
            case "score":
                return .score(instructions, levels: options.map { $0[0] ?? "" })
            default:
                let byLabel = Dictionary(
                    uniqueKeysWithValues: options.map { ($0[0] ?? "", $0.count > 1 ? $0[1] : nil) })
                return .noul(
                    instructions, falseDescription: byLabel["false"] ?? nil, trueDescription: byLabel["true"] ?? nil)
            }
        }
    }

    private struct ReferenceRow: Decodable {
        let suite: String
        let index: Int
        let dropped: Bool?
        let argmax: Int?
        let probabilities: [Float]?
    }

    private struct Options {
        var suites: String?
        var reference: String?
        var modelDirectory: String?
        var lengths: [Int] = [128, 256, 512, 1024]
        var report: String?
        var limit: Int?
        var only: Set<String>?
    }

    private static func parse(_ arguments: [String]) throws -> Options {
        var options = Options()
        var index = 0
        func value(_ flag: String) throws -> String {
            index += 1
            guard index < arguments.count else { throw LayaError.invalidAsset("\(flag) needs a value") }
            return arguments[index]
        }
        while index < arguments.count {
            switch arguments[index] {
            case "--suites": options.suites = try value("--suites")
            case "--reference": options.reference = try value("--reference")
            case "--model-dir": options.modelDirectory = try value("--model-dir")
            case "--lengths": options.lengths = try value("--lengths").split(separator: ",").compactMap { Int($0) }
            case "--report": options.report = try value("--report")
            case "--limit": options.limit = Int(try value("--limit"))
            case "--only": options.only = Set(try value("--only").split(separator: ",").map(String.init))
            default: throw LayaError.invalidAsset("Unknown argument \(arguments[index])")
            }
            index += 1
        }
        guard options.suites != nil else { throw LayaError.invalidAsset("--suites FILE is required") }
        return options
    }

    private static func loadLines<T: Decodable>(_ path: String, as type: T.Type) throws -> [T] {
        let data = try Data(contentsOf: URL(fileURLWithPath: path))
        let decoder = JSONDecoder()
        return try data.split(separator: UInt8(ascii: "\n")).filter { !$0.isEmpty }.map {
            try decoder.decode(T.self, from: Data($0))
        }
    }

    private static func execute(arguments: [String]) async throws {
        let options = try parse(arguments)
        var rows = try loadLines(options.suites!, as: Row.self)
        if let only = options.only { rows = rows.filter { only.contains($0.suite) } }
        if let limit = options.limit {
            var seen: [String: Int] = [:]
            rows = rows.filter { row in
                seen[row.suite, default: 0] += 1
                return seen[row.suite]! <= limit
            }
        }
        var reference: [String: ReferenceRow] = [:]
        if let path = options.reference {
            for item in try loadLines(path, as: ReferenceRow.self) {
                reference["\(item.suite)#\(item.index)"] = item
            }
        }
        let configuration = LayaManager.Configuration(lengths: options.lengths)
        let loadStarted = Date()
        let manager: LayaManager
        if let directory = options.modelDirectory {
            manager = try await LayaManager.load(from: URL(fileURLWithPath: directory), configuration: configuration)
        } else {
            manager = try await LayaManager.load(configuration: configuration)
        }
        let loadSeconds = Date().timeIntervalSince(loadStarted)
        logger.info(
            "Loaded buckets \(manager.lengths) in \(String(format: "%.1f", loadSeconds)) s; \(rows.count) questions")

        // Warm every bucket once so compile/first-call costs stay out of the per-question numbers.
        for length in manager.lengths {
            let filler = String(repeating: "warm up ", count: max(1, length / 3))
            _ = try? await manager.answer(state: filler, question: .noul("Warm?"))
        }

        struct SuiteStats {
            var n = 0
            var correct = 0
            var dropped = 0
            var agree = 0
            var compared = 0
            var maxDelta: Float = 0
            var latencies: [Double] = []
            var buckets: [Int: Int] = [:]
            var truncated = 0
        }
        var stats: [String: SuiteStats] = [:]
        var allLatencies: [Double] = []
        let started = Date()
        for row in rows {
            var suite = stats[row.suite, default: SuiteStats()]
            let t0 = DispatchTime.now().uptimeNanoseconds
            let answer: LayaAnswer
            do {
                answer = try await manager.answer(state: row.state, question: row.question)
            } catch {
                suite.dropped += 1
                stats[row.suite] = suite
                continue
            }
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            suite.n += 1
            suite.latencies.append(ms)
            allLatencies.append(ms)
            suite.buckets[answer.bucketLength, default: 0] += 1
            if answer.stateWasTruncated { suite.truncated += 1 }
            if answer.selectedIndex == row.gold { suite.correct += 1 }
            if let ref = reference["\(row.suite)#\(row.index)"], ref.dropped != true, let argmax = ref.argmax,
                let probabilities = ref.probabilities, probabilities.count == answer.probabilities.count
            {
                suite.compared += 1
                if argmax == answer.selectedIndex { suite.agree += 1 }
                let delta = zip(probabilities, answer.probabilities).map { abs($0 - $1) }.max() ?? 0
                suite.maxDelta = max(suite.maxDelta, delta)
            }
            stats[row.suite] = suite
        }
        let elapsed = Date().timeIntervalSince(started)

        func percentile(_ values: [Double], _ q: Double) -> Double {
            guard !values.isEmpty else { return 0 }
            let sorted = values.sorted()
            return sorted[min(sorted.count - 1, Int(Double(sorted.count) * q))]
        }
        var report: [String: Any] = [
            "lengths": manager.lengths, "questions": rows.count, "elapsed_s": elapsed, "load_s": loadSeconds,
            "latency_ms": ["p50": percentile(allLatencies, 0.5), "p95": percentile(allLatencies, 0.95)],
            "chip": chipName(),
        ]
        var suiteReports: [String: Any] = [:]
        func pad(_ text: String, _ width: Int) -> String {
            text.count >= width ? text : text + String(repeating: " ", count: width - text.count)
        }
        print(
            pad("suite", 26) + "     n      acc    agree  maxΔprob   p50 ms   p95 ms  trunc  buckets")
        for name in stats.keys.sorted() {
            let s = stats[name]!
            let accuracy = Double(s.correct) / Double(max(1, s.n))
            let agreement = s.compared > 0 ? Double(s.agree) / Double(s.compared) : Double.nan
            let buckets = s.buckets.keys.sorted().map { "L\($0):\(s.buckets[$0]!)" }.joined(separator: " ")
            let agreeText = s.compared > 0 ? String(format: "%.3f", agreement) : "  n/a"
            print(
                pad(name, 26)
                    + String(
                        format: " %5d %8.3f %8@ %9.4f %8.2f %8.2f %6d  ", s.n, accuracy, agreeText as NSString,
                        s.maxDelta, percentile(s.latencies, 0.5), percentile(s.latencies, 0.95), s.truncated)
                    + buckets)
            suiteReports[name] = [
                "n": s.n, "dropped": s.dropped, "accuracy": accuracy, "reference_compared": s.compared,
                "reference_argmax_agreement": s.compared > 0 ? agreement : NSNull(),
                "max_probability_delta_vs_reference": s.maxDelta, "state_truncated": s.truncated,
                "latency_ms": ["p50": percentile(s.latencies, 0.5), "p95": percentile(s.latencies, 0.95)],
                "buckets": Dictionary(uniqueKeysWithValues: s.buckets.map { ("L\($0.key)", $0.value) }),
            ]
        }
        report["suites"] = suiteReports
        print(
            String(
                format: "%d questions in %.1f s · p50 %.2f ms · p95 %.2f ms · load %.1f s", rows.count, elapsed,
                percentile(allLatencies, 0.5), percentile(allLatencies, 0.95), loadSeconds))
        if let path = options.report {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: path))
            print("wrote \(path)")
        }
    }

    private static func chipName() -> String {
        var size = 0
        sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0)
        var buffer = [CChar](repeating: 0, count: size)
        sysctlbyname("machdep.cpu.brand_string", &buffer, &size, nil, 0)
        return String(cString: buffer)
    }

    private static func printUsage() {
        print(
            """
            Usage: fluidaudiocli laya-benchmark --suites suites.jsonl [--reference reference-rows.jsonl]
                                                [--model-dir DIR] [--lengths 128,256,512,1024] [--only suite,…]
                                                [--limit N] [--report out.json]

            Suites and reference rows come from mobius models/computer-use/laya/coreml/benchmark.py.
            """)
    }
}
#endif
