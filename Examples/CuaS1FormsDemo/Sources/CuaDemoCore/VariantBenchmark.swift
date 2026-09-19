import CoreML
import CryptoKit
import FluidAudio
import Foundation

/// A release-mode Swift comparison on a pinned, explicitly bounded manifest.
public enum VariantBenchmark {
    /// A timed Swift manager call, with correctness checked outside the timer.
    public struct Sample: Codable, Sendable {
        public let variant: String
        public let block: Int
        public let pass: Int
        public let row: Int
        public let form: String
        public let milliseconds: Double
        public let selectedIndex: Int
        public let expectedIndex: Int
    }

    /// Warm-call statistics; p95 uses linear interpolation over sorted observations.
    public struct Summary: Codable, Sendable {
        public let calls: Int
        public let correct: Int
        public let medianMilliseconds: Double
        public let p95Milliseconds: Double
    }

    /// Recorded model provenance and load cost. System caches may already be warm.
    public struct Artifact: Codable, Sendable {
        public let filesSHA256: [String: String]
        public let loadMilliseconds: Double
        public let firstCallMilliseconds: Double
    }

    /// Machine-readable comparison, including every timed sample.
    public struct Report: Codable, Sendable {
        public let timestamp: String
        public let operatingSystem: String
        public let hardware: String
        public let build: String
        public var computeUnits = "cpuAndNeuralEngine"
        public var timingScope =
            "Swift manager.score: encoding + actor call + Core ML + output decoding; no UI or animation"
        public var protocolDescription = "Both resident; one 50-row warmup each; ABBA; two 50-row passes per block"
        public var limitations =
            "Exploratory local timing on 3 supplied forms; not a held-out generalization benchmark. Load may use system caches."
        public let datasetSHA256: String
        public let rows: [Int]
        public let artifacts: [String: Artifact]
        public let summaries: [String: Summary]
        public let forms: [String: [String: Summary]]
        public let samples: [Sample]
    }

    private struct Label: Decodable { let label: Int }

    /// Evaluation-only answer keys; these are never used to choose or apply browser actions.
    public static func labels() throws -> [Int] {
        try DemoCatalog.fixtureData().split(separator: 10).map {
            try JSONDecoder().decode(Label.self, from: Data($0)).label
        }
    }

    /// Interpolated percentile, defined for the nonempty samples produced by this harness.
    public static func percentile(_ values: [Double], fraction: Double) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let position = min(1, max(0, fraction)) * Double(sorted.count - 1)
        let lower = Int(position)
        let upper = min(lower + 1, sorted.count - 1)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
    }

    /// Summarize measured samples without dropping errors or slow calls.
    public static func summarize(_ samples: [Sample]) -> Summary {
        Summary(
            calls: samples.count, correct: samples.filter { $0.selectedIndex == $0.expectedIndex }.count,
            medianMilliseconds: percentile(samples.map(\.milliseconds), fraction: 0.5),
            p95Milliseconds: percentile(samples.map(\.milliseconds), fraction: 0.95))
    }

    /// Measure both exports using the same real 50-control manifest, with no rendering work.
    public static func run(baseline: URL, candidate: URL, output: URL, hardware: String) async throws -> Report {
        let scenarios = try DemoCatalog.load()
        let labels = try labels()
        let rows = scenarios.flatMap(\.controls)
        guard rows.count == 50, let first = scenarios.first?.controls.first, let options = scenarios.first?.options
        else {
            throw DemoError("The 50-control benchmark manifest changed.")
        }
        var managers: [String: CuaS1FormsManager] = [:]
        var artifacts: [String: Artifact] = [:]
        for (name, url) in [("baseline", baseline), ("ane-gather", candidate)] {
            let hashes = try packageHashes(url)
            let start = ContinuousClock.now
            let manager = try await CuaS1FormsManager.load(from: url, computeUnits: .cpuAndNeuralEngine)
            let load = elapsed(start)
            let callStart = ContinuousClock.now
            let prediction = try await manager.score(context: first.context, options: options)
            let firstCall = elapsed(callStart)
            guard prediction.selectedIndex == labels[first.id] else { throw DemoError("First-call parity failed.") }
            managers[name] = manager
            artifacts[name] = Artifact(filesSHA256: hashes, loadMilliseconds: load, firstCallMilliseconds: firstCall)
        }
        var samples: [Sample] = []
        // Warmup blocks retain the same validation as measured blocks but are not reported as timings.
        for (block, name) in ["baseline", "ane-gather", "baseline", "ane-gather", "ane-gather", "baseline"].enumerated()
        {
            guard let manager = managers[name] else { throw DemoError("A benchmark model is missing.") }
            for pass in 0..<(block < 2 ? 1 : 2) {
                for scenario in scenarios {
                    for control in scenario.controls {
                        let start = ContinuousClock.now
                        let result = try await manager.score(context: control.context, options: scenario.options)
                        let milliseconds = elapsed(start)
                        guard !result.contextWasTruncated, result.truncatedOptionIndices.isEmpty,
                            result.probabilities.allSatisfy({ $0.isFinite })
                        else { throw DemoError("Invalid or truncated benchmark output for row \(control.id).") }
                        guard result.selectedIndex == labels[control.id] else {
                            throw DemoError("\(name) disagreed with upstream row \(control.id).")
                        }
                        if block < 2 { continue }
                        samples.append(
                            Sample(
                                variant: name, block: block - 2, pass: pass, row: control.id, form: scenario.id,
                                milliseconds: milliseconds, selectedIndex: result.selectedIndex,
                                expectedIndex: labels[control.id]))
                    }
                }
            }
        }
        let names = ["baseline", "ane-gather"]
        let summaries = Dictionary(
            uniqueKeysWithValues: names.map { name in
                (name, summarize(samples.filter { $0.variant == name }))
            })
        let forms = Dictionary(
            uniqueKeysWithValues: scenarios.map { scenario in
                (
                    scenario.id,
                    Dictionary(
                        uniqueKeysWithValues: names.map { name in
                            (name, summarize(samples.filter { $0.variant == name && $0.form == scenario.id }))
                        })
                )
            })
        #if DEBUG
        let build = "debug"
        #else
        let build = "release"
        #endif
        let report = Report(
            timestamp: ISO8601DateFormatter().string(from: Date()),
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString, hardware: hardware, build: build,
            datasetSHA256: DemoCatalog.datasetSHA256, rows: rows.map(\.id), artifacts: artifacts,
            summaries: summaries, forms: forms, samples: samples)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(report).write(to: output, options: .atomic)
        for name in names {
            guard let summary = summaries[name] else { continue }
            print(
                String(
                    format: "%@: %d/%d correct; p50 %.3f ms, p95 %.3f ms", name,
                    summary.correct, summary.calls, summary.medianMilliseconds, summary.p95Milliseconds))
        }
        return report
    }

    /// Monotonic milliseconds since a captured clock instant.
    public static func elapsed(_ start: ContinuousClock.Instant) -> Double {
        let duration = start.duration(to: .now).components
        return Double(duration.seconds) * 1000 + Double(duration.attoseconds) / 1e15
    }

    private static func packageHashes(_ url: URL) throws -> [String: String] {
        guard url.pathExtension == "mlpackage" else {
            throw DemoError("Benchmark provenance requires .mlpackage inputs.")
        }
        return try Dictionary(
            uniqueKeysWithValues: [
                "Manifest.json", "Data/com.apple.CoreML/model.mlmodel",
                "Data/com.apple.CoreML/weights/weight.bin",
            ].map { file in
                let data = try Data(contentsOf: url.appendingPathComponent(file))
                return (file, SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
            })
    }
}
