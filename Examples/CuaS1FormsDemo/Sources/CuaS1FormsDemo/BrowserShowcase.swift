import AppKit
import CuaBrowserCore
import CuaDemoCore
import Foundation
import WebKit

/// Records actual browser actions on the public examples; no user profile is exported.
@MainActor
enum BrowserShowcase {
    struct FormRun: Encodable {
        let form: String
        let variant: String
        let decisions: [BrowserDecision]
        let correct: Int
        let filledFields: Int
        let events: [String: Int]
    }

    struct Report: Encodable {
        let datasetSHA256 = DemoCatalog.datasetSHA256
        let description =
            "Actual WKWebView DOM observations, real Core ML choices, dispatched events, independent DOM readback. Public examples only."
        let timingScope =
            "Trace timings include rendering activity and are not the isolated variant benchmark. Presentation paused 250 ms per control pair."
        let runs: [FormRun]
        let integrationChecks: [String]
    }

    static func run(session: BrowserSession, directory: URL) async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let labels = try VariantBenchmark.labels()
        var runs: [FormRun] = []
        var frame = 0
        func capture() async throws {
            let path = directory.appendingPathComponent(String(format: "frame-%04d.png", frame))
            try await snapshot(session: session, to: path)
            frame += 1
        }
        session.source.useExampleDetails()
        for index in session.source.scenarios.indices {
            session.source.selectScenario(index)
            try await session.reset()
            guard let scenario = session.source.scenario else { throw DemoError("Missing showcase form.") }
            // Initial observations must be exactly the original real fixture contexts.
            for lane in session.lanes {
                let controls = try await lane.driver.controls()
                let task = scenario.controls[0].prefix.components(separatedBy: "\n")[0]
                guard controls.map({ $0.context(task: task) }) == scenario.controls.map(\.context) else {
                    throw DemoError("Live browser contexts differ from the original fixture.")
                }
            }
            try await Task.sleep(for: .milliseconds(400))
            for _ in 0..<4 { try await capture() }
            try await session.run(frame: capture)
            for lane in session.lanes {
                let controls = try await lane.driver.controls()
                guard lane.decisions.count == scenario.controls.count else {
                    throw DemoError("Incomplete browser run.")
                }
                let correct = zip(lane.decisions, scenario.controls).filter { $0.selectedIndex == labels[$1.id] }.count
                guard correct == scenario.controls.count else {
                    throw DemoError("Browser predictions failed label parity.")
                }
                let events = try await lane.driver.counters()
                let mutations = lane.decisions.filter { ["fill", "check"].contains($0.effect) }.count
                guard events["submit"] == 0, events["input"] == mutations, events["change"] == mutations else {
                    throw DemoError("Browser event delivery or submission boundary failed.")
                }
                runs.append(
                    FormRun(
                        form: scenario.id, variant: lane.id, decisions: lane.decisions, correct: correct,
                        filledFields: controls.filter { $0.role == "Edit" && !$0.value.isEmpty }.count,
                        events: events))
            }
            // Return to the top so the completed form is easy to review in the recording.
            for lane in session.lanes {
                _ = try await lane.driver.script("window.scrollTo(0,0);return JSON.stringify(true);")
            }
            try await Task.sleep(for: .milliseconds(150))
            for _ in 0..<6 { try await capture() }
            try await snapshot(session: session, to: directory.appendingPathComponent("\(scenario.id).png"))
        }
        let checks = try await verifyDriverBoundaries(session)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        try encoder.encode(Report(runs: runs, integrationChecks: checks))
            .write(to: directory.appendingPathComponent("browser-validation.json"), options: .atomic)
        print(
            "PASS browser showcase: 100/100 labeled decisions, DOM readback, input/change events, no automatic submits")
    }

    private static func verifyDriverBoundaries(_ session: BrowserSession) async throws -> [String] {
        for lane in session.lanes {
            guard let observed = try await lane.driver.controls().first else { throw DemoError("No browser controls.") }
            // Change the real DOM between observation and action; the stale write must fail.
            _ = try await lane.driver.script(
                "document.getElementById(id).value='Edited during inference';return JSON.stringify(true);",
                arguments: ["id": observed.id])
            var rejected = false
            do { try await lane.driver.apply(observed.action("skip"), observed: observed) } catch { rejected = true }
            guard rejected else { throw DemoError("The driver accepted a stale control observation.") }
            _ = try await lane.driver.script(
                "document.getElementById(id).value=value;return JSON.stringify(true);",
                arguments: ["id": observed.id, "value": observed.value])
            // An explicit user-equivalent local button click produces a receipt.
            _ = try await lane.driver.script("document.querySelector('button').click();return JSON.stringify(true);")
            guard try await lane.driver.counters()["submit"] == 1 else {
                throw DemoError("The explicit local click failed.")
            }
        }
        return [
            "100/100 original choices", "Independent DOM readback after every action",
            "Input/change events match fill/check actions", "No model-triggered submissions",
            "Stale DOM observations rejected", "Explicit local button clicks produce receipts",
        ]
    }

    /// Composite this app's native view and WebKit's own snapshots; no screen-capture permission.
    static func snapshot(session: BrowserSession, to url: URL) async throws {
        guard let view = NSApplication.shared.windows.first(where: { $0.isVisible })?.contentView,
            let bitmap = view.bitmapImageRepForCachingDisplay(in: view.bounds), let contextImage = bitmap.cgImage
        else { throw DemoError("No browser demo window is available.") }
        view.cacheDisplay(in: view.bounds, to: bitmap)
        // Take WebKit snapshots separately because its composited surfaces may not be in cacheDisplay.
        var browserImages: [(NSImage, NSRect)] = []
        for lane in session.lanes {
            let webView = lane.driver.webView
            let snapshot = try await webView.takeSnapshot(configuration: nil)
            var rect = webView.convert(webView.bounds, to: view)
            if view.isFlipped { rect.origin.y = view.bounds.height - rect.maxY }
            browserImages.append((snapshot, rect))
        }
        let result = NSImage(size: view.bounds.size)
        result.lockFocus()
        NSImage(cgImage: bitmap.cgImage ?? contextImage, size: view.bounds.size).draw(in: view.bounds)
        for (image, rect) in browserImages { image.draw(in: rect) }
        result.unlockFocus()
        guard let tiff = result.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff),
            let data = rep.representation(using: .png, properties: [:])
        else { throw DemoError("Snapshot encoding failed.") }
        try data.write(to: url, options: .atomic)
    }
}
