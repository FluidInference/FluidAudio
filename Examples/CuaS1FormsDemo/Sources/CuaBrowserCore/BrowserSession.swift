import CuaDemoCore
import FluidAudio
import Foundation
import Observation

/// An actual observation, prediction, and independently verified browser effect.
public struct BrowserDecision: Codable, Sendable {
    public let controlID: String
    public let title: String
    public let context: String
    public let options: [String]
    public let selectedIndex: Int
    public let selectedOption: String
    public let probability: Float
    public let effect: String
    public let scoreMilliseconds: Double
    public let loopMilliseconds: Double
}

/// One real model and its independently controlled browser.
@MainActor
@Observable
public final class BrowserLane: Identifiable {
    public let id: String
    public let title: String
    public let driver = BrowserDriver()
    public private(set) var decisions: [BrowserDecision] = []
    public private(set) var status = "Loading model"
    @ObservationIgnored private var manager: CuaS1FormsManager?

    init(id: String, title: String) {
        self.id = id
        self.title = title
    }

    func load(_ url: URL) async throws {
        manager = try await CuaS1FormsManager.load(from: url)
        status = "Ready · CPU + Neural Engine"
    }

    func reset(_ scenario: DemoScenario) async throws {
        decisions = []
        try await driver.load(scenario)
        status = "Ready · live DOM"
    }

    func step(task: String, options: [String], stopped: () -> Bool) async throws {
        guard let manager else { throw DemoError("Load the model first.") }
        let loopStart = ContinuousClock.now
        let controls = try await driver.controls()
        guard controls.indices.contains(decisions.count) else { throw DemoError("The browser control list changed.") }
        let control = controls[decisions.count]
        status = "Inspecting \(control.title)"
        try await driver.highlight(control.id)
        let context = control.context(task: task)
        let start = ContinuousClock.now
        let result = try await manager.score(context: context, options: options)
        let scoreTime = VariantBenchmark.elapsed(start)
        guard !stopped() else { throw CancellationError() }
        guard !result.contextWasTruncated, result.truncatedOptionIndices.isEmpty else {
            throw DemoError("Input exceeded the byte limits; the browser action was not applied.")
        }
        let action = try control.action(result.selectedOption)
        try await driver.apply(action, observed: control)
        decisions.append(
            BrowserDecision(
                controlID: control.id, title: control.title, context: context, options: options,
                selectedIndex: result.selectedIndex, selectedOption: result.selectedOption,
                probability: result.probabilities[result.selectedIndex], effect: action.kind,
                scoreMilliseconds: scoreTime, loopMilliseconds: VariantBenchmark.elapsed(loopStart)))
        status = "\(decisions.count)/\(controls.count) decisions · DOM verified"
    }
}

/// Coordinates two local browser agents while retaining the existing editable profile behavior.
@MainActor
@Observable
public final class BrowserSession {
    public private(set) var source = DemoSession()
    public let lanes = [
        BrowserLane(id: "baseline", title: "Original export"),
        BrowserLane(id: "ane-gather", title: "More ANE operations"),
    ]
    public private(set) var isReady = false
    public private(set) var isBusy = false
    public private(set) var error: String?
    public private(set) var status = "Preparing both Core ML models…"
    @ObservationIgnored private var stopped = false

    public init() {}

    /// Load real pinned exports and a fresh page in both independent browsers.
    public func load(baseline: URL? = nil, candidate: URL? = nil) async throws {
        source = DemoSession(scenarios: try DemoCatalog.load())
        let baselineURL: URL
        let candidateURL: URL
        if let baseline { baselineURL = baseline } else { baselineURL = try await DemoAssets.modelURL() }
        if let candidate {
            candidateURL = candidate
        } else {
            candidateURL = try await DemoAssets.modelURL(aneGather: true)
        }
        try await lanes[0].load(baselineURL)
        try await lanes[1].load(candidateURL)
        try await reset()
        isReady = true
        status = "Enter details or use an example, then run both agents."
    }

    public var isComplete: Bool { lanes.allSatisfy { $0.decisions.count == source.controls.count } }

    /// Reload live pages when switching forms or changing source values.
    public func reset() async throws {
        guard !isBusy, let scenario = source.scenario else { return }
        isBusy = true
        defer { isBusy = false }
        error = nil
        for lane in lanes { try await lane.reset(scenario) }
        status = "Ready to inspect \(scenario.controls.count) live controls in each browser."
    }

    /// Stop before the next action; the in-flight prediction may finish but cannot change the DOM.
    public func stop() { stopped = true }

    /// Run both real agents on the selected form. The optional callback captures visible progress.
    public func run(singleStep: Bool = false, frame: (@MainActor () async throws -> Void)? = nil) async throws {
        guard isReady, !isBusy, !isComplete else { return }
        let options = try ProfileChoices.make(from: source.details)
        guard let task = source.scenario?.controls.first?.prefix.components(separatedBy: "\n").first else {
            throw DemoError("The selected form has no task.")
        }
        isBusy = true
        stopped = false
        error = nil
        defer { isBusy = false }
        do {
            repeat {
                let next = lanes.map { $0.decisions.count }.min() ?? 0
                for lane in lanes where lane.decisions.count == next {
                    guard !stopped else { throw CancellationError() }
                    try await lane.step(task: task, options: options, stopped: { self.stopped })
                }
                status = "Read live control → CUA decision → browser action → verify DOM"
                try await Task.sleep(for: .milliseconds(250))
                try await frame?()
                if singleStep { break }
            } while !isComplete && !stopped
            if isComplete { status = "Both forms filled. Review before clicking a submit button." }
        } catch is CancellationError {
            status = "Stopped. Continue with Run both agents."
        } catch {
            self.error = error.localizedDescription
            throw error
        }
    }

    /// Surface recoverable errors from UI tasks without silently dropping them.
    public func showError(_ error: Error) { self.error = error.localizedDescription }
}
