import FluidAudio
import Foundation
import Observation

/// The local effect of a model-selected action.
public enum DemoEffect: String, Sendable {
    /// Text copied from a supplied document entity.
    case filled = "Filled"
    /// A checkbox was checked in the local sample form.
    case checked = "Checked"
    /// The model chose to leave the element unchanged.
    case skipped = "Skipped"
    /// A button click is proposed for the user to review, never executed automatically.
    case review = "Review"
}

/// Applies only compatible local form actions; submission is a separate user operation.
public enum DemoActions {
    /// Apply a candidate string to the matching kind of control.
    public static func apply(_ option: String, to control: inout DemoControl) throws -> DemoEffect {
        if option == "skip" { return .skipped }
        if option == "click", control.role == "Button" { return .review }
        if option == "check", control.role == "CheckBox" {
            control.isChecked = true
            return .checked
        }
        if let parts = DemoCatalog.fillParts(option), control.role == "Edit" {
            control.value = parts.value
            return .filled
        }
        throw DemoError("The chosen action is incompatible with this control; nothing was applied.")
    }
}

/// An actual Core ML prediction and the local effect of applying it.
public struct DemoDecision: Identifiable, Sendable {
    /// Original demo row, also the local control ID.
    public let id: Int
    /// Control description passed to the model in this call.
    public let context: String
    /// Original control label.
    public let title: String
    /// Live Swift manager output.
    public let result: CuaS1FormsResult
    /// Time spent awaiting the Swift scoring call, excluding animation.
    public let milliseconds: Double
    /// Local form effect.
    public let effect: DemoEffect
}

/// Main-actor state shared by the SwiftUI example and its real-model verification mode.
@MainActor
@Observable
public final class DemoSession {
    /// The original three forms, without answer labels.
    public private(set) var scenarios: [DemoScenario] = []
    /// Selected scenario index.
    public private(set) var scenarioIndex = 0
    /// Current mutable form controls.
    public private(set) var controls: [DemoControl] = []
    /// Predictions made during this pass, in form order.
    public private(set) var decisions: [DemoDecision] = []
    /// Control displayed in the decision inspector.
    public var selectedControlID: Int?
    /// Whether a scoring sequence is in progress.
    public private(set) var isRunning = false
    /// Whether the real Core ML model has loaded.
    public private(set) var isReady = false
    /// Whether the example is loading or compiling its model.
    public private(set) var isLoading = false
    /// Whether the user has submitted this local demo form.
    public private(set) var isSubmitted = false
    /// Recoverable load, input, or prediction error for the UI.
    public private(set) var errorMessage: String?
    /// Next control to score in the current pass.
    public private(set) var nextIndex = 0

    @ObservationIgnored private var manager: CuaS1FormsManager?
    @ObservationIgnored private var runTask: Task<Void, Never>?
    @ObservationIgnored private var generation = 0

    /// Create an unloaded session.
    public init() {}

    /// The currently selected scenario.
    public var scenario: DemoScenario? {
        scenarios.indices.contains(scenarioIndex) ? scenarios[scenarioIndex] : nil
    }

    /// Selected prediction, or the newest decision when no control is selected.
    public var inspectedDecision: DemoDecision? {
        guard let selectedControlID else { return decisions.last }
        return decisions.last { $0.id == selectedControlID }
    }

    /// Whether all controls have been considered in this pass.
    public var isComplete: Bool { !controls.isEmpty && nextIndex == controls.count }

    /// Total Swift scoring-call time, excluding the presentation delay.
    public var totalMilliseconds: Double { decisions.reduce(0) { $0 + $1.milliseconds } }

    /// Load the bundled source fixture and real Core ML package.
    public func load(modelURL: URL? = nil) async {
        guard !isLoading, !isReady else { return }
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            scenarios = try DemoCatalog.load()
            reset()
            let url: URL
            if let modelURL { url = modelURL } else { url = try await DemoAssets.modelURL() }
            manager = try await CuaS1FormsManager.load(from: url)
            isReady = true
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    /// Switch forms and clear the previous pass.
    public func selectScenario(_ index: Int) {
        guard scenarios.indices.contains(index) else { return }
        scenarioIndex = index
        reset()
    }

    /// Cancel pending work and reset all controls to the original empty state.
    public func reset() {
        stop()
        controls = scenario?.controls ?? []
        clearPass()
        isSubmitted = false
        errorMessage = nil
    }

    /// Keep the current form values while starting a new scoring pass.
    public func recheck() {
        guard !isRunning else { return }
        clearPass()
        isSubmitted = false
        start()
    }

    private func clearPass() {
        decisions = []
        nextIndex = 0
        selectedControlID = nil
    }

    /// Stop animation and prevent a pending prediction from mutating a newer form.
    public func stop() {
        generation += 1
        runTask?.cancel()
        runTask = nil
        isRunning = false
    }

    /// Start a paced sequence, or execute just one control when stepping.
    public func start(singleStep: Bool = false) {
        guard isReady, !isRunning, !isComplete else { return }
        isRunning = true
        errorMessage = nil
        let startedGeneration = generation
        runTask = Task {
            defer {
                if generation == startedGeneration {
                    isRunning = false
                    runTask = nil
                }
            }
            do {
                repeat {
                    try await scoreNext()
                    if singleStep || isComplete { break }
                    try await Task.sleep(for: .milliseconds(240))
                } while !Task.isCancelled
            } catch is CancellationError {
                // A reset or stop is an ordinary user action.
            } catch {
                if generation == startedGeneration { errorMessage = error.localizedDescription }
            }
        }
    }

    /// Perform one real prediction and apply only a compatible local effect.
    public func scoreNext() async throws {
        guard let manager, let scenario, controls.indices.contains(nextIndex) else {
            throw DemoError("Load a model and select an unfinished form first.")
        }
        let index = nextIndex
        let startedGeneration = generation
        let control = controls[index]
        selectedControlID = control.id
        let start = ContinuousClock.now
        let result = try await manager.score(context: control.context, options: scenario.options)
        let elapsed = start.duration(to: .now).components
        let milliseconds = Double(elapsed.seconds) * 1000 + Double(elapsed.attoseconds) / 1e15
        try Task.checkCancellation()
        guard generation == startedGeneration else { throw CancellationError() }
        guard !result.contextWasTruncated, result.truncatedOptionIndices.isEmpty else {
            throw DemoError("This input exceeds the model's byte limits. Shorten it before applying a decision.")
        }
        let effect = try DemoActions.apply(result.selectedOption, to: &controls[index])
        decisions.append(
            DemoDecision(
                id: control.id, context: control.context, title: control.title, result: result,
                milliseconds: milliseconds, effect: effect))
        nextIndex += 1
    }

    /// Change a field value before starting a new scoring pass.
    public func setValue(_ value: String, for id: Int) {
        guard !isRunning, let index = controls.firstIndex(where: { $0.id == id }) else { return }
        controls[index].value = value
        isSubmitted = false
    }

    /// Change a checkbox before starting a new scoring pass.
    public func setChecked(_ value: Bool, for id: Int) {
        guard !isRunning, let index = controls.firstIndex(where: { $0.id == id }) else { return }
        controls[index].isChecked = value
        isSubmitted = false
    }

    /// Record a local receipt only after the user presses the demo submit button.
    public func submitLocally() {
        guard isComplete, !isRunning else { return }
        isSubmitted = true
    }
}
