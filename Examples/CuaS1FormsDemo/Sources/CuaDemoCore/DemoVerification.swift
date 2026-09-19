import Foundation

/// Bounded real-model verification, usable without the XCTest framework.
public enum DemoVerification {
    private struct LabeledRow: Decodable { let label: Int }

    /// Check the exact three initial forms and one filled-state decision; never execute a remote action.
    @MainActor
    public static func run(modelURL: URL?) async throws {
        let labels = try DemoCatalog.fixtureData().split(separator: 10).map {
            try JSONDecoder().decode(LabeledRow.self, from: Data($0)).label
        }
        let session = DemoSession()
        await session.load(modelURL: modelURL)
        guard session.isReady else { throw DemoError(session.errorMessage ?? "Model failed to load.") }
        var checked = 0
        for scenarioIndex in session.scenarios.indices {
            session.selectScenario(scenarioIndex)
            while !session.isComplete { try await session.scoreNext() }
            for decision in session.decisions {
                guard decision.result.selectedIndex == labels[decision.id] else {
                    throw DemoError("Wrong decision for upstream row \(decision.id).")
                }
                checked += 1
            }
            guard !session.isSubmitted else { throw DemoError("Scoring must never submit a form.") }
            session.submitLocally()
            guard session.isSubmitted else { throw DemoError("The explicit local submit action failed.") }
            print("PASS \(session.scenario?.shortTitle ?? "form"): \(session.decisions.count) real decisions")
        }
        guard checked == 50 else { throw DemoError("The selected 50-control manifest changed.") }
        session.selectScenario(0)
        try await session.scoreNext()
        let filledValue = session.controls[0].value
        session.reset()
        session.setValue(filledValue, for: session.controls[0].id)
        try await session.scoreNext()
        guard session.decisions.last?.effect == .skipped, session.controls[0].value == filledValue else {
            throw DemoError("The filled-state recheck did not preserve the existing field.")
        }
        print("PASS filled-state recheck: model selected skip and preserved the value")
        session.reset()
        session.start(singleStep: true)
        while session.isRunning { try await Task.sleep(for: .milliseconds(5)) }
        guard session.decisions.count == 1, session.nextIndex == 1 else {
            throw DemoError("Single-step did not execute exactly one decision.")
        }
        session.start()
        session.reset()
        try await Task.sleep(for: .milliseconds(20))
        guard session.decisions.isEmpty, session.nextIndex == 0, !session.isRunning,
            session.controls.allSatisfy({ $0.value.isEmpty && !$0.isChecked })
        else { throw DemoError("Reset did not cancel pending scoring safely.") }
        print("PASS single-step and cancellation/reset behavior")
        print(
            "Verified 50 original form decisions, filled-state recheck, and playback controls with the real Swift manager."
        )
    }
}
