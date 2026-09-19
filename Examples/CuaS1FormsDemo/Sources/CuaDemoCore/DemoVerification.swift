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
        let expectedFilledFields = [14, 10, 12]
        session.useExampleDetails()
        for scenarioIndex in session.scenarios.indices {
            session.selectScenario(scenarioIndex)
            guard session.isUsingExampleDetails,
                try ProfileChoices.make(from: session.details) == session.scenarios[scenarioIndex].options
            else { throw DemoError("The public example did not follow the selected form.") }
            while !session.isComplete { try await session.scoreNext() }
            for decision in session.decisions {
                guard decision.result.selectedIndex == labels[decision.id] else {
                    throw DemoError("Wrong decision for upstream row \(decision.id).")
                }
                checked += 1
            }
            let filledFields = session.controls.filter { $0.role == "Edit" && !$0.value.isEmpty }.count
            guard filledFields == expectedFilledFields[scenarioIndex] else {
                throw DemoError("The example did not fill the expected fields for form \(scenarioIndex).")
            }
            guard !session.isSubmitted else { throw DemoError("Scoring must never submit a form.") }
            session.submitLocally()
            guard session.isSubmitted else { throw DemoError("The explicit local submit action failed.") }
            print(
                "PASS \(session.scenario?.shortTitle ?? "form"): "
                    + "\(session.decisions.count) real decisions, \(filledFields) text fields filled")
        }
        guard checked == 50 else { throw DemoError("The selected 50-control manifest changed.") }
        session.selectScenario(0)
        session.useExampleDetails()
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
        try verifyExampleEditing(session)
        try await verifyEnteredProfile(session)
        print(
            "Verified 50 original form decisions, filled-state recheck, and playback controls with the real Swift manager."
        )
    }

    @MainActor
    private static func verifyExampleEditing(_ session: DemoSession) throws {
        session.useExampleDetails()
        guard let detail = session.details.first else { throw DemoError("Missing example details.") }
        session.updateDetail(detail.id, name: detail.name, value: detail.value)
        session.selectScenario(1)
        guard session.isUsingExampleDetails,
            try ProfileChoices.make(from: session.details) == session.scenarios[1].options
        else { throw DemoError("An unchanged editor update stopped example switching.") }
        guard let firstName = session.details.first(where: { $0.name == "First name" }) else {
            throw DemoError("Missing example first name.")
        }
        session.updateDetail(firstName.id, value: "Kenji")
        let editedChoices = try ProfileChoices.make(from: session.details)
        session.selectScenario(2)
        guard !session.isUsingExampleDetails,
            try ProfileChoices.make(from: session.details) == editedChoices
        else { throw DemoError("Switching forms overwrote an edited example.") }
        session.useExampleDetails()
        guard session.isUsingExampleDetails,
            try ProfileChoices.make(from: session.details) == session.scenarios[2].options
        else { throw DemoError("Use example did not restore the selected form's details.") }
        session.clearDetails()
        session.selectScenario(0)
        guard !session.isUsingExampleDetails, session.details.allSatisfy({ $0.value.isEmpty }) else {
            throw DemoError("Switching forms restored sample data after clearing the profile.")
        }
        print("PASS examples follow forms, edited profiles persist, and clear exits example mode")
    }

    @MainActor
    private static func verifyEnteredProfile(_ session: DemoSession) async throws {
        session.clearDetails()
        // Public sample values entered into a blank profile, reused across all forms.
        let supplied = [
            "First name": "Kenji", "Last name": "Tanaka",
            "Email": "ktanaka42@outlook.com", "Phone": "(720) 555-0186",
        ]
        for (name, value) in supplied {
            guard let detail = session.details.first(where: { $0.name == name }) else {
                throw DemoError("Missing profile input.")
            }
            session.updateDetail(detail.id, value: value)
        }
        let expectedValues = [
            "First name": "Kenji", "Last name": "Tanaka", "Full name": "Kenji Tanaka",
            "Email address": "ktanaka42@outlook.com", "Email": "ktanaka42@outlook.com",
            "Phone number": "(720) 555-0186", "Mobile phone": "(720) 555-0186",
            "Daytime phone": "(720) 555-0186",
        ]
        let choices = try ProfileChoices.make(from: session.details)
        guard choices.count == 8, choices.contains("fill Name: Kenji Tanaka") else {
            throw DemoError("Blank entries or combined-name construction changed the options.")
        }
        for index in session.scenarios.indices {
            session.selectScenario(index)
            while !session.isComplete { try await session.scoreNext() }
            guard session.decisions.allSatisfy({ $0.options == choices }) else {
                throw DemoError("A decision used stale or sample options instead of the entered profile.")
            }
            for control in session.controls where control.role == "Edit" {
                let expected = expectedValues[control.title] ?? ""
                guard control.value == expected else {
                    throw DemoError("Entered-profile result needs review for \(control.title) on form \(index).")
                }
            }
            print("PASS entered profile on \(session.scenario?.shortTitle ?? "form")")
        }
        guard let email = session.details.first(where: { $0.name == "Email" }) else {
            throw DemoError("Profile was lost when switching forms.")
        }
        let priorCount = session.decisions.count
        session.updateDetail(email.id, value: email.value)
        guard session.decisions.count == priorCount else {
            throw DemoError("An unchanged editor value invalidated the current results.")
        }
        session.updateDetail(email.id, value: "")
        let updatedChoices = try ProfileChoices.make(from: session.details)
        guard session.decisions.isEmpty, session.nextIndex == 0,
            session.controls.allSatisfy({ $0.value.isEmpty && !$0.isChecked }),
            !updatedChoices.contains(where: { $0.hasPrefix("fill Email:") })
        else { throw DemoError("Editing the profile did not invalidate old choices and results.") }
        print("PASS source edits reset the preview and remove old candidates")
    }
}
