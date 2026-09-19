import Foundation
import XCTest

@testable import CuaDemoCore

final class ExampleSelectionTests: XCTestCase {
    @MainActor
    func testUntouchedExamplesFollowAllFormsAndResetThePreview() async throws {
        let session = DemoSession(scenarios: try DemoCatalog.load())
        session.useExampleDetails()
        for index in [1, 2, 0] {
            let control = try XCTUnwrap(session.controls.first)
            session.setValue("A field under review", for: control.id)
            session.selectScenario(index)
            XCTAssertTrue(session.isUsingExampleDetails)
            XCTAssertEqual(try ProfileChoices.make(from: session.details), session.scenarios[index].options)
            XCTAssertEqual(session.controls.map(\.id), session.scenarios[index].controls.map(\.id))
            XCTAssertTrue(session.controls.allSatisfy { $0.value.isEmpty && !$0.isChecked })
            XCTAssertEqual(session.nextIndex, 0)
            XCTAssertTrue(session.decisions.isEmpty)
        }
    }

    @MainActor
    func testUnchangedEditorUpdatesKeepExamplesFollowingForms() async throws {
        let session = DemoSession(scenarios: try DemoCatalog.load())
        session.useExampleDetails()
        let detail = try XCTUnwrap(session.details.first)
        let control = try XCTUnwrap(session.controls.first)
        session.setValue("A field under review", for: control.id)
        session.updateDetail(detail.id, name: detail.name, value: detail.value)
        session.removeDetail(UUID())
        session.selectScenario(session.scenarioIndex)
        XCTAssertEqual(session.controls.first?.value, "A field under review")
        XCTAssertEqual(session.details.first?.id, detail.id)
        XCTAssertTrue(session.isUsingExampleDetails)
        session.selectScenario(2)
        XCTAssertEqual(try ProfileChoices.make(from: session.details), session.scenarios[2].options)
    }

    @MainActor
    func testEditingAnExamplePreservesTheEditedProfileAcrossForms() async throws {
        let session = DemoSession(scenarios: try DemoCatalog.load())
        session.useExampleDetails()
        let firstName = try XCTUnwrap(session.details.first { $0.name == "First name" })
        session.updateDetail(firstName.id, value: "Kenji")
        XCTAssertFalse(session.isUsingExampleDetails)
        let choices = try ProfileChoices.make(from: session.details)
        let ids = session.details.map(\.id)
        for index in [1, 2, 0] {
            session.selectScenario(index)
            XCTAssertEqual(try ProfileChoices.make(from: session.details), choices)
            XCTAssertEqual(session.details.map(\.id), ids)
        }
        session.useExampleDetails()
        session.selectScenario(1)
        XCTAssertTrue(session.isUsingExampleDetails)
        XCTAssertEqual(try ProfileChoices.make(from: session.details), session.scenarios[1].options)
    }

    @MainActor
    func testRenamingAddingOrRemovingDetailsLeavesExampleMode() async throws {
        let session = DemoSession(scenarios: try DemoCatalog.load())
        for edit in 0..<3 {
            session.useExampleDetails()
            let detail = try XCTUnwrap(session.details.first)
            switch edit {
            case 0: session.updateDetail(detail.id, name: "Document title")
            case 1: session.addDetail()
            default: session.removeDetail(detail.id)
            }
            XCTAssertFalse(session.isUsingExampleDetails)
            let ids = session.details.map(\.id)
            let choices = try ProfileChoices.make(from: session.details)
            session.selectScenario((session.scenarioIndex + 1) % session.scenarios.count)
            XCTAssertEqual(session.details.map(\.id), ids)
            XCTAssertEqual(try ProfileChoices.make(from: session.details), choices)
        }
    }

    @MainActor
    func testClearKeepsFormsBlankUntilAnExampleIsExplicitlyLoaded() async throws {
        let session = DemoSession(scenarios: try DemoCatalog.load())
        XCTAssertFalse(session.isUsingExampleDetails)
        session.useExampleDetails()
        session.clearDetails()
        for index in [1, 2, 0] {
            session.selectScenario(index)
            XCTAssertFalse(session.isUsingExampleDetails)
            XCTAssertTrue(session.details.allSatisfy { $0.value.isEmpty })
        }
        session.selectScenario(2)
        session.useExampleDetails()
        XCTAssertEqual(try ProfileChoices.make(from: session.details), session.scenarios[2].options)
        session.reset()
        XCTAssertTrue(session.isUsingExampleDetails)
        session.selectScenario(1)
        XCTAssertEqual(try ProfileChoices.make(from: session.details), session.scenarios[1].options)
    }
}
