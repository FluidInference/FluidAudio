import XCTest

@testable import CuaDemoCore

final class DemoCoreTests: XCTestCase {
    func testPinnedInitialFormsRetainAllCandidates() throws {
        let scenarios = try DemoCatalog.load()
        XCTAssertEqual(scenarios.map(\.controls.count), [18, 15, 17])
        XCTAssertEqual(scenarios.map(\.id), ["patient-registration", "job-application", "auto-claim"])
        XCTAssertEqual(scenarios[0].controls[0].id, 0)
        XCTAssertEqual(scenarios[1].controls[0].id, 68)
        XCTAssertEqual(scenarios[2].controls[0].id, 130)
        XCTAssertEqual(scenarios[0].options.count, 27)
        XCTAssertEqual(scenarios[0].entities.count, 24)
        XCTAssertTrue(scenarios[0].options.contains("fill Printed: 09/17/2026"))
        XCTAssertTrue(scenarios.allSatisfy { $0.controls.allSatisfy { $0.value.isEmpty && !$0.isChecked } })
    }

    func testFillActionPreservesValueAndChangesContext() throws {
        var control = try DemoCatalog.load()[0].controls[0]
        let effect = try DemoActions.apply("fill First name: Amara", to: &control)
        XCTAssertEqual(effect, .filled)
        XCTAssertEqual(control.value, "Amara")
        XCTAssertTrue(control.context.hasSuffix("ELEMENT Edit \"First name\" value=\"Amara\""))
        XCTAssertEqual(try DemoActions.apply("skip", to: &control), .skipped)
        XCTAssertEqual(control.value, "Amara")
    }

    func testCheckAndClickHaveDifferentLocalEffects() throws {
        let scenario = try DemoCatalog.load()[0]
        var checkbox = try XCTUnwrap(scenario.controls.first { $0.role == "CheckBox" })
        XCTAssertEqual(try DemoActions.apply("check", to: &checkbox), .checked)
        XCTAssertTrue(checkbox.isChecked)
        XCTAssertTrue(checkbox.context.hasSuffix(" checked"))
        var submit = try XCTUnwrap(scenario.controls.last)
        XCTAssertEqual(try DemoActions.apply("click", to: &submit), .review)
        XCTAssertEqual(submit.value, "")
    }

    func testIncompatibleActionsDoNotMutateControls() throws {
        var control = try DemoCatalog.load()[0].controls[0]
        XCTAssertThrowsError(try DemoActions.apply("check", to: &control))
        XCTAssertThrowsError(try DemoActions.apply("click", to: &control))
        XCTAssertTrue(control.value.isEmpty)
        XCTAssertFalse(control.isChecked)
    }

    func testFillParsingRetainsColonsInValues() {
        let parts = DemoCatalog.fillParts("fill Website: https://example.com:8443/profile")
        XCTAssertEqual(parts?.name, "Website")
        XCTAssertEqual(parts?.value, "https://example.com:8443/profile")
        XCTAssertNil(DemoCatalog.fillParts("skip"))
        XCTAssertNil(DemoCatalog.fillParts("fill without separator"))
    }
}
