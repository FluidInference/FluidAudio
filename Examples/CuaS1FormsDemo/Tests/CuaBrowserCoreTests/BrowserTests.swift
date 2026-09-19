import CuaDemoCore
import Foundation
import XCTest

@testable import CuaBrowserCore

final class BrowserTests: XCTestCase {
    func testLiveContextUsesObservedValuesAndMatchesInitialFixture() throws {
        for scenario in try DemoCatalog.load() {
            for original in scenario.controls {
                let control = BrowserControl(
                    id: "control-\(original.id)", role: original.role, title: original.title,
                    value: original.value, checked: original.isChecked, form: scenario.title)
                let task = String(original.prefix.split(separator: "\n")[0])
                XCTAssertEqual(control.context(task: task), original.context)
            }
        }
        let observed = BrowserControl(
            id: "control-68", role: "Edit", title: "First name", value: "Kenji",
            checked: false, form: "Changed visible title")
        XCTAssertTrue(observed.context(task: "TASK fill form").contains("FORM Changed visible title"))
        XCTAssertTrue(observed.context(task: "TASK fill form").hasSuffix("value=\"Kenji\""))
    }

    func testBrowserActionsPreservePunctuationAndRejectWrongRoles() throws {
        let control = BrowserControl(
            id: "control-68", role: "Edit", title: "First name", value: "",
            checked: false, form: "Form")
        XCTAssertEqual(try control.action("fill Note: A: B < C & D").value, "A: B < C & D")
        XCTAssertThrowsError(try control.action("check"))
        XCTAssertThrowsError(try control.action("click"))
        XCTAssertThrowsError(try control.action("unrecognized"))
        let button = BrowserControl(
            id: "button", role: "Button", title: "Submit", value: "",
            checked: false, form: "Form")
        XCTAssertEqual(try button.action("click").kind, "review")
        XCTAssertThrowsError(try button.action("fill First name: Kenji"))
    }

    func testPagesContainControlsButNoSourceValuesOrAnswers() throws {
        for scenario in try DemoCatalog.load() {
            let html = BrowserPage.html(scenario)
            for control in scenario.controls { XCTAssertTrue(html.contains("id='control-\(control.id)'")) }
            let email = try XCTUnwrap(scenario.entities.first { $0.value.contains("@") })
            XCTAssertFalse(html.contains(email.value))
            XCTAssertFalse(html.contains("selectedIndex"))
            XCTAssertFalse(html.contains("expectedIndex"))
        }
        XCTAssertEqual(BrowserPage.escape("<script>\"x\" & 'y'"), "&lt;script&gt;&quot;x&quot; &amp; &#39;y&#39;")
    }

    func testPercentilesRetainSlowSamplesAndUseInterpolation() {
        XCTAssertEqual(VariantBenchmark.percentile([1, 2, 3, 4, 100], fraction: 0.5), 3)
        XCTAssertEqual(VariantBenchmark.percentile([1, 2, 3, 4, 100], fraction: 0.95), 80.8, accuracy: 0.00001)
        XCTAssertEqual(VariantBenchmark.percentile([7], fraction: 0.95), 7)
    }
}
