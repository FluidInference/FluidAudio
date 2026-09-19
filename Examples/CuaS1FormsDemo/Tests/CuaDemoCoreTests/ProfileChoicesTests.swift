import XCTest

@testable import CuaDemoCore

final class ProfileChoicesTests: XCTestCase {
    func testBlankProfileHasNoHiddenSampleValues() {
        XCTAssertThrowsError(try ProfileChoices.make(from: ProfileChoices.emptyDetails()))
        XCTAssertTrue(ProfileChoices.emptyDetails().allSatisfy { $0.value.isEmpty })
    }

    func testOnlySuppliedValuesBecomeOptions() throws {
        let details = [ProfileDetail(name: "Email", value: "person@example.com"), ProfileDetail(name: "Phone")]
        XCTAssertEqual(
            try ProfileChoices.make(from: details), ["fill Email: person@example.com", "check", "click", "skip"])
    }

    func testCombinedNameDoesNotReplaceAnExplicitFullName() throws {
        var details = [
            ProfileDetail(name: "First name", value: "Kenji"), ProfileDetail(name: "Last name", value: "Tanaka"),
        ]
        XCTAssertEqual(ProfileChoices.combinedName(in: details), "Kenji Tanaka")
        details.append(ProfileDetail(name: "Name", value: "Kenji T. Tanaka"))
        XCTAssertNil(ProfileChoices.combinedName(in: details))
        XCTAssertEqual(try ProfileChoices.make(from: details).count, 6)
    }

    func testExampleChoicesRemainByteForByteEquivalent() throws {
        for scenario in try DemoCatalog.load() {
            let details = scenario.entities.map { ProfileDetail(name: $0.name, value: $0.value) }
            XCTAssertEqual(try ProfileChoices.make(from: details), scenario.options)
        }
    }

    func testInvalidLabelsAndLongUTF8ValuesAreRejected() {
        for detail in [
            ProfileDetail(name: "", value: "value"),
            ProfileDetail(name: "Email: personal", value: "person@example.com"),
            ProfileDetail(name: "Two\nlines", value: "value"),
            ProfileDetail(name: "Name", value: String(repeating: "🙂", count: 24)),
        ] {
            XCTAssertThrowsError(try ProfileChoices.make(from: [detail]))
        }
    }

    func testCapacityIncludesActionsAndCombinedName() throws {
        var details = (0..<29).map { ProfileDetail(name: "Detail \($0)", value: "value") }
        XCTAssertEqual(try ProfileChoices.make(from: details).count, 32)
        details.append(ProfileDetail(name: "Extra", value: "value"))
        XCTAssertThrowsError(try ProfileChoices.make(from: details))
        details =
            Array(details.prefix(27)) + [
                ProfileDetail(name: "First name", value: "Kenji"), ProfileDetail(name: "Last name", value: "Tanaka"),
            ]
        XCTAssertThrowsError(try ProfileChoices.make(from: details))
    }
}
