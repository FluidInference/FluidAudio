import Foundation
import XCTest

@testable import FluidAudio

final class NemotronChunkSizeTests: XCTestCase {

    // MARK: - P1: Raw Value

    func testRawValues() {
        XCTAssertEqual(NemotronChunkSize.ms1120.rawValue, 1120)
        XCTAssertEqual(NemotronChunkSize.ms560.rawValue, 560)
        XCTAssertEqual(NemotronChunkSize.ms160.rawValue, 160)
        XCTAssertEqual(NemotronChunkSize.ms80.rawValue, 80)
    }

    func testInitFromRawValue() {
        XCTAssertEqual(NemotronChunkSize(rawValue: 1120), .ms1120)
        XCTAssertEqual(NemotronChunkSize(rawValue: 560), .ms560)
        XCTAssertEqual(NemotronChunkSize(rawValue: 160), .ms160)
        XCTAssertEqual(NemotronChunkSize(rawValue: 80), .ms80)
    }

    func testInvalidRawValueReturnsNil() {
        XCTAssertNil(NemotronChunkSize(rawValue: 100))
        XCTAssertNil(NemotronChunkSize(rawValue: 320))
        XCTAssertNil(NemotronChunkSize(rawValue: 0))
        XCTAssertNil(NemotronChunkSize(rawValue: 9999))
    }

    // MARK: - P1: Repo Mapping

    func testRepoMapping() {
        XCTAssertEqual(NemotronChunkSize.ms1120.repo, .nemotronStreaming1120)
        XCTAssertEqual(NemotronChunkSize.ms560.repo, .nemotronStreaming560)
        XCTAssertEqual(NemotronChunkSize.ms160.repo, .nemotronStreaming160)
        XCTAssertEqual(NemotronChunkSize.ms80.repo, .nemotronStreaming80)
    }

    // MARK: - P1: Subdirectory Generation

    func testSubdirectoryGeneration() {
        XCTAssertEqual(NemotronChunkSize.ms1120.subdirectory, "nemotron_coreml_1120ms")
        XCTAssertEqual(NemotronChunkSize.ms560.subdirectory, "nemotron_coreml_560ms")
        XCTAssertEqual(NemotronChunkSize.ms160.subdirectory, "nemotron_coreml_160ms")
        XCTAssertEqual(NemotronChunkSize.ms80.subdirectory, "nemotron_coreml_80ms")
    }

    // MARK: - P1: CaseIterable

    func testAllCasesContainsAllVariants() {
        let allCases = NemotronChunkSize.allCases
        XCTAssertEqual(allCases.count, 4)
        XCTAssertTrue(allCases.contains(.ms1120))
        XCTAssertTrue(allCases.contains(.ms560))
        XCTAssertTrue(allCases.contains(.ms160))
        XCTAssertTrue(allCases.contains(.ms80))
    }

    func testAllCasesOrder() {
        let allCases = NemotronChunkSize.allCases
        // Order in enum definition: ms1120, ms560, ms160, ms80
        XCTAssertEqual(allCases[0], .ms1120)
        XCTAssertEqual(allCases[1], .ms560)
        XCTAssertEqual(allCases[2], .ms160)
        XCTAssertEqual(allCases[3], .ms80)
    }

    // MARK: - P1: Sendable Conformance

    func testSendableConformance() {
        // Compile-time check: if this compiles, Sendable works
        let chunk: NemotronChunkSize = .ms560
        Task {
            let _ = chunk  // Can capture in async context
        }
    }
}
