import Foundation
import XCTest

@testable import FluidAudio

final class NemotronChunkSizeTests: XCTestCase {

    // MARK: - P1: Raw Value

    func testRawValues() {
        XCTAssertEqual(NemotronChunkSize.ms2240.rawValue, 2240)
        XCTAssertEqual(NemotronChunkSize.ms1120.rawValue, 1120)
        XCTAssertEqual(NemotronChunkSize.ms560.rawValue, 560)
    }

    func testInitFromRawValue() {
        XCTAssertEqual(NemotronChunkSize(rawValue: 2240), .ms2240)
        XCTAssertEqual(NemotronChunkSize(rawValue: 1120), .ms1120)
        XCTAssertEqual(NemotronChunkSize(rawValue: 560), .ms560)
    }

    func testInvalidRawValueReturnsNil() {
        XCTAssertNil(NemotronChunkSize(rawValue: 100))
        XCTAssertNil(NemotronChunkSize(rawValue: 320))
        // v1 tiers were removed
        XCTAssertNil(NemotronChunkSize(rawValue: 160))
        XCTAssertNil(NemotronChunkSize(rawValue: 80))
        XCTAssertNil(NemotronChunkSize(rawValue: 0))
        XCTAssertNil(NemotronChunkSize(rawValue: 9999))
    }

    // MARK: - P1: Repo Mapping

    func testRepoMapping() {
        XCTAssertEqual(NemotronChunkSize.ms2240.repo, .nemotronStreaming2240)
        XCTAssertEqual(NemotronChunkSize.ms1120.repo, .nemotronStreaming1120)
        XCTAssertEqual(NemotronChunkSize.ms560.repo, .nemotronStreaming560)
    }

    // MARK: - P1: Subdirectory Generation

    func testSubdirectoryGeneration() {
        XCTAssertEqual(NemotronChunkSize.ms2240.subdirectory, "nemotron_coreml_2240ms")
        XCTAssertEqual(NemotronChunkSize.ms1120.subdirectory, "nemotron_coreml_1120ms")
        XCTAssertEqual(NemotronChunkSize.ms560.subdirectory, "nemotron_coreml_560ms")
    }

    // MARK: - P1: CaseIterable

    func testAllCasesContainsAllVariants() {
        let allCases = NemotronChunkSize.allCases
        XCTAssertEqual(allCases.count, 3)
        XCTAssertTrue(allCases.contains(.ms2240))
        XCTAssertTrue(allCases.contains(.ms1120))
        XCTAssertTrue(allCases.contains(.ms560))
    }

    func testAllCasesOrder() {
        let allCases = NemotronChunkSize.allCases
        // Order in enum definition: ms2240 (default), ms1120, ms560
        XCTAssertEqual(allCases[0], .ms2240)
        XCTAssertEqual(allCases[1], .ms1120)
        XCTAssertEqual(allCases[2], .ms560)
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
