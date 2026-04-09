import XCTest

@testable import FluidAudio

final class ASRConfigTests: XCTestCase {

    func testDefaultParallelChunkConcurrency() {
        XCTAssertEqual(ASRConfig.default.parallelChunkConcurrency, 4)
    }

    func testParallelChunkConcurrencyClampsToAtLeastOne() {
        XCTAssertEqual(ASRConfig(parallelChunkConcurrency: 0).parallelChunkConcurrency, 1)
        XCTAssertEqual(ASRConfig(parallelChunkConcurrency: -3).parallelChunkConcurrency, 1)
    }

    func testParallelChunkConcurrencyPreservesExplicitValue() {
        XCTAssertEqual(ASRConfig(parallelChunkConcurrency: 6).parallelChunkConcurrency, 6)
    }
}
