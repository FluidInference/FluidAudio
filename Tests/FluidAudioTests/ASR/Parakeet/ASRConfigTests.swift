import XCTest

@testable import FluidAudio

final class ASRConfigTests: XCTestCase {

    func testMelContextDefaultsPreserveConversationalChunking() {
        let versions: [AsrModelVersion?] = [.v2, .v3, .redux, .ultra, .phonon2, .tdtCtc110m, .tdtJa, nil]
        for config in [ASRConfig(), .default, ASRConfig(melChunkContext: nil)] {
            XCTAssertNil(config.melChunkContextOverride)
            for version in versions {
                XCTAssertTrue(
                    config.resolvedMelChunkContext(for: version), "Issue #954: default must retain mel context")
            }
        }
    }

    func testExplicitMelContextOverridesArePreservedForEveryModel() {
        let versions: [AsrModelVersion?] = [.v2, .v3, .redux, .ultra, .phonon2, .tdtCtc110m, .tdtJa, nil]
        for enabled in [true, false] {
            let config = ASRConfig(melChunkContext: enabled)
            XCTAssertEqual(config.melChunkContextOverride, enabled)
            for version in versions {
                XCTAssertEqual(config.resolvedMelChunkContext(for: version), enabled)
            }
        }
    }

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
