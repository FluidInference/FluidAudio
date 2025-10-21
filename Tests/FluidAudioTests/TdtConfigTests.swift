import Foundation
import XCTest

@testable import FluidAudio

final class TdtConfigTests: XCTestCase {

    // MARK: - Default Configuration Tests

    func testDefaultConfiguration() {
        let config = TdtConfig.default

        XCTAssertTrue(config.includeTokenDuration, "Default should include token duration")
        XCTAssertEqual(config.maxSymbolsPerStep, 10, "Default max symbols per step should be 10")
        XCTAssertEqual(config.durationBins, [0, 1, 2, 3, 4], "Default duration bins should be [0, 1, 2, 3, 4]")
        XCTAssertEqual(config.blankId, 8192, "Default blank ID for v3 model should be 8192")
    }

    func testDefaultConfigurationImmutability() {
        let config1 = TdtConfig.default
        let config2 = TdtConfig.default

        // Verify they have the same values (not necessarily same instance)
        XCTAssertEqual(config1.includeTokenDuration, config2.includeTokenDuration)
        XCTAssertEqual(config1.maxSymbolsPerStep, config2.maxSymbolsPerStep)
        XCTAssertEqual(config1.durationBins, config2.durationBins)
        XCTAssertEqual(config1.blankId, config2.blankId)
    }

    // MARK: - Custom Configuration Tests

    func testCustomConfiguration() {
        let customConfig = TdtConfig(
            includeTokenDuration: false,
            maxSymbolsPerStep: 5,
            durationBins: [0, 2, 4],
            blankId: 4096
        )

        XCTAssertFalse(customConfig.includeTokenDuration, "Custom config should not include token duration")
        XCTAssertEqual(customConfig.maxSymbolsPerStep, 5, "Custom max symbols per step should be 5")
        XCTAssertEqual(customConfig.durationBins, [0, 2, 4], "Custom duration bins should match")
        XCTAssertEqual(customConfig.blankId, 4096, "Custom blank ID should be 4096")
    }

    func testPartialCustomConfiguration() {
        let config = TdtConfig(maxSymbolsPerStep: 15)

        // Should use default for unspecified parameters
        XCTAssertTrue(config.includeTokenDuration, "Should use default includeTokenDuration")
        XCTAssertEqual(config.maxSymbolsPerStep, 15, "Should use custom maxSymbolsPerStep")
        XCTAssertEqual(config.durationBins, [0, 1, 2, 3, 4], "Should use default duration bins")
        XCTAssertEqual(config.blankId, 8192, "Should use default blank ID")
    }

    // MARK: - Edge Cases

    func testEmptyDurationBins() {
        let config = TdtConfig(durationBins: [])

        XCTAssertTrue(config.durationBins.isEmpty, "Should accept empty duration bins")
    }

    func testSingleDurationBin() {
        let config = TdtConfig(durationBins: [2])

        XCTAssertEqual(config.durationBins, [2], "Should accept single duration bin")
    }

    func testZeroMaxSymbols() {
        let config = TdtConfig(maxSymbolsPerStep: 0)

        XCTAssertEqual(config.maxSymbolsPerStep, 0, "Should accept zero max symbols")
    }

    func testNegativeBlankId() {
        let config = TdtConfig(blankId: -1)

        XCTAssertEqual(config.blankId, -1, "Should accept negative blank ID")
    }

    // MARK: - V2 vs V3 Model Compatibility

    func testV2ModelBlankId() {
        // Test configuration that would be suitable for v2 model
        let v2Config = TdtConfig(blankId: 1024)

        XCTAssertEqual(v2Config.blankId, 1024, "Should support v2 model blank ID")
    }

    func testV3ModelBlankId() {
        // Test default v3 model configuration
        let v3Config = TdtConfig.default

        XCTAssertEqual(v3Config.blankId, 8192, "Should use v3 model blank ID by default")
    }

    // MARK: - Configuration Equality

    func testConfigurationEquality() {
        let config1 = TdtConfig(
            includeTokenDuration: true,
            maxSymbolsPerStep: 8,
            durationBins: [0, 1, 2],
            blankId: 512
        )

        let config2 = TdtConfig(
            includeTokenDuration: true,
            maxSymbolsPerStep: 8,
            durationBins: [0, 1, 2],
            blankId: 512
        )

        // Since TdtConfig doesn't implement Equatable, we test field-by-field
        XCTAssertEqual(config1.includeTokenDuration, config2.includeTokenDuration)
        XCTAssertEqual(config1.maxSymbolsPerStep, config2.maxSymbolsPerStep)
        XCTAssertEqual(config1.durationBins, config2.durationBins)
        XCTAssertEqual(config1.blankId, config2.blankId)
    }

    func testConfigurationInequality() {
        let config1 = TdtConfig.default
        let config2 = TdtConfig(maxSymbolsPerStep: 20)

        XCTAssertNotEqual(config1.maxSymbolsPerStep, config2.maxSymbolsPerStep, "Configs should differ")
    }

    // MARK: - Performance Tests

    func testConfigurationCreationPerformance() {
        measure {
            for _ in 0..<1000 {
                _ = TdtConfig.default
            }
        }
    }

    func testCustomConfigurationCreationPerformance() {
        measure {
            for _ in 0..<1000 {
                _ = TdtConfig(
                    includeTokenDuration: true,
                    maxSymbolsPerStep: 10,
                    durationBins: [0, 1, 2, 3, 4],
                    blankId: 8192
                )
            }
        }
    }
}
