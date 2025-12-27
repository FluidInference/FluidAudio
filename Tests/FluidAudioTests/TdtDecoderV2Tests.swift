@preconcurrency import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class TdtDecoderV2Tests: XCTestCase {

    func testAdaptConfigRemapsBlankIdToV2Value() {
        let baseConfig = makeBaseConfig(blankId: 8192)
        let decoder = TdtDecoderV2(config: baseConfig)
        let adaptedConfig = extractConfig(from: decoder)

        XCTAssertEqual(baseConfig.tdtConfig.blankId, 8192)
        XCTAssertEqual(adaptedConfig.sampleRate, baseConfig.sampleRate)

        // Blank ID should be remapped for v2 models
        XCTAssertEqual(adaptedConfig.tdtConfig.blankId, 1024)

        // All other configuration knobs should be preserved
        XCTAssertEqual(adaptedConfig.tdtConfig.includeTokenDuration, baseConfig.tdtConfig.includeTokenDuration)
        XCTAssertEqual(adaptedConfig.tdtConfig.maxSymbolsPerStep, baseConfig.tdtConfig.maxSymbolsPerStep)
        XCTAssertEqual(adaptedConfig.tdtConfig.durationBins, baseConfig.tdtConfig.durationBins)
        XCTAssertEqual(adaptedConfig.tdtConfig.boundarySearchFrames, baseConfig.tdtConfig.boundarySearchFrames)
        XCTAssertEqual(adaptedConfig.tdtConfig.maxTokensPerChunk, baseConfig.tdtConfig.maxTokensPerChunk)
        XCTAssertEqual(
            adaptedConfig.tdtConfig.consecutiveBlankLimit,
            baseConfig.tdtConfig.consecutiveBlankLimit
        )
    }

    func testAdaptConfigLeavesExistingV2ConfigUnchanged() {
        let baseConfig = makeBaseConfig(blankId: 1024)
        let decoder = TdtDecoderV2(config: baseConfig)
        let adaptedConfig = extractConfig(from: decoder)

        XCTAssertEqual(adaptedConfig.sampleRate, baseConfig.sampleRate)
        XCTAssertEqual(adaptedConfig.tdtConfig.blankId, 1024)
        XCTAssertEqual(adaptedConfig.tdtConfig.includeTokenDuration, baseConfig.tdtConfig.includeTokenDuration)
        XCTAssertEqual(adaptedConfig.tdtConfig.maxSymbolsPerStep, baseConfig.tdtConfig.maxSymbolsPerStep)
        XCTAssertEqual(adaptedConfig.tdtConfig.durationBins, baseConfig.tdtConfig.durationBins)
        XCTAssertEqual(adaptedConfig.tdtConfig.boundarySearchFrames, baseConfig.tdtConfig.boundarySearchFrames)
        XCTAssertEqual(adaptedConfig.tdtConfig.maxTokensPerChunk, baseConfig.tdtConfig.maxTokensPerChunk)
        XCTAssertEqual(
            adaptedConfig.tdtConfig.consecutiveBlankLimit,
            baseConfig.tdtConfig.consecutiveBlankLimit
        )
    }

    // MARK: - Helpers

    private func makeBaseConfig(blankId: Int) -> ASRConfig {
        let tdtConfig = TdtConfig(
            includeTokenDuration: false,
            maxSymbolsPerStep: 7,
            durationBins: [0, 2, 4, 6],
            blankId: blankId,
            boundarySearchFrames: 12,
            maxTokensPerChunk: 42,
            consecutiveBlankLimit: 3
        )
        return ASRConfig(sampleRate: 44_100, tdtConfig: tdtConfig)
    }

    private func extractConfig(
        from decoder: TdtDecoderV2,
        file: StaticString = #filePath,
        line: UInt = #line
    ) -> ASRConfig {
        let mirror = Mirror(reflecting: decoder)
        for child in mirror.children {
            if child.label == "config", let config = child.value as? ASRConfig {
                return config
            }
        }
        XCTFail("Unable to mirror config from TdtDecoderV2", file: file, line: line)
        return .default
    }
}
