import Foundation
import XCTest

@testable import FluidAudio

final class BatchStyleStreamingConfigTests: XCTestCase {

    // MARK: - Default Config Tests

    func testDefaultConfigValues() {
        let config = BatchStyleStreamingConfig.default

        XCTAssertEqual(config.chunkSeconds, 14.0, "Default chunk should be 14 seconds")
        XCTAssertEqual(config.overlapSeconds, 2.0, "Default overlap should be 2 seconds")
        XCTAssertEqual(config.minChunkSeconds, 2.0, "Default min chunk should be 2 seconds")
    }

    func testDefaultConfigSampleCalculations() {
        let config = BatchStyleStreamingConfig.default

        // At 16kHz sample rate
        XCTAssertEqual(config.chunkSamples, 224_000, "14s * 16000 = 224000 samples")
        XCTAssertEqual(config.overlapSamples, 32_000, "2s * 16000 = 32000 samples")
        XCTAssertEqual(config.minChunkSamples, 32_000, "2s * 16000 = 32000 samples")
    }

    // MARK: - Low Latency Config Tests

    func testLowLatencyConfigValues() {
        let config = BatchStyleStreamingConfig.lowLatency

        XCTAssertEqual(config.chunkSeconds, 10.0, "Low latency chunk should be 10 seconds")
        XCTAssertEqual(config.overlapSeconds, 2.0, "Low latency overlap should be 2 seconds")
        XCTAssertEqual(config.minChunkSeconds, 2.0, "Low latency min chunk should be 2 seconds")
    }

    func testLowLatencyConfigSampleCalculations() {
        let config = BatchStyleStreamingConfig.lowLatency

        XCTAssertEqual(config.chunkSamples, 160_000, "10s * 16000 = 160000 samples")
        XCTAssertEqual(config.overlapSamples, 32_000, "2s * 16000 = 32000 samples")
        XCTAssertEqual(config.minChunkSamples, 32_000, "2s * 16000 = 32000 samples")
    }

    // MARK: - Custom Config Tests

    func testCustomConfig() {
        let config = BatchStyleStreamingConfig(
            chunkSeconds: 8.0,
            overlapSeconds: 1.5,
            minChunkSeconds: 1.0
        )

        XCTAssertEqual(config.chunkSeconds, 8.0)
        XCTAssertEqual(config.overlapSeconds, 1.5)
        XCTAssertEqual(config.minChunkSeconds, 1.0)

        XCTAssertEqual(config.chunkSamples, 128_000, "8s * 16000 = 128000 samples")
        XCTAssertEqual(config.overlapSamples, 24_000, "1.5s * 16000 = 24000 samples")
        XCTAssertEqual(config.minChunkSamples, 16_000, "1s * 16000 = 16000 samples")
    }

    func testConfigWithDefaultParameters() {
        // Test init with default parameter values
        let config = BatchStyleStreamingConfig()

        XCTAssertEqual(config.chunkSeconds, 14.0, "Default init should use 14s chunk")
        XCTAssertEqual(config.overlapSeconds, 2.0, "Default init should use 2s overlap")
        XCTAssertEqual(config.minChunkSeconds, 2.0, "Default init should use 2s min chunk")
    }

    // MARK: - Stride Calculation Tests

    func testStrideCalculation() {
        // Stride = chunkSamples - overlapSamples
        // This is how far we advance between chunks

        let config = BatchStyleStreamingConfig.default
        let stride = config.chunkSamples - config.overlapSamples

        XCTAssertEqual(stride, 192_000, "Stride should be 224000 - 32000 = 192000 samples (12s)")

        // 12 seconds stride
        let strideSeconds = Double(stride) / 16000.0
        XCTAssertEqual(strideSeconds, 12.0, accuracy: 0.001, "Stride should be 12 seconds")
    }

    func testLowLatencyStrideCalculation() {
        let config = BatchStyleStreamingConfig.lowLatency
        let stride = config.chunkSamples - config.overlapSamples

        XCTAssertEqual(stride, 128_000, "Low latency stride should be 160000 - 32000 = 128000 samples (8s)")

        let strideSeconds = Double(stride) / 16000.0
        XCTAssertEqual(strideSeconds, 8.0, accuracy: 0.001, "Low latency stride should be 8 seconds")
    }

    // MARK: - Boundary Validation Tests

    func testChunkSizeUnderModelLimit() {
        // The CoreML model has a 240,000 sample (15 second) limit
        let modelLimit = 240_000

        let defaultConfig = BatchStyleStreamingConfig.default
        XCTAssertLessThan(defaultConfig.chunkSamples, modelLimit, "Default chunk should be under 15s model limit")

        let lowLatencyConfig = BatchStyleStreamingConfig.lowLatency
        XCTAssertLessThan(lowLatencyConfig.chunkSamples, modelLimit, "Low latency chunk should be under 15s model limit")
    }

    func testOverlapSmallerThanChunk() {
        let config = BatchStyleStreamingConfig.default

        XCTAssertLessThan(
            config.overlapSamples, config.chunkSamples,
            "Overlap must be smaller than chunk size"
        )

        XCTAssertLessThan(
            config.overlapSeconds, config.chunkSeconds,
            "Overlap seconds must be smaller than chunk seconds"
        )
    }

    func testMinChunkSmallerThanChunk() {
        let config = BatchStyleStreamingConfig.default

        XCTAssertLessThanOrEqual(
            config.minChunkSamples, config.chunkSamples,
            "Min chunk must be <= chunk size"
        )
    }

    // MARK: - Edge Cases

    func testVerySmallConfig() {
        let config = BatchStyleStreamingConfig(
            chunkSeconds: 1.0,
            overlapSeconds: 0.5,
            minChunkSeconds: 0.5
        )

        XCTAssertEqual(config.chunkSamples, 16_000, "1s = 16000 samples")
        XCTAssertEqual(config.overlapSamples, 8_000, "0.5s = 8000 samples")
        XCTAssertEqual(config.minChunkSamples, 8_000, "0.5s = 8000 samples")
    }

    func testMaxReasonableConfig() {
        // Maximum reasonable config just under model limit
        let config = BatchStyleStreamingConfig(
            chunkSeconds: 14.9,  // Just under 15s
            overlapSeconds: 3.0,
            minChunkSeconds: 3.0
        )

        XCTAssertEqual(config.chunkSamples, 238_400, "14.9s * 16000 = 238400 samples")
        XCTAssertLessThan(config.chunkSamples, 240_000, "Should be under model limit")
    }
}
