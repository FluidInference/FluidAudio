import Foundation
import XCTest

@testable import FluidAudio

final class PerformanceMetricsTests: XCTestCase {

    // MARK: - ASRPerformanceMetrics Summary

    func testASRPerformanceMetricsSummaryFormatting() {
        let metrics = ASRPerformanceMetrics(
            preprocessorTime: 0.123,
            encoderTime: 0.456,
            decoderTime: 0.789,
            totalProcessingTime: 1.368,
            rtfx: 10.5,
            peakMemoryMB: 256.3,
            gpuUtilization: 85.0
        )

        let summary = metrics.summary
        XCTAssertTrue(summary.contains("0.123"), "Summary should contain preprocessor time")
        XCTAssertTrue(summary.contains("0.456"), "Summary should contain encoder time")
        XCTAssertTrue(summary.contains("0.789"), "Summary should contain decoder time")
        XCTAssertTrue(summary.contains("1.368"), "Summary should contain total time")
        XCTAssertTrue(summary.contains("10.5"), "Summary should contain RTFx")
        XCTAssertTrue(summary.contains("256.3"), "Summary should contain peak memory")
        XCTAssertTrue(summary.contains("85.0%"), "Summary should contain GPU utilization")
    }

    func testASRPerformanceMetricsSummaryWithNilGPU() {
        let metrics = ASRPerformanceMetrics(
            preprocessorTime: 0.1,
            encoderTime: 0.2,
            decoderTime: 0.3,
            totalProcessingTime: 0.6,
            rtfx: 5.0,
            peakMemoryMB: 100.0,
            gpuUtilization: nil
        )

        let summary = metrics.summary
        XCTAssertTrue(summary.contains("N/A"), "Summary should show N/A for nil GPU utilization")
    }
}
