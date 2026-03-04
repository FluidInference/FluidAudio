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

    // MARK: - AggregatedMetrics

    func testAggregatedMetricsSummaryFormatting() {
        let aggregated = AggregatedMetrics(
            averageRTFx: 8.5,
            averageProcessingTime: 1.234,
            maxMemoryMB: 512.0,
            sampleCount: 10
        )

        let summary = aggregated.summary
        XCTAssertTrue(summary.contains("10 samples"), "Summary should contain sample count")
        XCTAssertTrue(summary.contains("8.5"), "Summary should contain average RTFx")
        XCTAssertTrue(summary.contains("1.234"), "Summary should contain average processing time")
        XCTAssertTrue(summary.contains("512.0"), "Summary should contain max memory")
    }

    // MARK: - PerformanceMonitor

    func testAggregatedMetricsEmptyReturnsNil() async {
        let monitor = PerformanceMonitor()
        let result = await monitor.getAggregatedMetrics()
        XCTAssertNil(result, "Empty monitor should return nil for aggregated metrics")
    }

    func testResetClearsMetrics() async throws {
        let monitor = PerformanceMonitor()

        // Track a session to add metrics
        _ = try await monitor.trackSession(operation: "test", audioLengthSeconds: 1.0) {
            return 42
        }

        // Verify metrics exist
        let before = await monitor.getAggregatedMetrics()
        XCTAssertNotNil(before)

        // Reset and verify empty
        await monitor.reset()
        let after = await monitor.getAggregatedMetrics()
        XCTAssertNil(after, "After reset, aggregated metrics should be nil")
    }

    func testTrackSessionReturnsMetrics() async throws {
        let monitor = PerformanceMonitor()

        let (result, metrics) = try await monitor.trackSession(
            operation: "test",
            audioLengthSeconds: 2.0
        ) {
            return "hello"
        }

        XCTAssertEqual(result, "hello")
        XCTAssertGreaterThanOrEqual(metrics.totalProcessingTime, 0)
        XCTAssertGreaterThan(metrics.rtfx, 0)
    }

    func testAggregatedMetricsComputation() async throws {
        let monitor = PerformanceMonitor()

        for i in 0..<3 {
            _ = try await monitor.trackSession(
                operation: "test\(i)",
                audioLengthSeconds: Float(i + 1)
            ) {
                return i
            }
        }

        let aggregated = await monitor.getAggregatedMetrics()
        XCTAssertNotNil(aggregated)
        XCTAssertEqual(aggregated?.sampleCount, 3)
        XCTAssertGreaterThan(aggregated!.averageRTFx, 0)
        XCTAssertGreaterThan(aggregated!.averageProcessingTime, 0)
    }
}
