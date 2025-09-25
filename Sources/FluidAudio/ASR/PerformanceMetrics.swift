import Foundation
import os

/// Performance metrics for ASR processing
public struct ASRPerformanceMetrics: Sendable {
    public let preprocessorTime: TimeInterval
    public let encoderTime: TimeInterval
    public let decoderTime: TimeInterval
    public let totalProcessingTime: TimeInterval
    public let rtfx: Float  // Real-time factor
    public let peakMemoryMB: Float
    public let gpuUtilization: Float?

    public var summary: String {
        """
        Performance Metrics:
        - Preprocessor: \(String(format: "%.3f", preprocessorTime))s
        - Encoder: \(String(format: "%.3f", encoderTime))s
        - Decoder: \(String(format: "%.3f", decoderTime))s
        - Total: \(String(format: "%.3f", totalProcessingTime))s
        - RTFx: \(String(format: "%.1f", rtfx))x real-time
        - Peak Memory: \(String(format: "%.1f", peakMemoryMB)) MB
        - GPU Utilization: \(gpuUtilization.map { String(format: "%.1f%%", $0) } ?? "N/A")
        """
    }
}

/// Performance monitor for tracking ASR metrics
@available(macOS 13.0, iOS 16.0, *)
public actor PerformanceMonitor {

    public init() {}
    private let logger = AppLogger(category: "Performance")
    private var metrics: [ASRPerformanceMetrics] = []
    private let signpostLogger = OSSignposter(subsystem: AppLogger.defaultSubsystem, category: "Performance")

    /// Track performance for a processing session
    public func trackSession<T>(
        operation: String,
        audioLengthSeconds: Float,
        block: () async throws -> T
    ) async throws -> (result: T, metrics: ASRPerformanceMetrics) {
        let sessionID = signpostLogger.makeSignpostID()
        let state = signpostLogger.beginInterval("ASR.Operation", id: sessionID)

        let startTime = Date()
        let startMemory = getCurrentMemoryUsage()

        // Track individual components
        let preprocessorTime: TimeInterval = 0
        let encoderTime: TimeInterval = 0
        let decoderTime: TimeInterval = 0

        // Execute the operation
        let result = try await block()

        let totalTime = Date().timeIntervalSince(startTime)
        let peakMemory = max(startMemory, getCurrentMemoryUsage())
        let rtfx = audioLengthSeconds / Float(totalTime)

        signpostLogger.endInterval("ASR.Operation", state)

        let metrics = ASRPerformanceMetrics(
            preprocessorTime: preprocessorTime,
            encoderTime: encoderTime,
            decoderTime: decoderTime,
            totalProcessingTime: totalTime,
            rtfx: rtfx,
            peakMemoryMB: peakMemory,
            gpuUtilization: nil  // Would require Metal performance counters
        )

        self.metrics.append(metrics)
        logger.info("\(operation) completed: \(metrics.summary)")

        return (result, metrics)
    }

    /// Track a specific component's execution time
    public func trackComponent<T>(
        _ component: String,
        block: () async throws -> T
    ) async throws -> (result: T, time: TimeInterval) {
        let componentID = signpostLogger.makeSignpostID()
        let state = signpostLogger.beginInterval("ASR.Component", id: componentID)

        let startTime = Date()
        let result = try await block()
        let time = Date().timeIntervalSince(startTime)

        signpostLogger.endInterval("ASR.Component", state)

        return (result, time)
    }

    /// Get aggregated metrics
    public func getAggregatedMetrics() -> AggregatedMetrics? {
        guard !metrics.isEmpty else { return nil }

        let avgRTFx = metrics.map { $0.rtfx }.reduce(0, +) / Float(metrics.count)
        let avgProcessingTime = metrics.map { $0.totalProcessingTime }.reduce(0, +) / Double(metrics.count)
        let maxMemory = metrics.map { $0.peakMemoryMB }.max() ?? 0

        return AggregatedMetrics(
            averageRTFx: avgRTFx,
            averageProcessingTime: avgProcessingTime,
            maxMemoryMB: maxMemory,
            sampleCount: metrics.count
        )
    }

    /// Clear all stored metrics
    public func reset() {
        metrics.removeAll()
    }

    /// Get current memory usage in MB
    private func getCurrentMemoryUsage() -> Float {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4

        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count)
            }
        }

        if result == KERN_SUCCESS {
            return Float(info.resident_size) / 1024.0 / 1024.0
        }

        return 0
    }
}

/// Aggregated performance metrics
public struct AggregatedMetrics: Sendable {
    public let averageRTFx: Float
    public let averageProcessingTime: TimeInterval
    public let maxMemoryMB: Float
    public let sampleCount: Int

    public var summary: String {
        """
        Aggregated Metrics (\(sampleCount) samples):
        - Average RTFx: \(String(format: "%.1f", averageRTFx))x real-time
        - Average Processing Time: \(String(format: "%.3f", averageProcessingTime))s
        - Max Memory Usage: \(String(format: "%.1f", maxMemoryMB)) MB
        """
    }
}

/// Extension to AsrManager for performance tracking
@available(macOS 13.0, iOS 16.0, *)
extension AsrManager {
    /// Transcribe with performance metrics
    public func transcribeWithMetrics(
        _ audioSamples: [Float],
        monitor: PerformanceMonitor? = nil
    ) async throws -> (result: ASRResult, metrics: ASRPerformanceMetrics?) {
        let audioLengthSeconds = Float(audioSamples.count) / Float(config.sampleRate)

        if let monitor = monitor {
            return try await monitor.trackSession(
                operation: "Transcription",
                audioLengthSeconds: audioLengthSeconds
            ) {
                try await self.transcribe(audioSamples)
            }
        } else {
            let result = try await transcribe(audioSamples)
            return (result, nil)
        }
    }
}

// Signpost extension removed - use built-in methods
