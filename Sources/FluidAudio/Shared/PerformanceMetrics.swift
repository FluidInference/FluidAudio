import Foundation

/// Performance metrics for ASR processing
public struct ASRPerformanceMetrics: Codable, Sendable {
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
