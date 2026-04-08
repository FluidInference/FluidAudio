import CoreML
import Foundation

/// Utility for managing encoder cache states in streaming ASR models.
/// Consolidates common cache initialization and update patterns used by both
/// EOU and Nemotron streaming managers.
public struct EncoderCacheManager {

    /// Configuration for encoder cache initialization
    public struct CacheConfig {
        let channelShape: [Int]
        let timeShape: [Int]
        let lenShape: [Int]

        public init(channelShape: [Int], timeShape: [Int], lenShape: [Int]) {
            self.channelShape = channelShape
            self.timeShape = timeShape
            self.lenShape = lenShape
        }
    }

    /// Create initial encoder cache arrays filled with zeros
    public static func createInitialCaches(
        config: CacheConfig
    ) throws -> (
        channel: MLMultiArray,
        time: MLMultiArray,
        len: MLMultiArray
    ) {
        let channel = try MLMultiArray(
            shape: config.channelShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        channel.reset(to: 0)

        let time = try MLMultiArray(
            shape: config.timeShape.map { NSNumber(value: $0) },
            dataType: .float32
        )
        time.reset(to: 0)

        let len = try MLMultiArray(
            shape: config.lenShape.map { NSNumber(value: $0) },
            dataType: .int32
        )
        len.reset(to: 0)

        return (channel: channel, time: time, len: len)
    }

    /// Extract updated cache arrays from encoder output
    /// - Parameters:
    ///   - output: Model output containing cache arrays
    ///   - channelKey: Feature name for channel cache (default: "cache_channel_out")
    ///   - timeKey: Feature name for time cache (default: "cache_time_out")
    ///   - lenKey: Feature name for length cache (default: "cache_len_out")
    /// - Returns: Tuple of updated cache arrays
    public static func extractCachesFromOutput(
        _ output: MLFeatureProvider,
        channelKey: String = "cache_channel_out",
        timeKey: String = "cache_time_out",
        lenKey: String = "cache_len_out"
    ) -> (channel: MLMultiArray?, time: MLMultiArray?, len: MLMultiArray?) {
        let newChannel = output.featureValue(for: channelKey)?.multiArrayValue
        let newTime = output.featureValue(for: timeKey)?.multiArrayValue
        let newLen = output.featureValue(for: lenKey)?.multiArrayValue

        return (channel: newChannel, time: newTime, len: newLen)
    }

    /// Create a zero-initialized array with specified shape and type
    public static func createZeroArray(
        shape: [Int],
        dataType: MLMultiArrayDataType = .float32
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: shape.map { NSNumber(value: $0) },
            dataType: dataType
        )
        array.reset(to: 0)
        return array
    }
}
