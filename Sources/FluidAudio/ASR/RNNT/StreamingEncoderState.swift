import CoreML
import Foundation

/// State for cache-aware streaming encoder
///
/// Holds the encoder cache tensors that persist across streaming chunks.
/// Based on NVIDIA's cache-aware streaming FastConformer architecture:
/// - cache_last_channel: Attention context cache [17, 1, 70, 512]
/// - cache_last_time: Time convolution cache [17, 1, 512, 8]
/// - cache_last_channel_len: Current cache usage length
public final class StreamingEncoderState: @unchecked Sendable {
    /// Attention context cache - shape [17, 1, 70, 512]
    /// (num_layers, batch, cache_size, hidden_dim)
    public let cacheLastChannel: MLMultiArray

    /// Time convolution cache - shape [17, 1, 512, 8]
    /// (num_layers, batch, hidden_dim, time_cache)
    public let cacheLastTime: MLMultiArray

    /// Current cache usage length
    public var cacheLastChannelLen: Int

    /// Configuration for streaming
    public let config: StreamingEncoderConfig

    public init(config: StreamingEncoderConfig = .parakeetEOUStreaming) throws {
        self.config = config

        // Initialize cache_last_channel: [17, 1, 70, 512]
        self.cacheLastChannel = try MLMultiArray(
            shape: [
                NSNumber(value: config.numLayers),
                1,
                NSNumber(value: config.cacheChannelSize),
                NSNumber(value: config.encoderHiddenSize),
            ],
            dataType: .float32
        )

        // Initialize cache_last_time: [17, 1, 512, 8]
        self.cacheLastTime = try MLMultiArray(
            shape: [
                NSNumber(value: config.numLayers),
                1,
                NSNumber(value: config.encoderHiddenSize),
                NSNumber(value: config.cacheTimeSize),
            ],
            dataType: .float32
        )

        self.cacheLastChannelLen = 0

        // Zero-initialize caches
        reset()
    }

    /// Reset cache state to zeros
    public func reset() {
        let channelPtr = cacheLastChannel.dataPointer.bindMemory(
            to: Float.self, capacity: cacheLastChannel.count)
        memset(channelPtr, 0, cacheLastChannel.count * MemoryLayout<Float>.stride)

        let timePtr = cacheLastTime.dataPointer.bindMemory(
            to: Float.self, capacity: cacheLastTime.count)
        memset(timePtr, 0, cacheLastTime.count * MemoryLayout<Float>.stride)

        cacheLastChannelLen = 0
    }

    /// Update cache from encoder output
    public func updateCache(
        newCacheChannel: MLMultiArray,
        newCacheTime: MLMultiArray,
        newCacheLen: Int
    ) {
        // Copy new cache values
        let srcChannelPtr = newCacheChannel.dataPointer.bindMemory(
            to: Float.self, capacity: newCacheChannel.count)
        let dstChannelPtr = cacheLastChannel.dataPointer.bindMemory(
            to: Float.self, capacity: cacheLastChannel.count)
        memcpy(
            dstChannelPtr, srcChannelPtr,
            min(cacheLastChannel.count, newCacheChannel.count) * MemoryLayout<Float>.stride)

        let srcTimePtr = newCacheTime.dataPointer.bindMemory(
            to: Float.self, capacity: newCacheTime.count)
        let dstTimePtr = cacheLastTime.dataPointer.bindMemory(
            to: Float.self, capacity: cacheLastTime.count)
        memcpy(dstTimePtr, srcTimePtr, min(cacheLastTime.count, newCacheTime.count) * MemoryLayout<Float>.stride)

        cacheLastChannelLen = newCacheLen
    }

    /// Create MLMultiArray for cache_last_channel_len input
    public func createCacheLenArray() throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1], dataType: .int32)
        arr[0] = NSNumber(value: cacheLastChannelLen)
        return arr
    }
}

/// Configuration for streaming encoder
public struct StreamingEncoderConfig: Sendable {
    /// Number of encoder layers (FastConformer)
    public let numLayers: Int

    /// Encoder hidden dimension
    public let encoderHiddenSize: Int

    /// Attention cache size (last_channel_cache_size)
    public let cacheChannelSize: Int

    /// Time convolution cache size
    public let cacheTimeSize: Int

    /// Mel spectrogram features
    public let melDim: Int

    /// Sample rate
    public let sampleRate: Int

    /// Default configuration for Parakeet EOU streaming
    public static let parakeetEOUStreaming = StreamingEncoderConfig(
        numLayers: 17,
        encoderHiddenSize: 512,
        cacheChannelSize: 70,
        cacheTimeSize: 8,
        melDim: 128,
        sampleRate: 16000
    )

    public init(
        numLayers: Int = 17,
        encoderHiddenSize: Int = 512,
        cacheChannelSize: Int = 70,
        cacheTimeSize: Int = 8,
        melDim: Int = 128,
        sampleRate: Int = 16000
    ) {
        self.numLayers = numLayers
        self.encoderHiddenSize = encoderHiddenSize
        self.cacheChannelSize = cacheChannelSize
        self.cacheTimeSize = cacheTimeSize
        self.melDim = melDim
        self.sampleRate = sampleRate
    }
}
