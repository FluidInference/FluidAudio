import CoreML
import Foundation

/// State for cache-aware streaming encoder (split architecture)
///
/// Holds the encoder cache tensors that persist across streaming chunks.
/// Based on NVIDIA's cache-aware streaming FastConformer architecture with split encoder:
/// - pre_encode_cache: Conv subsampling overlap cache [1, 9, 128]
/// - cache_last_channel: Attention context cache [17, 1, 70, 512]
/// - cache_last_time: Time convolution cache [17, 1, 512, 8]
/// - cache_last_channel_len: Current cache usage length
public final class StreamingEncoderState: @unchecked Sendable {
    /// Pre-encode cache for conv subsampling overlap - shape [1, 9, 128]
    /// (batch, pre_cache_size, mel_dim)
    public let preEncodeCache: MLMultiArray

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

        // Initialize pre_encode_cache: [1, 9, 128]
        self.preEncodeCache = try MLMultiArray(
            shape: [
                1,
                NSNumber(value: config.preCacheSize),
                NSNumber(value: config.melDim),
            ],
            dataType: .float32
        )

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
        let preEncPtr = preEncodeCache.dataPointer.bindMemory(
            to: Float.self, capacity: preEncodeCache.count)
        memset(preEncPtr, 0, preEncodeCache.count * MemoryLayout<Float>.stride)

        let channelPtr = cacheLastChannel.dataPointer.bindMemory(
            to: Float.self, capacity: cacheLastChannel.count)
        memset(channelPtr, 0, cacheLastChannel.count * MemoryLayout<Float>.stride)

        let timePtr = cacheLastTime.dataPointer.bindMemory(
            to: Float.self, capacity: cacheLastTime.count)
        memset(timePtr, 0, cacheLastTime.count * MemoryLayout<Float>.stride)

        cacheLastChannelLen = 0
    }

    /// Update pre-encode cache
    public func updatePreEncodeCache(newCache: MLMultiArray) {
        let srcPtr = newCache.dataPointer.bindMemory(
            to: Float.self, capacity: newCache.count)
        let dstPtr = preEncodeCache.dataPointer.bindMemory(
            to: Float.self, capacity: preEncodeCache.count)
        memcpy(dstPtr, srcPtr, min(preEncodeCache.count, newCache.count) * MemoryLayout<Float>.stride)
    }

    /// Update conformer cache from encoder output
    /// Note: The encoder output may have different strides than our state cache,
    /// so we must copy element-by-element respecting the strides of both arrays.
    public func updateConformerCache(
        newCacheChannel: MLMultiArray,
        newCacheTime: MLMultiArray,
        newCacheLen: Int
    ) {
        // Copy cache_last_channel: shape [17, 1, 70, 512]
        copyMLMultiArrayWithStrides(from: newCacheChannel, to: cacheLastChannel)

        // Copy cache_last_time: shape [17, 1, 512, 8]
        copyMLMultiArrayWithStrides(from: newCacheTime, to: cacheLastTime)

        cacheLastChannelLen = newCacheLen
    }

    /// Copy MLMultiArray data respecting strides of both source and destination
    private func copyMLMultiArrayWithStrides(from src: MLMultiArray, to dst: MLMultiArray) {
        let srcStrides = src.strides.map { $0.intValue }
        let dstStrides = dst.strides.map { $0.intValue }
        let shape = dst.shape.map { $0.intValue }

        // Handle Float32 -> Float32
        if src.dataType == .float32 && dst.dataType == .float32 {
            let srcPtr = src.dataPointer.bindMemory(to: Float.self, capacity: src.count)
            let dstPtr = dst.dataPointer.bindMemory(to: Float.self, capacity: dst.count)

            for i0 in 0..<shape[0] {
                for i1 in 0..<shape[1] {
                    for i2 in 0..<shape[2] {
                        for i3 in 0..<shape[3] {
                            let srcIdx = i0 * srcStrides[0] + i1 * srcStrides[1] + i2 * srcStrides[2] + i3 * srcStrides[3]
                            let dstIdx = i0 * dstStrides[0] + i1 * dstStrides[1] + i2 * dstStrides[2] + i3 * dstStrides[3]
                            dstPtr[dstIdx] = srcPtr[srcIdx]
                        }
                    }
                }
            }
        }
        // Handle Float16 -> Float32
        else if src.dataType == .float16 && dst.dataType == .float32 {
            let srcPtr = src.dataPointer.bindMemory(to: Float16.self, capacity: src.count)
            let dstPtr = dst.dataPointer.bindMemory(to: Float.self, capacity: dst.count)

            for i0 in 0..<shape[0] {
                for i1 in 0..<shape[1] {
                    for i2 in 0..<shape[2] {
                        for i3 in 0..<shape[3] {
                            let srcIdx = i0 * srcStrides[0] + i1 * srcStrides[1] + i2 * srcStrides[2] + i3 * srcStrides[3]
                            let dstIdx = i0 * dstStrides[0] + i1 * dstStrides[1] + i2 * dstStrides[2] + i3 * dstStrides[3]
                            dstPtr[dstIdx] = Float(srcPtr[srcIdx])
                        }
                    }
                }
            }
        } else {
            print("Error: Unsupported data type combination in copyMLMultiArrayWithStrides: src=\(src.dataType), dst=\(dst.dataType)")
        }
    }

    /// Update cache from encoder output (legacy compatibility)
    public func updateCache(
        newCacheChannel: MLMultiArray,
        newCacheTime: MLMultiArray,
        newCacheLen: Int
    ) {
        updateConformerCache(newCacheChannel: newCacheChannel, newCacheTime: newCacheTime, newCacheLen: newCacheLen)
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

    /// Pre-encode cache size (for conv subsampling overlap)
    public let preCacheSize: Int

    /// Mel spectrogram features
    public let melDim: Int

    /// Sample rate
    public let sampleRate: Int

    /// Default configuration for Parakeet EOU streaming (split encoder)
    public static let parakeetEOUStreaming = StreamingEncoderConfig(
        numLayers: 17,
        encoderHiddenSize: 512,
        cacheChannelSize: 70,
        cacheTimeSize: 8,
        preCacheSize: 9,
        melDim: 128,
        sampleRate: 16000
    )

    public init(
        numLayers: Int = 17,
        encoderHiddenSize: Int = 512,
        cacheChannelSize: Int = 70,
        cacheTimeSize: Int = 8,
        preCacheSize: Int = 9,
        melDim: Int = 128,
        sampleRate: Int = 16000
    ) {
        self.numLayers = numLayers
        self.encoderHiddenSize = encoderHiddenSize
        self.cacheChannelSize = cacheChannelSize
        self.cacheTimeSize = cacheTimeSize
        self.preCacheSize = preCacheSize
        self.melDim = melDim
        self.sampleRate = sampleRate
    }
}
