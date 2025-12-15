import CoreML
import Foundation

/// Manages the state (cache) for the streaming encoder.
/// Dimensions verified via pure CoreML Python pipeline:
/// - cache_last_channel: (17, 1, 70, 512)
/// - cache_last_time: (17, 1, 512, 8)
/// - cache_last_channel_len: (1,)
public struct StreamingEncoderState {
    public var preCache: MLMultiArray
    public var cacheLastChannel: MLMultiArray
    public var cacheLastTime: MLMultiArray
    public var cacheLastChannelLen: MLMultiArray

    // Dimensions
    private static let numLayers = 17
    private static let hiddenDim = 512
    private static let cacheChannelSize = 70
    private static let cacheTimeSize = 8
    private static let preCacheSize = 9
    private static let melDim = 128

    public init() {
        // Initialize with zeros
        // Shape: [Layers, Batch, Cache, Dim] = [17, 1, 70, 512]
        self.cacheLastChannel = try! MLMultiArray(
            shape: [
                NSNumber(value: Self.numLayers),
                NSNumber(value: 1),
                NSNumber(value: Self.cacheChannelSize),
                NSNumber(value: Self.hiddenDim),
            ], dataType: .float32)

        // Shape: [Layers, Batch, Dim, Time] = [17, 1, 512, 8]
        self.cacheLastTime = try! MLMultiArray(
            shape: [
                NSNumber(value: Self.numLayers),
                NSNumber(value: 1),
                NSNumber(value: Self.hiddenDim),
                NSNumber(value: Self.cacheTimeSize),
            ], dataType: .float32)

        self.cacheLastChannelLen = try! MLMultiArray(
            shape: [NSNumber(value: 1)], dataType: .int32)

        // Shape: [Batch, PreCache, MelDim] = [1, 9, 128]
        self.preCache = try! MLMultiArray(
            shape: [
                NSNumber(value: 1),
                NSNumber(value: Self.preCacheSize),
                NSNumber(value: Self.melDim),
            ], dataType: .float32)

        reset()
    }

    public mutating func reset() {
        // Fill with zeros
        let channelCount = cacheLastChannel.count
        cacheLastChannel.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            ptr.baseAddress?.assign(repeating: 0, count: channelCount)
        }

        let timeCount = cacheLastTime.count
        cacheLastTime.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            ptr.baseAddress?.assign(repeating: 0, count: timeCount)
        }

        cacheLastChannelLen[0] = 0

        let preCacheCount = preCache.count
        preCache.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, _ in
            ptr.baseAddress?.assign(repeating: 0, count: preCacheCount)
        }
    }

    public func toFeatureProvider() -> MLFeatureProvider {
        return try! MLDictionaryFeatureProvider(dictionary: [
            "cache_last_channel": MLFeatureValue(multiArray: cacheLastChannel),
            "cache_last_time": MLFeatureValue(multiArray: cacheLastTime),
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLastChannelLen),
            "pre_cache": MLFeatureValue(multiArray: preCache),
        ])
    }

    public mutating func update(from output: MLFeatureProvider) {
        if let newChannel = output.featureValue(for: "new_cache_channel")?.multiArrayValue {
            self.cacheLastChannel = Self.copyMultiArray(newChannel)
        }
        if let newTime = output.featureValue(for: "new_cache_time")?.multiArrayValue {
            self.cacheLastTime = Self.copyMultiArray(newTime)
        }
        if let newLen = output.featureValue(for: "new_cache_len")?.multiArrayValue {
            self.cacheLastChannelLen = Self.copyMultiArray(newLen)
        }
        if let newPreCache = output.featureValue(for: "new_pre_cache")?.multiArrayValue {
            self.preCache = Self.copyMultiArray(newPreCache)
        }
    }

    public static func copyMultiArray(_ source: MLMultiArray) -> MLMultiArray {
        guard let newArray = try? MLMultiArray(shape: source.shape, dataType: source.dataType) else {
            print("Error: Failed to create new MLMultiArray for copy")
            return source  // Fallback to shallow copy if creation fails
        }

        let count = source.count

        if source.dataType == .float32 {
            source.withUnsafeBufferPointer(ofType: Float.self) { srcPtr in
                newArray.withUnsafeMutableBufferPointer(ofType: Float.self) { dstPtr, _ in
                    if let srcBase = srcPtr.baseAddress, let dstBase = dstPtr.baseAddress {
                        dstBase.assign(from: srcBase, count: count)
                    }
                }
            }
        } else if source.dataType == .int32 {
            source.withUnsafeBufferPointer(ofType: Int32.self) { srcPtr in
                newArray.withUnsafeMutableBufferPointer(ofType: Int32.self) { dstPtr, _ in
                    if let srcBase = srcPtr.baseAddress, let dstBase = dstPtr.baseAddress {
                        dstBase.assign(from: srcBase, count: count)
                    }
                }
            }
        } else {
            // Fallback (slow)
            for i in 0..<count {
                newArray[i] = source[i]
            }
        }
        return newArray
    }
}
