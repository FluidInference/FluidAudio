import CoreML
import Foundation

/// Manages the state (cache) for the streaming encoder.
/// Dimensions verified via pure CoreML Python pipeline:
/// - cache_last_channel: (17, 1, 70, 512)
/// - cache_last_time: (17, 1, 512, 8)
/// - cache_last_channel_len: (1,)
public struct StreamingEncoderState {
    public var cacheLastChannel: MLMultiArray
    public var cacheLastTime: MLMultiArray
    public var cacheLastChannelLen: MLMultiArray

    // Dimensions
    private static let numLayers = 17
    private static let hiddenDim = 512
    private static let cacheChannelSize = 70
    private static let cacheTimeSize = 8

    public init() {
        // Initialize with zeros
        self.cacheLastChannel = try! MLMultiArray(
            shape: [
                NSNumber(value: Self.numLayers),
                NSNumber(value: 1),
                NSNumber(value: Self.cacheChannelSize),
                NSNumber(value: Self.hiddenDim),
            ], dataType: .float32)
        
        self.cacheLastTime = try! MLMultiArray(
            shape: [
                NSNumber(value: Self.numLayers),
                NSNumber(value: 1),
                NSNumber(value: Self.hiddenDim),
                NSNumber(value: Self.cacheTimeSize),
            ], dataType: .float32)
        
        self.cacheLastChannelLen = try! MLMultiArray(
            shape: [NSNumber(value: 1)], dataType: .int32)
            
        self.preCache = try! MLMultiArray(
            shape: [
                NSNumber(value: 1),
                NSNumber(value: Self.preCacheSize),
                NSNumber(value: Self.melDim)
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
            "cache_last_channel_len": MLFeatureValue(multiArray: cacheLastChannelLen)
        ])
    }
    
    public mutating func update(from output: MLFeatureProvider) {
        if let newChannel = output.featureValue(for: "new_cache_channel")?.multiArrayValue {
            self.cacheLastChannel = newChannel
        }
        if let newTime = output.featureValue(for: "new_cache_time")?.multiArrayValue {
            self.cacheLastTime = newTime
        }
        if let newLen = output.featureValue(for: "new_cache_len")?.multiArrayValue {
            self.cacheLastChannelLen = newLen
        }
    }
    
    // MARK: - Pre-Encode Cache
    public var preCache: MLMultiArray
    private static let preCacheSize = 9
    private static let melDim = 128
    
    public mutating func updatePreCache(from output: MLFeatureProvider) {
        if let newPreCache = output.featureValue(for: "new_pre_cache")?.multiArrayValue {
            self.preCache = newPreCache
        }
    }
}
