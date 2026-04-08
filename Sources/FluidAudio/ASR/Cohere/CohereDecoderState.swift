@preconcurrency import CoreML
import Foundation

/// Manages KV cache state for Cohere Transcribe decoder.
///
/// Following the Parakeet TDT pattern: cache is managed externally in Swift,
/// passed into the CoreML model as inputs, and extracted from outputs each step.
struct CohereDecoderState: Sendable {
    // KV cache for 8 decoder layers
    // Each cache: [1, 8 heads, 108 seq_len, 128 head_dim]
    var kCaches: [MLMultiArray]  // 8 layers
    var vCaches: [MLMultiArray]  // 8 layers

    /// Current position in the cache (0-indexed)
    /// - At step 0: past_kv_len = 0 (no previous tokens)
    /// - At step 1: past_kv_len = 1 (1 token in cache)
    var pastKvLen: Int

    /// Last decoded token (for potential streaming context)
    var lastToken: Int?

    /// Initialize empty decoder state for Cohere Transcribe.
    /// - Parameters:
    ///   - numLayers: Number of decoder layers (default: 8 for Cohere)
    ///   - numHeads: Number of attention heads (default: 8)
    ///   - maxSeqLen: Maximum sequence length (default: 108)
    ///   - headDim: Dimension of each attention head (default: 128)
    init(
        numLayers: Int = 8,
        numHeads: Int = 8,
        maxSeqLen: Int = 108,
        headDim: Int = 128
    ) throws {
        let shape = [
            1,
            NSNumber(value: numHeads),
            NSNumber(value: maxSeqLen),
            NSNumber(value: headDim),
        ]

        // Initialize K and V caches for each layer
        kCaches = try (0..<numLayers).map { _ in
            try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: .float32)
        }

        vCaches = try (0..<numLayers).map { _ in
            try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: .float32)
        }

        // Initialize all caches to zero
        for i in 0..<numLayers {
            kCaches[i].resetData(to: 0)
            vCaches[i].resetData(to: 0)
        }

        pastKvLen = 0
        lastToken = nil
    }

    /// Create decoder state (cannot throw).
    static func make(
        numLayers: Int = 8,
        numHeads: Int = 8,
        maxSeqLen: Int = 108,
        headDim: Int = 128
    ) -> CohereDecoderState {
        do {
            return try CohereDecoderState(
                numLayers: numLayers,
                numHeads: numHeads,
                maxSeqLen: maxSeqLen,
                headDim: headDim
            )
        } catch {
            fatalError("Failed to allocate Cohere decoder state: \(error)")
        }
    }

    /// Update cache arrays from decoder output.
    ///
    /// The decoder returns updated K/V caches as outputs. Extract them and
    /// update our state arrays. This follows the Parakeet pattern where
    /// CoreML returns new cache tensors each step.
    ///
    /// - Parameter decoderOutput: Feature provider from decoder model prediction
    mutating func updateFromOutput(_ decoderOutput: MLFeatureProvider) {
        for i in 0..<kCaches.count {
            if let kOut = decoderOutput.featureValue(for: "k_cache_\(i)_out")?.multiArrayValue {
                kCaches[i] = kOut
            }
            if let vOut = decoderOutput.featureValue(for: "v_cache_\(i)_out")?.multiArrayValue {
                vCaches[i] = vOut
            }
        }

        // Increment position counter
        pastKvLen += 1
    }

    /// Reset all state to initial values.
    mutating func reset() {
        for i in 0..<kCaches.count {
            kCaches[i].resetData(to: 0)
            vCaches[i].resetData(to: 0)
        }
        pastKvLen = 0
        lastToken = nil
    }

    /// Copy constructor for state forking.
    init(from other: CohereDecoderState) throws {
        kCaches = try other.kCaches.map { cache in
            let newCache = try MLMultiArray(shape: cache.shape, dataType: .float32)
            newCache.copyData(from: cache)
            return newCache
        }

        vCaches = try other.vCaches.map { cache in
            let newCache = try MLMultiArray(shape: cache.shape, dataType: .float32)
            newCache.copyData(from: cache)
            return newCache
        }

        pastKvLen = other.pastKvLen
        lastToken = other.lastToken
    }
}
