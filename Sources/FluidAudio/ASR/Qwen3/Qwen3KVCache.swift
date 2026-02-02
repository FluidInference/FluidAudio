import CoreML
import Foundation

// MARK: - KV-Cache for Qwen3-ASR Decoder Stack

/// Manages stacked key-value cache tensors for the consolidated decoder stack.
///
/// The decoder stack model takes stacked KV caches with shape
/// [numLayers, numKVHeads, seqLen, headDim] and outputs updated caches
/// with shape [numLayers, numKVHeads, seqLen+1, headDim].
///
/// During autoregressive generation, each step replaces the cache wholesale
/// with the model's output (which already includes the appended new entry).
public struct Qwen3KVCache {
    public let numLayers: Int
    public let numKVHeads: Int
    public let headDim: Int

    /// Stacked K-cache: [numLayers, numKVHeads, currentSeqLen, headDim]
    public var kCaches: MLMultiArray

    /// Stacked V-cache: [numLayers, numKVHeads, currentSeqLen, headDim]
    public var vCaches: MLMultiArray

    /// Current sequence length (number of cached tokens, excluding the dummy entry).
    public private(set) var sequenceLength: Int = 0

    /// Indices in the sequence dimension that contain padding zeros (not real KV entries).
    /// These positions must be masked with -1e9 in the attention mask during decode.
    /// Padding is used to skip CoreML attention bad zones (see `padToMinimumLength`).
    public private(set) var paddingIndices: Range<Int>?

    /// Actual sequence dimension of the cache arrays (includes the dummy entry at index 0).
    /// Use this when constructing the attention mask.
    public var cacheSequenceDimension: Int {
        kCaches.shape[2].intValue
    }

    /// Create an empty KV-cache with minimal shape [numLayers, numKVHeads, 1, headDim].
    public init(config: Qwen3AsrConfig) {
        self.numLayers = config.numDecoderLayers
        self.numKVHeads = config.numKVHeads
        self.headDim = config.headDim
        self.kCaches = Self.emptyStackedCache(
            numLayers: config.numDecoderLayers,
            numKVHeads: config.numKVHeads,
            headDim: config.headDim
        )
        self.vCaches = Self.emptyStackedCache(
            numLayers: config.numDecoderLayers,
            numKVHeads: config.numKVHeads,
            headDim: config.headDim
        )
    }

    /// Replace both caches with decoder stack output.
    ///
    /// - Parameter tokenCount: Number of tokens processed in this step (1 for decode, N for prefill).
    public mutating func update(kCachesOut: MLMultiArray, vCachesOut: MLMultiArray, tokenCount: Int = 1) {
        kCaches = kCachesOut
        vCaches = vCachesOut
        sequenceLength += tokenCount
    }

    /// Trim the cache to keep only the first `newSequenceLength` real tokens (plus dummy).
    ///
    /// After padded prefill, the cache contains entries for padding positions that must
    /// be discarded before autoregressive decode begins.
    public mutating func trim(toSequenceLength newSequenceLength: Int) {
        let targetSeqDim = newSequenceLength + 1  // real tokens + 1 dummy
        let currentSeqDim = kCaches.shape[2].intValue

        guard targetSeqDim < currentSeqDim else { return }

        let shape: [NSNumber] = [
            NSNumber(value: numLayers),
            NSNumber(value: numKVHeads),
            NSNumber(value: targetSeqDim),
            NSNumber(value: headDim),
        ]

        // swiftlint:disable force_try
        let newK = try! MLMultiArray(shape: shape, dataType: .float32)
        let newV = try! MLMultiArray(shape: shape, dataType: .float32)
        // swiftlint:enable force_try

        let srcK = kCaches.dataPointer.assumingMemoryBound(to: Float.self)
        let dstK = newK.dataPointer.assumingMemoryBound(to: Float.self)
        let srcV = vCaches.dataPointer.assumingMemoryBound(to: Float.self)
        let dstV = newV.dataPointer.assumingMemoryBound(to: Float.self)

        let copyBytes = targetSeqDim * headDim * MemoryLayout<Float>.stride
        for layer in 0..<numLayers {
            for head in 0..<numKVHeads {
                let srcOffset = (layer * numKVHeads * currentSeqDim + head * currentSeqDim) * headDim
                let dstOffset = (layer * numKVHeads * targetSeqDim + head * targetSeqDim) * headDim
                memcpy(dstK + dstOffset, srcK + srcOffset, copyBytes)
                memcpy(dstV + dstOffset, srcV + srcOffset, copyBytes)
            }
        }

        kCaches = newK
        vCaches = newV
        sequenceLength = newSequenceLength
    }

    /// Strip the dummy entry at index 0 from both K and V caches.
    ///
    /// CoreML requires cache dim >= 1, so we start with a zeros dummy entry that gets
    /// masked with -1e9 during prefill. After prefill completes, stripping the dummy
    /// avoids numerical divergence during decode (CoreML's -1e9 masking becomes
    /// increasingly unstable as cache grows).
    public mutating func stripDummy() {
        let currentSeqDim = kCaches.shape[2].intValue
        guard currentSeqDim > 1 else { return }  // nothing to strip if only dummy

        let newSeqDim = currentSeqDim - 1
        let shape: [NSNumber] = [
            NSNumber(value: numLayers),
            NSNumber(value: numKVHeads),
            NSNumber(value: newSeqDim),
            NSNumber(value: headDim),
        ]

        // swiftlint:disable force_try
        let newK = try! MLMultiArray(shape: shape, dataType: .float32)
        let newV = try! MLMultiArray(shape: shape, dataType: .float32)
        // swiftlint:enable force_try

        let srcK = kCaches.dataPointer.assumingMemoryBound(to: Float.self)
        let dstK = newK.dataPointer.assumingMemoryBound(to: Float.self)
        let srcV = vCaches.dataPointer.assumingMemoryBound(to: Float.self)
        let dstV = newV.dataPointer.assumingMemoryBound(to: Float.self)

        let copyBytes = newSeqDim * headDim * MemoryLayout<Float>.stride
        for layer in 0..<numLayers {
            for head in 0..<numKVHeads {
                // Skip index 0 (dummy) — copy from index 1 onward
                let srcOffset = (layer * numKVHeads * currentSeqDim + head * currentSeqDim) * headDim + headDim
                let dstOffset = (layer * numKVHeads * newSeqDim + head * newSeqDim) * headDim
                memcpy(dstK + dstOffset, srcK + srcOffset, copyBytes)
                memcpy(dstV + dstOffset, srcV + srcOffset, copyBytes)
            }
        }

        kCaches = newK
        vCaches = newV
        // sequenceLength stays the same — dummy was never counted
    }

    /// Pad the cache sequence dimension to at least `minSeqDim` to avoid CoreML bad zones.
    ///
    /// CoreML's coremltools conversion produces catastrophic numerical errors in the
    /// attention computation when the KV cache sequence dimension falls in certain ranges
    /// (notably 112-126, just below HEAD_DIM=128). Padding past these zones and masking
    /// the padding positions with -1e9 in the attention mask prevents the issue.
    public mutating func padToMinimumLength(_ minSeqDim: Int) {
        let currentSeqDim = cacheSequenceDimension
        guard currentSeqDim < minSeqDim else { return }

        paddingIndices = currentSeqDim..<minSeqDim

        let shape: [NSNumber] = [
            NSNumber(value: numLayers),
            NSNumber(value: numKVHeads),
            NSNumber(value: minSeqDim),
            NSNumber(value: headDim),
        ]

        // swiftlint:disable force_try
        let newK = try! MLMultiArray(shape: shape, dataType: .float32)
        let newV = try! MLMultiArray(shape: shape, dataType: .float32)
        // swiftlint:enable force_try

        let totalCount = numLayers * numKVHeads * minSeqDim * headDim
        let dstK = newK.dataPointer.assumingMemoryBound(to: Float.self)
        let dstV = newV.dataPointer.assumingMemoryBound(to: Float.self)

        // Zero-fill entirely (padding positions will be all zeros)
        dstK.initialize(repeating: 0.0, count: totalCount)
        dstV.initialize(repeating: 0.0, count: totalCount)

        // Copy existing cache data into the new arrays
        let srcK = kCaches.dataPointer.assumingMemoryBound(to: Float.self)
        let srcV = vCaches.dataPointer.assumingMemoryBound(to: Float.self)
        let copyBytes = currentSeqDim * headDim * MemoryLayout<Float>.stride
        for layer in 0..<numLayers {
            for head in 0..<numKVHeads {
                let srcOffset = (layer * numKVHeads * currentSeqDim + head * currentSeqDim) * headDim
                let dstOffset = (layer * numKVHeads * minSeqDim + head * minSeqDim) * headDim
                memcpy(dstK + dstOffset, srcK + srcOffset, copyBytes)
                memcpy(dstV + dstOffset, srcV + srcOffset, copyBytes)
            }
        }

        kCaches = newK
        vCaches = newV
        // sequenceLength unchanged — padding doesn't count as real tokens
    }

    /// Reset the cache to empty (for starting a new generation).
    public mutating func reset() {
        sequenceLength = 0
        paddingIndices = nil
        kCaches = Self.emptyStackedCache(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim
        )
        vCaches = Self.emptyStackedCache(
            numLayers: numLayers,
            numKVHeads: numKVHeads,
            headDim: headDim
        )
    }

    // MARK: Private

    /// Create a zero-filled stacked cache: [numLayers, numKVHeads, 1, headDim]
    ///
    /// MLMultiArray doesn't support zero-length dimensions, so we use 1 as the
    /// minimum sequence length with all zeros.
    private static func emptyStackedCache(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int
    ) -> MLMultiArray {
        let shape: [NSNumber] = [
            NSNumber(value: numLayers),
            NSNumber(value: numKVHeads),
            1,
            NSNumber(value: headDim),
        ]
        do {
            let array = try MLMultiArray(shape: shape, dataType: .float32)
            let count = numLayers * numKVHeads * 1 * headDim
            let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            ptr.initialize(repeating: 0.0, count: count)
            return array
        } catch {
            fatalError("Failed to create empty stacked KV-cache: \(error)")
        }
    }
}
