@preconcurrency import CoreML
import Foundation

/// Holds one path's KV cache state for the 12-layer decoder_step model
/// (rank-4 split-K/V production layout).
///
/// Each layer has:
///   - `cache_k{i}` : `MLMultiArray` shaped `[1, 512, numHeads, headDim]` fp16
///   - `cache_v{i}` : `MLMultiArray` shaped `[1, 512, numHeads, headDim]` fp16
///   - `position{i}`: `MLMultiArray` shaped `[1]` fp16 (scalar index into the cache)
///
/// After each `decoder_step` forward pass the model returns new K, V and
/// position buffers under output names that do not match the input names
/// (scatter rewrite). Names are hard-coded in
/// `mobius/.../generate_coreml.py` (`DECODER_CACHE_K_OUT_KEYS`,
/// `DECODER_CACHE_V_OUT_KEYS`, `DECODER_POSITION_KEYS`). This Swift port
/// mirrors that list and should be regenerated if the Python compile pipeline
/// changes.
public final class MagpieKvCache {

    /// Per-layer K cache output names (12 layers).
    public static let cacheKOutputKeys: [String] = [
        "new_k_1", "new_k_3", "new_k_5", "new_k_7",
        "new_k_9", "new_k_11", "new_k_13", "new_k_15",
        "new_k_17", "new_k_19", "new_k_21", "new_k",
    ]

    /// Per-layer V cache output names (12 layers).
    public static let cacheVOutputKeys: [String] = [
        "new_v_1", "new_v_3", "new_v_5", "new_v_7",
        "new_v_9", "new_v_11", "new_v_13", "new_v_15",
        "new_v_17", "new_v_19", "new_v_21", "new_v",
    ]

    /// Per-layer scalar position output names (12 layers).
    public static let positionOutputKeys: [String] = [
        "var_169", "var_339", "var_509", "var_679",
        "var_849", "var_1019", "var_1189", "var_1359",
        "var_1529", "var_1699", "var_1869", "var_2039",
    ]

    /// Per-layer combined K/V output keys for `decoder_prefill.mlmodelc`.
    /// Each output is shaped `[2, 1, 512, 12, 64]` fp16 where index 0 = K and
    /// index 1 = V (axis-0 stacked).
    public static let prefillCacheOutputKeys: [String] = [
        "var_208", "var_374", "var_540", "var_706",
        "var_872", "var_1038", "var_1204", "var_1370",
        "var_1536", "var_1702", "var_1868", "var_1958",
    ]

    public static let decoderHiddenKey = "input"

    public private(set) var cachesK: [MLMultiArray]
    public private(set) var cachesV: [MLMultiArray]
    public private(set) var positions: [MLMultiArray]

    /// Back-buffer set for double-buffered AR loop. Used as `outputBackings` so
    /// CoreML writes new K/V/pos straight into our pre-allocated arrays instead
    /// of allocating ~18.9 MB of fresh fp16 buffers per step. After each
    /// `decoder_step` call the synthesizer calls `swapBackings()` to promote
    /// the back set to the new front (used as the next step's inputs).
    private var cachesKBack: [MLMultiArray]
    private var cachesVBack: [MLMultiArray]
    private var positionsBack: [MLMultiArray]

    public let numLayers: Int
    public let maxCacheLength: Int
    public let numHeads: Int
    public let headDim: Int

    public init(numLayers: Int, maxCacheLength: Int, numHeads: Int, headDim: Int) throws {
        self.numLayers = numLayers
        self.maxCacheLength = maxCacheLength
        self.numHeads = numHeads
        self.headDim = headDim
        let cacheShape: [NSNumber] = [
            1, NSNumber(value: maxCacheLength),
            NSNumber(value: numHeads),
            NSNumber(value: headDim),
        ]
        func makeCacheArr() throws -> MLMultiArray {
            let arr = try MLMultiArray(shape: cacheShape, dataType: .float16)
            arr.zeroFillFloat16()
            return arr
        }
        func makePosArr() throws -> MLMultiArray {
            let arr = try MLMultiArray(shape: [1], dataType: .float16)
            arr.zeroFillFloat16()
            return arr
        }
        self.cachesK = try (0..<numLayers).map { _ in try makeCacheArr() }
        self.cachesV = try (0..<numLayers).map { _ in try makeCacheArr() }
        self.positions = try (0..<numLayers).map { _ in try makePosArr() }
        self.cachesKBack = try (0..<numLayers).map { _ in try makeCacheArr() }
        self.cachesVBack = try (0..<numLayers).map { _ in try makeCacheArr() }
        self.positionsBack = try (0..<numLayers).map { _ in try makePosArr() }
    }

    /// Populate `inputs` with `cache_k{i}` + `cache_v{i}` + `position{i}` keys.
    public func addInputs(to inputs: inout [String: MLMultiArray]) {
        for i in 0..<numLayers {
            inputs["cache_k\(i)"] = cachesK[i]
            inputs["cache_v\(i)"] = cachesV[i]
            inputs["position\(i)"] = positions[i]
        }
    }

    /// Populate `outputBackings` with the back-buffer arrays under each output
    /// key. CoreML will write directly into these arrays instead of allocating.
    public func addOutputBackings(to backings: inout [String: Any]) {
        for i in 0..<numLayers {
            backings[Self.cacheKOutputKeys[i]] = cachesKBack[i]
            backings[Self.cacheVOutputKeys[i]] = cachesVBack[i]
            backings[Self.positionOutputKeys[i]] = positionsBack[i]
        }
    }

    /// Promote the back-buffer set to the new front (which now holds the just-
    /// written K/V/pos for layer i). The old front becomes the new back and
    /// will be overwritten on the next prediction call. Cheap pointer-swap;
    /// no data copy.
    public func swapBackings() {
        swap(&cachesK, &cachesKBack)
        swap(&cachesV, &cachesVBack)
        swap(&positions, &positionsBack)
    }

    /// Slow path: pull new K/V/pos out of an output `MLFeatureProvider` and
    /// replace front pointers. Used when `outputBackings` is unavailable
    /// (e.g. if a future macOS revision rejects our buffer layout).
    public func absorbOutputs(_ output: MLFeatureProvider) throws {
        for i in 0..<numLayers {
            guard let newK = output.featureValue(for: Self.cacheKOutputKeys[i])?.multiArrayValue
            else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_step",
                    underlying: "missing K cache output key \(Self.cacheKOutputKeys[i])")
            }
            guard let newV = output.featureValue(for: Self.cacheVOutputKeys[i])?.multiArrayValue
            else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_step",
                    underlying: "missing V cache output key \(Self.cacheVOutputKeys[i])")
            }
            guard let newPos = output.featureValue(for: Self.positionOutputKeys[i])?.multiArrayValue
            else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_step",
                    underlying: "missing position output key \(Self.positionOutputKeys[i])")
            }
            cachesK[i] = newK
            cachesV[i] = newV
            positions[i] = newPos
        }
    }

    /// Current decoder position as read from layer 0's position tensor.
    public var position: Int {
        guard numLayers > 0 else { return 0 }
        return Int(positions[0][0].floatValue)
    }

    /// Seed cache state from `decoder_prefill.mlmodelc` outputs.
    ///
    /// Each prefill output is a `[2, 1, 512, numHeads, headDim]` fp16 tensor
    /// where slice 0 along axis 0 is K and slice 1 is V. After this call,
    /// `cachesK[i] / cachesV[i]` hold those slices and `positions[i]` is set to
    /// `prefillLength` (= 110 for the 110-token speaker context).
    public func seedFromPrefillOutputs(
        _ output: MLFeatureProvider, prefillLength: Int
    ) throws {
        let perLayerCount = maxCacheLength * numHeads * headDim
        let bytesPerSlice = perLayerCount * MemoryLayout<UInt16>.size

        for i in 0..<numLayers {
            let key = Self.prefillCacheOutputKeys[i]
            guard let stacked = output.featureValue(for: key)?.multiArrayValue else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_prefill",
                    underlying: "missing prefill output key \(key)")
            }
            guard stacked.dataType == .float16 else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_prefill",
                    underlying: "prefill output \(key) dtype is \(stacked.dataType.rawValue), expected fp16")
            }
            // Validate shape: [2, 1, 512, numHeads, headDim].
            let s = stacked.shape.map { $0.intValue }
            guard s.count == 5,
                s[0] == 2, s[1] == 1, s[2] == maxCacheLength,
                s[3] == numHeads, s[4] == headDim
            else {
                throw MagpieError.inferenceFailed(
                    stage: "decoder_prefill",
                    underlying: "prefill output \(key) has unexpected shape \(s)")
            }

            let basePtr = stacked.dataPointer.assumingMemoryBound(to: UInt8.self)
            // K slice = bytes [0, bytesPerSlice). V slice = bytes [bytesPerSlice, 2*bytesPerSlice).
            memcpy(cachesK[i].dataPointer, basePtr, bytesPerSlice)
            memcpy(cachesV[i].dataPointer, basePtr.advanced(by: bytesPerSlice), bytesPerSlice)

            // Position: scalar fp16 = prefillLength.
            let pos = try MLMultiArray(shape: [1], dataType: .float16)
            pos.zeroFillFloat16()
            pos[0] = NSNumber(value: Float(prefillLength))
            positions[i] = pos
        }
    }
}

// MARK: - Helpers

extension MLMultiArray {
    /// Zero-fill an fp16 `MLMultiArray` fast (uses `memset`).
    internal func zeroFillFloat16() {
        guard dataType == .float16 else {
            for i in 0..<count { self[i] = NSNumber(value: 0.0) }
            return
        }
        // fp16 zero is the same byte pattern as int16 zero (0x0000).
        let bytes = count * 2
        memset(dataPointer, 0, bytes)
    }
}
