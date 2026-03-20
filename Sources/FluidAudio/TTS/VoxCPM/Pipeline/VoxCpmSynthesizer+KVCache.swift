@preconcurrency import CoreML
import Foundation

extension VoxCpmSynthesizer {

    /// Mutable KV cache state for both the base and residual LMs.
    struct CacheState {
        /// 48 tensors: k0,v0,k1,v1,...,k23,v23 each [1, numKvHeads, maxSeqLen, headDim].
        var baseCaches: [MLMultiArray]
        /// 16 tensors: k0,v0,k1,v1,...,k7,v7 each [1, numKvHeads, maxSeqLen, headDim].
        var residualCaches: [MLMultiArray]
    }

    /// Create a fresh zero-initialized cache state.
    static func createCacheState() throws -> CacheState {
        let shape: [NSNumber] = [
            1,
            NSNumber(value: VoxCpmConstants.numKvHeads),
            NSNumber(value: VoxCpmConstants.maxSeqLen),
            NSNumber(value: VoxCpmConstants.headDim),
        ]

        var baseCaches: [MLMultiArray] = []
        for _ in 0..<VoxCpmConstants.baseCacheCount {
            let arr = try MLMultiArray(shape: shape, dataType: .float32)
            let ptr = arr.dataPointer.bindMemory(
                to: Float.self,
                capacity: VoxCpmConstants.numKvHeads * VoxCpmConstants.maxSeqLen
                    * VoxCpmConstants.headDim)
            memset(ptr, 0, arr.count * MemoryLayout<Float>.size)
            baseCaches.append(arr)
        }

        var residualCaches: [MLMultiArray] = []
        for _ in 0..<VoxCpmConstants.residualCacheCount {
            let arr = try MLMultiArray(shape: shape, dataType: .float32)
            let ptr = arr.dataPointer.bindMemory(
                to: Float.self,
                capacity: VoxCpmConstants.numKvHeads * VoxCpmConstants.maxSeqLen
                    * VoxCpmConstants.headDim)
            memset(ptr, 0, arr.count * MemoryLayout<Float>.size)
            residualCaches.append(arr)
        }

        return CacheState(baseCaches: baseCaches, residualCaches: residualCaches)
    }

    /// Result from a base LM step.
    struct BaseLmResult {
        let lmHidden: MLMultiArray  // [1, 1024]
        let lmHiddenFsq: MLMultiArray  // [1, 1024]
        let stopLogit: MLMultiArray  // [1, 2]
    }

    /// Run one base LM step and update KV caches in-place.
    ///
    /// - Parameters:
    ///   - embed: [1, 1024] embedding input
    ///   - position: current KV cache position
    ///   - caches: in-out cache state (48 tensors updated)
    ///   - model: base_lm_step MLModel
    static func runBaseLmStep(
        embed: MLMultiArray,
        position: Int,
        caches: inout CacheState,
        model: MLModel
    ) throws -> BaseLmResult {
        let inputDict = try MLDictionaryFeatureProvider(
            dictionary: buildBaseLmInputDict(
                embed: embed, position: position, caches: caches))

        let pred = try model.prediction(from: inputDict)

        // Update caches from outputs
        for i in 0..<VoxCpmConstants.baseLmLayers {
            caches.baseCaches[i * 2] = pred.featureValue(for: BaseLmKeys.outK(i))!.multiArrayValue!
            caches.baseCaches[i * 2 + 1] = pred.featureValue(for: BaseLmKeys.outV(i))!
                .multiArrayValue!
        }

        return BaseLmResult(
            lmHidden: try ensureFloat32(
                pred.featureValue(for: BaseLmKeys.lmHidden)!.multiArrayValue!),
            lmHiddenFsq: try ensureFloat32(
                pred.featureValue(for: BaseLmKeys.lmHiddenFsq)!.multiArrayValue!),
            stopLogit: try ensureFloat32(
                pred.featureValue(for: BaseLmKeys.stopLogit)!.multiArrayValue!)
        )
    }

    /// Run one residual LM step and update KV caches in-place.
    ///
    /// - Parameters:
    ///   - embed: [1, 1024] embedding input
    ///   - position: current KV cache position
    ///   - caches: in-out cache state (16 residual tensors updated)
    ///   - model: residual_lm_step MLModel
    static func runResidualLmStep(
        embed: MLMultiArray,
        position: Int,
        caches: inout CacheState,
        model: MLModel
    ) throws -> MLMultiArray {
        let inputDict = try MLDictionaryFeatureProvider(
            dictionary: buildResidualLmInputDict(
                embed: embed, position: position, caches: caches))

        let pred = try model.prediction(from: inputDict)

        // Update caches from outputs
        for i in 0..<VoxCpmConstants.residualLmLayers {
            caches.residualCaches[i * 2] = pred.featureValue(for: ResidualLmKeys.outK(i))!
                .multiArrayValue!
            caches.residualCaches[i * 2 + 1] = pred.featureValue(for: ResidualLmKeys.outV(i))!
                .multiArrayValue!
        }

        return try ensureFloat32(
            pred.featureValue(for: ResidualLmKeys.resHidden)!.multiArrayValue!)
    }

    // MARK: - Input Dictionary Builders

    private static func buildBaseLmInputDict(
        embed: MLMultiArray,
        position: Int,
        caches: CacheState
    ) throws -> [String: MLFeatureValue] {
        var dict: [String: MLFeatureValue] = [
            BaseLmKeys.embed: MLFeatureValue(multiArray: embed),
            BaseLmKeys.position: MLFeatureValue(
                multiArray: try createPositionArray(position)),
        ]
        for i in 0..<VoxCpmConstants.baseLmLayers {
            dict[BaseLmKeys.k(i)] = MLFeatureValue(multiArray: caches.baseCaches[i * 2])
            dict[BaseLmKeys.v(i)] = MLFeatureValue(multiArray: caches.baseCaches[i * 2 + 1])
        }
        return dict
    }

    private static func buildResidualLmInputDict(
        embed: MLMultiArray,
        position: Int,
        caches: CacheState
    ) throws -> [String: MLFeatureValue] {
        var dict: [String: MLFeatureValue] = [
            ResidualLmKeys.embed: MLFeatureValue(multiArray: embed),
            ResidualLmKeys.position: MLFeatureValue(
                multiArray: try createPositionArray(position)),
        ]
        for i in 0..<VoxCpmConstants.residualLmLayers {
            dict[ResidualLmKeys.k(i)] = MLFeatureValue(multiArray: caches.residualCaches[i * 2])
            dict[ResidualLmKeys.v(i)] = MLFeatureValue(
                multiArray: caches.residualCaches[i * 2 + 1])
        }
        return dict
    }

    /// Create position index as [1] Int32 MLMultiArray.
    static func createPositionArray(_ position: Int) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [1], dataType: .int32)
        arr[0] = NSNumber(value: Int32(position))
        return arr
    }
}
