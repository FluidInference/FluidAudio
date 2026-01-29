@preconcurrency import CoreML
import FluidAudio
import Foundation

extension PocketTtsSynthesizer {

    /// Mutable KV cache state passed through conditioning and generation steps.
    struct KVCacheState {
        /// 6 KV cache arrays, each [2, 1, 200, 16, 64].
        var caches: [MLMultiArray]
        /// 6 position counters, each [1].
        var positions: [MLMultiArray]
    }

    /// Create an empty KV cache state (all zeros, positions at 0).
    static func emptyKVCacheState() throws -> KVCacheState {
        let layers = PocketTtsConstants.kvCacheLayers
        let shape: [NSNumber] = [
            2, 1, NSNumber(value: PocketTtsConstants.kvCacheMaxLen), 16, 64,
        ]

        var caches: [MLMultiArray] = []
        var positions: [MLMultiArray] = []
        caches.reserveCapacity(layers)
        positions.reserveCapacity(layers)

        for _ in 0..<layers {
            let cache = try MLMultiArray(shape: shape, dataType: .float32)
            let cachePtr = cache.dataPointer.bindMemory(
                to: Float.self, capacity: cache.count)
            cachePtr.initialize(repeating: 0, count: cache.count)
            caches.append(cache)

            let pos = try MLMultiArray(shape: [1], dataType: .float32)
            pos[0] = NSNumber(value: Float(0))
            positions.append(pos)
        }

        return KVCacheState(caches: caches, positions: positions)
    }

    /// Clone a KV cache state for independent use.
    static func cloneKVCacheState(_ state: KVCacheState) throws -> KVCacheState {
        var newCaches: [MLMultiArray] = []
        var newPositions: [MLMultiArray] = []
        newCaches.reserveCapacity(state.caches.count)
        newPositions.reserveCapacity(state.positions.count)

        for cache in state.caches {
            let copy = try MLMultiArray(shape: cache.shape, dataType: .float32)
            let srcPtr = cache.dataPointer
            let dstPtr = copy.dataPointer
            dstPtr.copyMemory(from: srcPtr, byteCount: cache.count * MemoryLayout<Float>.size)
            newCaches.append(copy)
        }

        for pos in state.positions {
            let copy = try MLMultiArray(shape: [1], dataType: .float32)
            copy[0] = pos[0]
            newPositions.append(copy)
        }

        return KVCacheState(caches: newCaches, positions: newPositions)
    }

    /// Run the conditioning step model for a single token, updating the KV cache in place.
    static func runCondStep(
        conditioning: MLMultiArray,
        state: inout KVCacheState,
        model: MLModel
    ) async throws {
        var inputDict: [String: Any] = [
            "conditioning": conditioning
        ]

        for i in 0..<PocketTtsConstants.kvCacheLayers {
            inputDict["cache\(i)"] = state.caches[i]
            inputDict["position\(i)"] = state.positions[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        for i in 0..<PocketTtsConstants.kvCacheLayers {
            guard let newCache = output.featureValue(for: CondStepKeys.cacheKeys[i])?.multiArrayValue
            else {
                throw TTSError.processingFailed(
                    "Missing cond_step cache output: \(CondStepKeys.cacheKeys[i])")
            }
            guard let newPos = output.featureValue(for: CondStepKeys.positionKeys[i])?.multiArrayValue
            else {
                throw TTSError.processingFailed(
                    "Missing cond_step position output: \(CondStepKeys.positionKeys[i])")
            }
            state.caches[i] = newCache
            state.positions[i] = newPos
        }
    }

    /// Prefill the KV cache with voice and text conditioning tokens.
    ///
    /// Processes voice tokens first, then text tokens (critical ordering).
    static func prefillKVCache(
        voiceData: PocketTtsVoiceData,
        textEmbeddings: [[Float]],
        model: MLModel
    ) async throws -> KVCacheState {
        var state = try emptyKVCacheState()
        let dim = PocketTtsConstants.embeddingDim

        // Voice tokens first (positions 0..124)
        let voiceTokenCount = voiceData.promptLength
        for tokenIdx in 0..<voiceTokenCount {
            let token = try createConditioningToken(
                from: voiceData.audioPrompt,
                offset: tokenIdx * dim,
                dim: dim
            )
            try await runCondStep(conditioning: token, state: &state, model: model)
        }

        // Text tokens next
        for embedding in textEmbeddings {
            let token = try createConditioningToken(from: embedding, offset: 0, dim: dim)
            try await runCondStep(conditioning: token, state: &state, model: model)
        }

        let finalPos = state.positions[0][0].floatValue
        logger.info("KV cache prefilled to position \(Int(finalPos))")

        return state
    }

    /// Create a [1, 1, 1024] MLMultiArray from a float slice.
    private static func createConditioningToken(
        from source: [Float], offset: Int, dim: Int
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(
            shape: [1, 1, NSNumber(value: dim)], dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: dim)
        source.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            ptr.update(from: base.advanced(by: offset), count: dim)
        }
        return array
    }

    /// Run the generation step model, returning transformer output and EOS logit.
    static func runFlowLMStep(
        sequence: MLMultiArray,
        bosEmb: MLMultiArray,
        state: inout KVCacheState,
        model: MLModel
    ) async throws -> (transformerOut: MLMultiArray, eosLogit: Float) {
        var inputDict: [String: Any] = [
            "sequence": sequence,
            "bos_emb": bosEmb,
        ]

        for i in 0..<PocketTtsConstants.kvCacheLayers {
            inputDict["cache\(i)"] = state.caches[i]
            inputDict["position\(i)"] = state.positions[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Extract transformer output
        guard let transformerOut = output.featureValue(for: FlowLMStepKeys.transformerOut)?.multiArrayValue
        else {
            throw TTSError.processingFailed("Missing flowlm_step transformer output")
        }

        // Extract EOS logit
        guard let eosArray = output.featureValue(for: FlowLMStepKeys.eosLogit)?.multiArrayValue
        else {
            throw TTSError.processingFailed("Missing flowlm_step EOS logit")
        }
        let eosLogit = eosArray[0].floatValue

        // Update caches and positions
        for i in 0..<PocketTtsConstants.kvCacheLayers {
            guard
                let newCache = output.featureValue(for: FlowLMStepKeys.cacheKeys[i])?.multiArrayValue
            else {
                throw TTSError.processingFailed(
                    "Missing flowlm_step cache output: \(FlowLMStepKeys.cacheKeys[i])")
            }
            guard let newPos = output.featureValue(for: FlowLMStepKeys.positionKeys[i])?.multiArrayValue
            else {
                throw TTSError.processingFailed(
                    "Missing flowlm_step position output: \(FlowLMStepKeys.positionKeys[i])")
            }
            state.caches[i] = newCache
            state.positions[i] = newPos
        }

        return (transformerOut: transformerOut, eosLogit: eosLogit)
    }
}
