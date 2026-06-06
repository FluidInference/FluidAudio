@preconcurrency import CoreML
import Foundation

extension PocketTtsSynthesizer {

    /// Mutable KV cache state passed through conditioning and generation steps.
    ///
    /// One cache per transformer layer stores the K (key) and V (value) projections
    /// for every processed token. This avoids recomputing K/V for past tokens —
    /// each new step only computes its own K/V, then reads all cached K/V via attention.
    struct KVCacheState {
        /// `N` KV cache arrays (one per transformer layer), each shaped
        /// `[2, 1, kvCacheMaxLen, 16, 64]`:
        ///  - `2`: K and V tensors (index 0 = keys, index 1 = values)
        ///  - `1`: batch size
        ///  - `kvCacheMaxLen` (512): pre-allocated position slots
        ///  - `16`: attention heads
        ///  - `64`: dims per head (16 × 64 = 1024 total)
        ///
        /// `N` is 6 for 6L packs and 24 for `*_24l` packs.
        var caches: [MLMultiArray]
        /// `N` position counters (one per layer) tracking the next write slot
        /// in each cache.
        var positions: [MLMultiArray]
    }

    /// Create an empty KV cache state (all zeros, positions at 0).
    static func emptyKVCacheState(layers: Int) throws -> KVCacheState {
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
        KVCacheState(
            caches: try state.caches.map(deepCopy),
            positions: try state.positions.map(deepCopy)
        )
    }

    /// Deep-copy an `MLMultiArray`, preserving shape and dtype.
    ///
    /// Handles fp16 (UInt16-sized) and fp32-or-other (Float-sized) data
    /// types, and gracefully no-ops for zero-element tensors (e.g. Mimi's
    /// `res*_conv1_prev: [1, 128, 0]`).
    static func deepCopy(_ array: MLMultiArray) throws -> MLMultiArray {
        let copy = try MLMultiArray(shape: array.shape, dataType: array.dataType)
        let byteSize: Int
        switch array.dataType {
        case .float16:
            byteSize = array.count * MemoryLayout<UInt16>.size
        default:
            byteSize = array.count * MemoryLayout<Float>.size
        }
        if byteSize > 0 {
            copy.dataPointer.copyMemory(from: array.dataPointer, byteCount: byteSize)
        }
        return copy
    }

    /// Run the conditioning step model for a single token, updating the KV cache in place.
    ///
    /// `cond_step` and `flowlm_step` share the same transformer weights. This function
    /// runs the transformer in "prefill mode": it processes one conditioning token
    /// (voice embedding or text embedding), computes K/V projections, and writes them
    /// into the cache at the current position. No audio is produced.
    static func runCondStep(
        conditioning: MLMultiArray,
        state: inout KVCacheState,
        model: MLModel,
        layerKeys: PocketTtsLayerKeys
    ) async throws {
        let layers = layerKeys.layerCount
        var inputDict: [String: Any] = [
            "conditioning": conditioning
        ]

        for i in 0..<layers {
            inputDict["cache\(i)"] = state.caches[i]
            inputDict["position\(i)"] = state.positions[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        for i in 0..<layers {
            guard let newCache = output.featureValue(for: layerKeys.cacheKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing cond_step cache output: \(layerKeys.cacheKeys[i])")
            }
            guard let newPos = output.featureValue(for: layerKeys.positionKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing cond_step position output: \(layerKeys.positionKeys[i])")
            }
            state.caches[i] = newCache
            state.positions[i] = newPos
        }
    }

    /// Run the one-shot conditioning prefill model: fill the whole voice or
    /// text block in a single `predict()` instead of one call per token.
    ///
    /// `flatConditioning` is `tokenCount × embeddingDim` row-major. The block is
    /// copied into a `[1, T_max, 1024]` input and zero-padded; `valid_len` =
    /// `tokenCount` tells the model how many rows are real, so padded rows write
    /// to a masked dump slot and the position advances by `tokenCount` only
    /// (see mobius traceable_cond_prefill.py — neutralizes the Trial 8 padding
    /// corruption). Output schema is identical to `cond_step`.
    static func runCondPrefill(
        flatConditioning: [Float],
        tokenCount: Int,
        state: inout KVCacheState,
        model: MLModel,
        layerKeys: PocketTtsLayerKeys
    ) async throws {
        let dim = PocketTtsConstants.embeddingDim
        let tMax = PocketTtsConstants.condPrefillMaxTokens
        guard tokenCount > 0, tokenCount <= tMax else {
            throw PocketTTSError.processingFailed(
                "cond_prefill token count \(tokenCount) out of range (1...\(tMax))")
        }
        let layers = layerKeys.layerCount

        let conditioning = try MLMultiArray(
            shape: [1, NSNumber(value: tMax), NSNumber(value: dim)], dataType: .float32)
        let condPtr = conditioning.dataPointer.bindMemory(to: Float.self, capacity: tMax * dim)
        condPtr.initialize(repeating: 0, count: tMax * dim)
        let copyCount = min(tokenCount * dim, flatConditioning.count)
        flatConditioning.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return }
            condPtr.update(from: base, count: copyCount)
        }

        let validLen = try MLMultiArray(shape: [1], dataType: .float32)
        validLen[0] = NSNumber(value: Float(tokenCount))

        var inputDict: [String: Any] = [
            "conditioning": conditioning,
            "valid_len": validLen,
        ]
        for i in 0..<layers {
            inputDict["cache\(i)"] = state.caches[i]
            inputDict["position\(i)"] = state.positions[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        for i in 0..<layers {
            guard let newCache = output.featureValue(for: layerKeys.cacheKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing cond_prefill cache output: \(layerKeys.cacheKeys[i])")
            }
            guard let newPos = output.featureValue(for: layerKeys.positionKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing cond_prefill position output: \(layerKeys.positionKeys[i])")
            }
            state.caches[i] = newCache
            state.positions[i] = newPos
        }
    }

    /// Prefill a KV cache state with voice conditioning tokens.
    ///
    /// Prepends a single `bos_before_voice` token to match pocket-tts 2.0.0's
    /// `flow_lm.bos_before_voice` prefix (see FluidAudio #592 — without it
    /// `cond_step` diverges from the deployed flowlm/flow_decoder weights and
    /// the LM emits EOS within a few steps, producing garbled audio). Then
    /// processes all voice tokens from `voiceData.audioPrompt`, writing K/V
    /// projections into the cache starting at the current position.
    ///
    /// `bosBeforeVoice` must be provided whenever `voiceData.audioPrompt`
    /// has content (i.e. cloned voices); shipped v2 voices skip this path
    /// entirely via `cacheSnapshot`.
    static func prefillKVCacheVoice(
        state: KVCacheState,
        voiceData: PocketTtsVoiceData,
        bosBeforeVoice: [Float]?,
        model: MLModel,
        layerKeys: PocketTtsLayerKeys,
        prefillModel: MLModel,
        prefillLayerKeys: PocketTtsLayerKeys?,
        useFastPrefill: Bool
    ) async throws -> KVCacheState {
        var state = state
        let dim = PocketTtsConstants.embeddingDim

        let voiceTokenCount = voiceData.promptLength
        guard voiceTokenCount > 0 else {
            // Nothing to prefill (e.g. session warmup with empty cloned
            // voice). Skip the BOS prepend too — runtime callers that go
            // through `prefillKVCache` only hit this branch when both
            // `cacheSnapshot == nil` and `promptLength == 0`, which is a
            // no-op.
            return state
        }

        guard let bosBeforeVoice else {
            throw PocketTTSError.processingFailed(
                "PocketTTS v1 cloned-voice prefill requires bos_before_voice constant. "
                    + "Re-download the language pack to get constants_bin/bos_before_voice.bin "
                    + "(added in the FluidAudio #592 fix)."
            )
        }
        guard bosBeforeVoice.count == dim else {
            throw PocketTTSError.processingFailed(
                "bos_before_voice has \(bosBeforeVoice.count) floats, expected \(dim)"
            )
        }

        // Fast path: one-shot prefill of [bos, voice...] in a single predict
        // when cond_prefill is available and the block fits T_max.
        let totalVoiceTokens = 1 + voiceTokenCount
        if useFastPrefill, let prefillLayerKeys,
            totalVoiceTokens <= PocketTtsConstants.condPrefillMaxTokens
        {
            var flat = [Float]()
            flat.reserveCapacity(totalVoiceTokens * dim)
            flat.append(contentsOf: bosBeforeVoice)
            flat.append(contentsOf: voiceData.audioPrompt[0..<(voiceTokenCount * dim)])
            try await runCondPrefill(
                flatConditioning: flat, tokenCount: totalVoiceTokens,
                state: &state, model: prefillModel, layerKeys: prefillLayerKeys)
            return state
        }

        let bosToken = try createConditioningToken(
            from: bosBeforeVoice, offset: 0, dim: dim)
        try await runCondStep(
            conditioning: bosToken, state: &state, model: model, layerKeys: layerKeys)

        for tokenIdx in 0..<voiceTokenCount {
            let token = try createConditioningToken(
                from: voiceData.audioPrompt,
                offset: tokenIdx * dim,
                dim: dim
            )
            try await runCondStep(
                conditioning: token, state: &state, model: model, layerKeys: layerKeys)
        }

        return state
    }

    /// Prefill a KV cache state with text embedding tokens.
    ///
    /// Processes all text embeddings, writing K/V projections into the cache
    /// starting at the current position.
    static func prefillKVCacheText(
        state: KVCacheState,
        textEmbeddings: [[Float]],
        model: MLModel,
        layerKeys: PocketTtsLayerKeys,
        prefillModel: MLModel,
        prefillLayerKeys: PocketTtsLayerKeys?,
        useFastPrefill: Bool
    ) async throws -> KVCacheState {
        var state = state
        let dim = PocketTtsConstants.embeddingDim

        // Fast path: one-shot prefill of the whole text block in a single
        // predict when cond_prefill is available and the block fits T_max.
        if useFastPrefill, let prefillLayerKeys,
            !textEmbeddings.isEmpty,
            textEmbeddings.count <= PocketTtsConstants.condPrefillMaxTokens
        {
            var flat = [Float]()
            flat.reserveCapacity(textEmbeddings.count * dim)
            for embedding in textEmbeddings { flat.append(contentsOf: embedding) }
            try await runCondPrefill(
                flatConditioning: flat, tokenCount: textEmbeddings.count,
                state: &state, model: prefillModel, layerKeys: prefillLayerKeys)
            return state
        }

        for embedding in textEmbeddings {
            let token = try createConditioningToken(from: embedding, offset: 0, dim: dim)
            try await runCondStep(
                conditioning: token, state: &state, model: model, layerKeys: layerKeys)
        }

        return state
    }

    /// Build a `KVCacheState` from a pre-baked v2 voice snapshot.
    ///
    /// Each layer's source cache `[2, 1, seqLen, 16, 64]` is copied into the
    /// first `seqLen` positions of a fresh `[2, 1, kvCacheMaxLen, 16, 64]`
    /// allocation. The K block (outer dim 0) and V block (outer dim 1) are
    /// copied independently because the dest seq capacity is larger than
    /// the source — they don't lie at adjacent offsets in the dest.
    /// `position{i}` is initialized from the snapshot's per-layer offset
    /// (typically equal to `seqLen`).
    static func kvCacheStateFromSnapshot(
        _ snapshot: PocketTtsVoiceCacheSnapshot,
        layers: Int
    ) throws -> KVCacheState {
        guard snapshot.layers.count == layers else {
            throw PocketTTSError.processingFailed(
                "voice snapshot layer count \(snapshot.layers.count) != model layer count \(layers)"
            )
        }
        let destSeq = PocketTtsConstants.kvCacheMaxLen
        let srcSeq = snapshot.cacheSeqLen
        guard srcSeq <= destSeq else {
            throw PocketTTSError.processingFailed(
                "voice snapshot seqLen \(srcSeq) exceeds model capacity \(destSeq)"
            )
        }

        // For shape [2, 1, seq, 16, 64] row-major:
        //   K block size = 1 * seq * 16 * 64 floats
        //   V block size = same
        let perKVFloats = 1 * srcSeq * 16 * 64
        let destPerKVFloats = 1 * destSeq * 16 * 64
        let copyBytes = perKVFloats * MemoryLayout<Float>.size

        let shape: [NSNumber] = [
            2, 1, NSNumber(value: destSeq), 16, 64,
        ]

        var caches: [MLMultiArray] = []
        var positions: [MLMultiArray] = []
        caches.reserveCapacity(layers)
        positions.reserveCapacity(layers)

        for layerIdx in 0..<layers {
            let source = snapshot.layers[layerIdx]
            // Source flat array MUST equal 2 * perKVFloats elements.
            guard source.cache.count == 2 * perKVFloats else {
                throw PocketTTSError.processingFailed(
                    "voice snapshot layer \(layerIdx) has \(source.cache.count) floats, expected \(2 * perKVFloats)"
                )
            }

            let cache = try MLMultiArray(shape: shape, dataType: .float32)
            let cachePtr = cache.dataPointer.bindMemory(to: Float.self, capacity: cache.count)
            cachePtr.initialize(repeating: 0, count: cache.count)

            // Copy K (source[0..perKVFloats) → dest[0..perKVFloats))
            // Copy V (source[perKVFloats..2*perKVFloats) → dest[destPerKVFloats..destPerKVFloats+perKVFloats))
            source.cache.withUnsafeBufferPointer { src in
                guard let srcBase = src.baseAddress else { return }
                memcpy(cachePtr, srcBase, copyBytes)
                memcpy(
                    cachePtr.advanced(by: destPerKVFloats),
                    srcBase.advanced(by: perKVFloats),
                    copyBytes)
            }
            caches.append(cache)

            let pos = try MLMultiArray(shape: [1], dataType: .float32)
            pos[0] = NSNumber(value: Float(source.offset))
            positions.append(pos)
        }

        return KVCacheState(caches: caches, positions: positions)
    }

    /// Prefill the KV cache with voice and text conditioning tokens.
    ///
    /// Processes voice tokens first, then text tokens. This ordering is critical —
    /// the model was trained with voice conditioning before text, so reversing it
    /// produces garbage. Each chunk gets a fresh cache because the 512-position
    /// limit can't hold multiple chunks' worth of context.
    ///
    /// Two voice paths:
    ///  - **Snapshot** (shipped voices): drop pre-baked K/V into cache, skip
    ///    `cond_step` voice prefill entirely. `bos_before_voice` is already
    ///    baked into the snapshot.
    ///  - **Flat audio prompt** (cloned voices): feed `bos_before_voice`
    ///    then every voice token through `cond_step`.
    /// Text prefill runs identically in both cases.
    static func prefillKVCache(
        voiceData: PocketTtsVoiceData,
        textEmbeddings: [[Float]],
        bosBeforeVoice: [Float]?,
        model: MLModel,
        layerKeys: PocketTtsLayerKeys,
        prefillModel: MLModel,
        prefillLayerKeys: PocketTtsLayerKeys?,
        useFastPrefill: Bool
    ) async throws -> KVCacheState {
        var state: KVCacheState
        if let snapshot = voiceData.cacheSnapshot {
            state = try kvCacheStateFromSnapshot(snapshot, layers: layerKeys.layerCount)
        } else {
            let emptyState = try emptyKVCacheState(layers: layerKeys.layerCount)
            state = try await prefillKVCacheVoice(
                state: emptyState, voiceData: voiceData,
                bosBeforeVoice: bosBeforeVoice,
                model: model, layerKeys: layerKeys,
                prefillModel: prefillModel, prefillLayerKeys: prefillLayerKeys,
                useFastPrefill: useFastPrefill
            )
        }
        state = try await prefillKVCacheText(
            state: state, textEmbeddings: textEmbeddings, model: model, layerKeys: layerKeys,
            prefillModel: prefillModel, prefillLayerKeys: prefillLayerKeys,
            useFastPrefill: useFastPrefill
        )

        let finalPos = state.positions[0][0].floatValue
        logger.info("KV cache prefilled to position \(Int(finalPos))")

        return state
    }

    /// Create a `[1, 1, 1024]` MLMultiArray from a float slice.
    ///
    /// Shape: batch=1, sequence_length=1 (one token at a time), embedding_dim=1024.
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
    ///
    /// Same transformer as `cond_step`, now in "generate mode". Takes the previous
    /// audio latent (or NaN for BOS), attends to all cached K/V from conditioning
    /// and prior generation steps, and produces a 1024-d hidden state (for flow_decoder)
    /// plus an EOS logit indicating whether the model is done speaking.
    /// Also writes this step's own K/V into the cache for future steps.
    static func runFlowLMStep(
        sequence: MLMultiArray,
        bosEmb: MLMultiArray,
        state: inout KVCacheState,
        model: MLModel,
        layerKeys: PocketTtsLayerKeys
    ) async throws -> (transformerOut: MLMultiArray, eosLogit: Float) {
        guard let transformerKey = layerKeys.transformerOut, let eosKey = layerKeys.eosLogit
        else {
            throw PocketTTSError.processingFailed(
                "flowlm_step layer keys missing transformer/eos outputs")
        }

        let layers = layerKeys.layerCount
        var inputDict: [String: Any] = [
            "sequence": sequence,
            "bos_emb": bosEmb,
        ]

        for i in 0..<layers {
            inputDict["cache\(i)"] = state.caches[i]
            inputDict["position\(i)"] = state.positions[i]
        }

        let input = try MLDictionaryFeatureProvider(dictionary: inputDict)
        let output = try await model.compatPrediction(from: input, options: MLPredictionOptions())

        // Extract transformer output
        guard let transformerOut = output.featureValue(for: transformerKey)?.multiArrayValue
        else {
            throw PocketTTSError.processingFailed("Missing flowlm_step transformer output")
        }

        // Extract EOS logit
        guard let eosArray = output.featureValue(for: eosKey)?.multiArrayValue
        else {
            throw PocketTTSError.processingFailed("Missing flowlm_step EOS logit")
        }
        let eosLogit = eosArray[0].floatValue

        // Update caches and positions
        for i in 0..<layers {
            guard
                let newCache = output.featureValue(for: layerKeys.cacheKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing flowlm_step cache output: \(layerKeys.cacheKeys[i])")
            }
            guard let newPos = output.featureValue(for: layerKeys.positionKeys[i])?.multiArrayValue
            else {
                throw PocketTTSError.processingFailed(
                    "Missing flowlm_step position output: \(layerKeys.positionKeys[i])")
            }
            state.caches[i] = newCache
            state.positions[i] = newPos
        }

        return (transformerOut: transformerOut, eosLogit: eosLogit)
    }
}
