@preconcurrency import CoreML
import Foundation

/// Discovered CoreML output names for one transformer model (cond_step or
/// flowlm_step).
///
/// CoreML auto-generates output names during tracing (`new_cache_N_internal_tensor_assign_2`,
/// `var_NNN`) and the exact numeric suffixes differ between 6L and 24L packs.
/// Rather than hardcoding the names per pack, we scan the model's output
/// description at load time and group outputs by tensor shape:
///
/// - `[2, 1, kvCacheMaxLen, 16, 64]` → KV cache (one per layer)
/// - `[1]`                           → position scalar (one per layer)
/// - `[1, 1, transformerDim]`        → transformer hidden state (flowlm_step only)
/// - `[1, 1, 1]`                     → EOS logit                (flowlm_step only)
///
/// Within each group we order by the numeric suffix in the name. Cache names
/// follow the closed form `new_cache_{2*i+1}_internal_tensor_assign_2` for
/// layers 0..N-2 with the last layer being `new_cache_internal_tensor_assign_2`
/// (no number — sorted last). Position names use `var_NNN` with irregular
/// strides that nevertheless increase monotonically per layer.
struct PocketTtsLayerKeys: Sendable {
    /// One cache output name per transformer layer, ordered by layer index.
    /// For split-KV (rank-4 `_ane`) models this holds the K cache names.
    let cacheKeys: [String]
    /// V cache output names for split-KV (rank-4 `_ane`) models, ordered by
    /// layer index. `nil` for the rank-5 packs (K+V share one tensor).
    let vCacheKeys: [String]?
    /// One position output name per transformer layer, ordered by layer index.
    let positionKeys: [String]
    /// Hidden-state output name (flowlm_step only). `nil` for cond_step.
    let transformerOut: String?
    /// EOS logit output name (flowlm_step only). `nil` for cond_step.
    let eosLogit: String?

    var layerCount: Int { cacheKeys.count }
    /// Whether the model uses rank-4 split k/v cache I/O.
    var isSplitKV: Bool { vCacheKeys != nil }

    /// Keys for the rank-4 `_ane` models, which ship EXPLICIT output names
    /// (the k/v caches share a shape, so shape-bucket discovery can't tell
    /// them apart — see mobius convert_flowlm_step_ane.py). No discovery:
    /// names are `new_k_cache{i}` / `new_v_cache{i}` / `new_position{i}`,
    /// plus `transformer_out` / `is_eos` for the FlowLM step.
    static func aneKeys(layers: Int, kind: ModelKind) -> PocketTtsLayerKeys {
        PocketTtsLayerKeys(
            cacheKeys: (0..<layers).map { "new_k_cache\($0)" },
            vCacheKeys: (0..<layers).map { "new_v_cache\($0)" },
            positionKeys: (0..<layers).map { "new_position\($0)" },
            transformerOut: kind == .flowlmStep ? "transformer_out" : nil,
            eosLogit: kind == .flowlmStep ? "is_eos" : nil
        )
    }

    enum DiscoveryError: Error, LocalizedError {
        case shapeMismatch(modelName: String, expectedLayers: Int, actualCaches: Int)
        case positionMismatch(modelName: String, cacheCount: Int, positionCount: Int)
        case missingFlowLMOutputs(modelName: String, hasTransformer: Bool, hasEos: Bool)

        var errorDescription: String? {
            switch self {
            case .shapeMismatch(let modelName, let expected, let actual):
                return
                    "PocketTTS layer-key discovery on \(modelName): expected \(expected) cache outputs, found \(actual)"
            case .positionMismatch(let modelName, let cacheCount, let positionCount):
                return
                    "PocketTTS layer-key discovery on \(modelName): \(cacheCount) cache outputs but \(positionCount) position outputs"
            case .missingFlowLMOutputs(let modelName, let hasTransformer, let hasEos):
                return
                    "PocketTTS \(modelName) missing flowlm outputs (transformer=\(hasTransformer), eos=\(hasEos))"
            }
        }
    }

    /// Discover the output keys for a `cond_step` or `flowlm_step` CoreML model.
    ///
    /// - Parameters:
    ///   - model: The compiled CoreML model.
    ///   - kind: Which model this is — affects whether transformer/eos
    ///     outputs are required.
    ///   - expectedLayers: Sanity check for the layer count (6 or 24).
    static func discover(
        from model: MLModel,
        kind: ModelKind,
        expectedLayers: Int,
        modelName: String
    ) throws -> PocketTtsLayerKeys {
        let outputs = model.modelDescription.outputDescriptionsByName

        // Bucket outputs by shape.
        var cacheCandidates: [String] = []
        var positionCandidates: [String] = []
        var transformerCandidate: String?
        var eosCandidate: String?

        let cacheShape = [
            2, 1, PocketTtsConstants.kvCacheMaxLen, 16, 64,
        ]
        let transformerShape = [1, 1, PocketTtsConstants.transformerDim]
        let eosShape = [1, 1, 1]
        let positionShape = [1]

        for (name, desc) in outputs {
            guard let constraint = desc.multiArrayConstraint else { continue }
            let shape = constraint.shape.map { $0.intValue }

            if shape == cacheShape {
                cacheCandidates.append(name)
            } else if shape == positionShape {
                positionCandidates.append(name)
            } else if shape == transformerShape {
                transformerCandidate = name
            } else if shape == eosShape {
                eosCandidate = name
            }
        }

        // Sort caches by extracted numeric suffix; "new_cache_internal_..."
        // (no number) sorts as "last" (largest layer index).
        cacheCandidates.sort { lhs, rhs in
            let li = cacheLayerIndex(from: lhs) ?? Int.max
            let ri = cacheLayerIndex(from: rhs) ?? Int.max
            if li != ri { return li < ri }
            return lhs < rhs
        }

        // Sort positions by trailing numeric suffix.
        positionCandidates.sort { lhs, rhs in
            let li = trailingNumber(in: lhs) ?? Int.max
            let ri = trailingNumber(in: rhs) ?? Int.max
            if li != ri { return li < ri }
            return lhs < rhs
        }

        if cacheCandidates.count != expectedLayers {
            throw DiscoveryError.shapeMismatch(
                modelName: modelName,
                expectedLayers: expectedLayers,
                actualCaches: cacheCandidates.count
            )
        }

        if positionCandidates.count != cacheCandidates.count {
            throw DiscoveryError.positionMismatch(
                modelName: modelName,
                cacheCount: cacheCandidates.count,
                positionCount: positionCandidates.count
            )
        }

        switch kind {
        case .condStep:
            return PocketTtsLayerKeys(
                cacheKeys: cacheCandidates,
                vCacheKeys: nil,
                positionKeys: positionCandidates,
                transformerOut: nil,
                eosLogit: nil
            )
        case .flowlmStep:
            guard let transformerOut = transformerCandidate, let eosLogit = eosCandidate else {
                throw DiscoveryError.missingFlowLMOutputs(
                    modelName: modelName,
                    hasTransformer: transformerCandidate != nil,
                    hasEos: eosCandidate != nil
                )
            }
            return PocketTtsLayerKeys(
                cacheKeys: cacheCandidates,
                vCacheKeys: nil,
                positionKeys: positionCandidates,
                transformerOut: transformerOut,
                eosLogit: eosLogit
            )
        }
    }

    enum ModelKind {
        case condStep
        case flowlmStep
    }

    // MARK: - Name parsing

    /// Extract the layer index from a cache output name.
    ///
    /// Pattern:
    ///  - `new_cache_<N>_internal_tensor_assign_2` → returns `(N - 1) / 2`
    ///  - `new_cache_internal_tensor_assign_2`     → returns `nil` (sorts last)
    private static func cacheLayerIndex(from name: String) -> Int? {
        // Strip the "new_cache_" prefix, then take everything up to the next "_".
        guard name.hasPrefix("new_cache_") else { return nil }
        let after = name.dropFirst("new_cache_".count)
        guard let underscore = after.firstIndex(of: "_") else { return nil }
        let head = after[..<underscore]
        guard let raw = Int(head) else { return nil }
        // Cache numbering uses odd numbers (1, 3, 5, ...) so map to 0, 1, 2, ...
        return (raw - 1) / 2
    }

    /// Extract the trailing run of digits from a name like `var_445`.
    /// Shared with `PocketTtsMimiKeys` for `var_NNN` output ordering.
    static func trailingNumber(in name: String) -> Int? {
        let suffix = name.reversed().prefix(while: \.isNumber)
        guard !suffix.isEmpty else { return nil }
        return Int(String(suffix.reversed()))
    }
}
