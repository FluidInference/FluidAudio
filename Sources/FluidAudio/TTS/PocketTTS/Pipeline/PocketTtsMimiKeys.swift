@preconcurrency import CoreML
import Foundation

/// Discovered CoreML input/output schema for the Mimi audio decoder.
///
/// The Mimi decoder ships in two upstream variants that differ in attention
/// cache layout and (auto-generated) output names:
///  - **Legacy English** — `attn{0,1}_cache` shape `[2, 1, 8, 256, 64]`
///    (heads-first) plus `attn{0,1}_end_offset` inputs.
///  - **v2 multi-language packs** — `attn{0,1}_cache` shape
///    `[2, 1, 256, 8, 64]` (seq-first), no `_end_offset` inputs.
///
/// CoreML auto-generates non-passthrough output names (`var_NNN`) at
/// conversion time so they differ between packs. To keep one Swift runtime
/// path we discover both the streaming-state mapping and the audio output
/// name from the loaded model.
struct PocketTtsMimiKeys: Sendable {

    /// Output name for the `[1, 1, 1920]` audio frame.
    let audioOutput: String

    /// Ordered streaming-state input → output mapping. Only contains state
    /// inputs the loaded model actually accepts (so packs missing
    /// `attn*_end_offset` simply omit those entries).
    let stateMapping: [(input: String, output: String)]

    /// Declared shape per state input (as integers). Used by
    /// `loadMimiInitialState` to allocate tensors matching the model.
    let stateShapes: [String: [Int]]

    enum DiscoveryError: Error, LocalizedError {
        case missingAudioOutput
        case unmatchedStateInput(name: String, shape: [Int])
        case ambiguousMatch(name: String)

        var errorDescription: String? {
            switch self {
            case .missingAudioOutput:
                return "PocketTTS Mimi decoder is missing a [1, 1, 1920] audio output"
            case .unmatchedStateInput(let name, let shape):
                return "PocketTTS Mimi decoder: no output of shape \(shape) for state input '\(name)'"
            case .ambiguousMatch(let name):
                return "PocketTTS Mimi decoder: could not deterministically pair state input '\(name)'"
            }
        }
    }

    /// Canonical streaming-state input order. Used to disambiguate
    /// shape-bucket pairing (e.g. `attn0_cache` before `attn1_cache`).
    /// Inputs absent in a given pack (e.g. `attn{0,1}_end_offset` for v2)
    /// are simply skipped.
    private static let canonicalStateOrder: [String] = [
        "upsample_partial",
        "attn0_cache", "attn0_offset", "attn0_end_offset",
        "attn1_cache", "attn1_offset", "attn1_end_offset",
        "conv0_prev", "conv0_first",
        "convtr0_partial",
        "res0_conv0_prev", "res0_conv0_first",
        "res0_conv1_prev", "res0_conv1_first",
        "convtr1_partial",
        "res1_conv0_prev", "res1_conv0_first",
        "res1_conv1_prev", "res1_conv1_first",
        "convtr2_partial",
        "res2_conv0_prev", "res2_conv0_first",
        "res2_conv1_prev", "res2_conv1_first",
        "conv_final_prev", "conv_final_first",
    ]

    /// Discover the Mimi schema from a loaded `MLModel`.
    static func discover(from model: MLModel) throws -> PocketTtsMimiKeys {
        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName

        // 1. Audio output is the only `[1, 1, 1920]` tensor.
        let audioShape = [1, 1, PocketTtsConstants.samplesPerFrame]
        let audioOutput = outputs.first { _, desc in
            guard let constraint = desc.multiArrayConstraint else { return false }
            return constraint.shape.map { $0.intValue } == audioShape
        }?.key

        guard let audio = audioOutput else {
            throw DiscoveryError.missingAudioOutput
        }

        // 2. Build state input set + shapes (everything except `latent`).
        var stateShapes: [String: [Int]] = [:]
        for (name, desc) in inputs where name != "latent" {
            guard let constraint = desc.multiArrayConstraint else { continue }
            stateShapes[name] = constraint.shape.map { $0.intValue }
        }

        // 3. Pair inputs to outputs.
        //    - Pass-through: output name equals input name (e.g. `conv0_first`,
        //      `res*_conv1_prev` zero-shape carry-throughs).
        //    - Semantic-named: outputs containing `end_offset` are reserved
        //      for inputs containing `end_offset` so they can't be confused
        //      with similarly-shaped `var_NNN` offset outputs sharing
        //      shape `[1]`.
        //    - Otherwise: match by shape, then disambiguate within a shape
        //      bucket by sorting outputs by trailing `var_NNN` and inputs
        //      in canonical order.
        var availableOutputs = outputs
        availableOutputs.removeValue(forKey: audio)

        // Remove pass-throughs first (cheap to identify).
        var passThroughMap: [String: String] = [:]
        for inputName in stateShapes.keys {
            if availableOutputs[inputName] != nil {
                passThroughMap[inputName] = inputName
                availableOutputs.removeValue(forKey: inputName)
            }
        }

        // Reserve `*end_offset*` outputs exclusively for `*end_offset*` inputs.
        // Pair them in canonical order, sorted by trailing digits ascending so
        // `attn0_end_offset → new_end_offset_1`, `attn1_end_offset → new_end_offset`.
        let endOffsetInputs = canonicalStateOrder.filter {
            $0.contains("end_offset") && stateShapes[$0] != nil && passThroughMap[$0] == nil
        }
        var endOffsetOutputs = availableOutputs.keys.filter { $0.contains("end_offset") }.sorted {
            let li = trailingNumber(in: $0) ?? Int.max
            let ri = trailingNumber(in: $1) ?? Int.max
            if li != ri { return li < ri }
            return $0 < $1
        }
        var endOffsetMap: [String: String] = [:]
        for inputName in endOffsetInputs {
            guard !endOffsetOutputs.isEmpty else {
                throw DiscoveryError.unmatchedStateInput(
                    name: inputName, shape: stateShapes[inputName] ?? [])
            }
            let chosen = endOffsetOutputs.removeFirst()
            endOffsetMap[inputName] = chosen
            availableOutputs.removeValue(forKey: chosen)
        }

        // Bucket remaining outputs by shape, sorted by var-number ascending.
        var outputsByShape: [[Int]: [String]] = [:]
        for (name, desc) in availableOutputs {
            guard let constraint = desc.multiArrayConstraint else { continue }
            let shape = constraint.shape.map { $0.intValue }
            outputsByShape[shape, default: []].append(name)
        }
        for key in outputsByShape.keys {
            outputsByShape[key]?.sort { lhs, rhs in
                let li = trailingNumber(in: lhs) ?? Int.max
                let ri = trailingNumber(in: rhs) ?? Int.max
                if li != ri { return li < ri }
                return lhs < rhs
            }
        }

        // Walk canonical order, taking outputs from each shape bucket. Skip
        // inputs already resolved via pass-through or end-offset reservation.
        var nonPassThroughInputs: [String] = []
        for name in canonicalStateOrder
        where stateShapes[name] != nil
            && passThroughMap[name] == nil
            && endOffsetMap[name] == nil
        {
            nonPassThroughInputs.append(name)
        }
        // Any inputs not in canonical list (defensive) appended in name order.
        for name in stateShapes.keys.sorted()
        where !canonicalStateOrder.contains(name)
            && passThroughMap[name] == nil
            && endOffsetMap[name] == nil
        {
            nonPassThroughInputs.append(name)
        }

        var resolvedMapping: [String: String] = passThroughMap
        for (k, v) in endOffsetMap { resolvedMapping[k] = v }
        for inputName in nonPassThroughInputs {
            guard let shape = stateShapes[inputName] else { continue }
            guard var bucket = outputsByShape[shape], !bucket.isEmpty else {
                throw DiscoveryError.unmatchedStateInput(name: inputName, shape: shape)
            }
            let chosen = bucket.removeFirst()
            outputsByShape[shape] = bucket
            resolvedMapping[inputName] = chosen
        }

        // Emit mapping in canonical order so iteration is deterministic.
        var orderedMapping: [(input: String, output: String)] = []
        for name in canonicalStateOrder {
            if let out = resolvedMapping[name] {
                orderedMapping.append((input: name, output: out))
            }
        }
        // Append any non-canonical inputs at the end (defensive).
        for name in stateShapes.keys.sorted() where !canonicalStateOrder.contains(name) {
            if let out = resolvedMapping[name] {
                orderedMapping.append((input: name, output: out))
            }
        }

        return PocketTtsMimiKeys(
            audioOutput: audio,
            stateMapping: orderedMapping,
            stateShapes: stateShapes
        )
    }

    /// Extract the trailing run of digits from a name like `var_445`.
    private static func trailingNumber(in name: String) -> Int? {
        var digits = ""
        for char in name.reversed() {
            if char.isNumber {
                digits.append(char)
            } else {
                break
            }
        }
        guard !digits.isEmpty else { return nil }
        return Int(String(digits.reversed()))
    }
}
