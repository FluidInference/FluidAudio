@preconcurrency import CoreML
import Foundation

/// Discovered CoreML input/output schema for the Mimi audio decoder.
///
/// `attn{0,1}_cache` ships shape `[2, 1, 256, 8, 64]` (seq-first). CoreML
/// auto-generates non-passthrough output names (`var_NNN`) at conversion
/// time, so we discover both the streaming-state mapping and the audio
/// output name from the loaded model.
struct PocketTtsMimiKeys: Sendable {

    /// Output name for the `[1, 1, 1920]` audio frame.
    let audioOutput: String

    /// Ordered streaming-state input → output mapping.
    let stateMapping: [(input: String, output: String)]

    /// Declared shape per state input (as integers). Used by
    /// `loadMimiInitialState` to allocate tensors matching the model.
    let stateShapes: [String: [Int]]

    enum DiscoveryError: Error, LocalizedError {
        case missingAudioOutput
        case unmatchedStateInput(name: String, shape: [Int])

        var errorDescription: String? {
            switch self {
            case .missingAudioOutput:
                return "PocketTTS Mimi decoder is missing a [1, 1, 1920] audio output"
            case .unmatchedStateInput(let name, let shape):
                return "PocketTTS Mimi decoder: no output of shape \(shape) for state input '\(name)'"
            }
        }
    }

    /// Canonical streaming-state input order. Used to disambiguate
    /// shape-bucket pairing (e.g. `attn0_cache` before `attn1_cache`).
    private static let canonicalStateOrder: [String] = [
        "upsample_partial",
        "attn0_cache", "attn0_offset",
        "attn1_cache", "attn1_offset",
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
        //    - Otherwise: match by shape, then disambiguate within a shape
        //      bucket by sorting outputs by trailing `var_NNN` and inputs
        //      in canonical order.
        //
        //    Single pass over outputs: identify pass-throughs by name match
        //    against `stateShapes`, bucket the rest by shape (excluding audio).
        var passThroughMap: [String: String] = [:]
        var outputsByShape: [[Int]: [String]] = [:]
        for (name, desc) in outputs where name != audio {
            if stateShapes[name] != nil {
                passThroughMap[name] = name
                continue
            }
            guard let constraint = desc.multiArrayConstraint else { continue }
            let shape = constraint.shape.map { $0.intValue }
            outputsByShape[shape, default: []].append(name)
        }
        for key in outputsByShape.keys {
            outputsByShape[key]?.sort { lhs, rhs in
                let li = PocketTtsLayerKeys.trailingNumber(in: lhs) ?? Int.max
                let ri = PocketTtsLayerKeys.trailingNumber(in: rhs) ?? Int.max
                if li != ri { return li < ri }
                return lhs < rhs
            }
        }

        // Walk canonical order once: pass-throughs land directly, others draw
        // from their shape bucket. Inputs outside the canonical list are
        // silently ignored — the canonical list is exhaustive across the v2
        // packs, and downstream shape matching would fail for any unknown
        // schema anyway.
        var orderedMapping: [(input: String, output: String)] = []
        for inputName in canonicalStateOrder {
            guard let shape = stateShapes[inputName] else { continue }
            if let passThrough = passThroughMap[inputName] {
                orderedMapping.append((input: inputName, output: passThrough))
                continue
            }
            guard var bucket = outputsByShape[shape], !bucket.isEmpty else {
                throw DiscoveryError.unmatchedStateInput(name: inputName, shape: shape)
            }
            let chosen = bucket.removeFirst()
            outputsByShape[shape] = bucket
            orderedMapping.append((input: inputName, output: chosen))
        }

        return PocketTtsMimiKeys(
            audioOutput: audio,
            stateMapping: orderedMapping,
            stateShapes: stateShapes
        )
    }

}
