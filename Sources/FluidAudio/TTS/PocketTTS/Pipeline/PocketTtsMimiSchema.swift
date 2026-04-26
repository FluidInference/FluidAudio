@preconcurrency import CoreML
import Foundation

/// Discovered Mimi decoder I/O schema.
///
/// The Mimi decoder takes a `latent` input + N state tensors that carry the
/// streaming context across frames, and returns an audio waveform + N updated
/// state tensors.
///
/// Two schema variants exist in the wild:
///
/// - **v2 (semantic, mobius post-rename):** outputs are renamed to
///   `audio` + `<input_name>_out` at conversion time, so discovery is a
///   trivial pairing.
///
/// - **v1 (legacy, current HF FluidInference/pocket-tts-coreml English):**
///   outputs use CoreML's auto-generated `var_NNN` / `cast_NN` names. Inputs
///   are still semantic, so we fall back to a hand-curated input → output
///   table baked at conversion time.
///
/// Discovery checks for the v2 convention first (zero hardcoded names) and
/// falls back to v1 only when the model lacks an `audio` output.
struct PocketTtsMimiSchema: Sendable {

    /// Output name carrying the audio waveform (`[1, 1, samplesPerFrame]`).
    let audioOutputName: String

    /// Pairing of state input name → corresponding state output name.
    /// Order is preserved from the model's input description so callers can
    /// iterate deterministically when copying state forward.
    let stateMapping: [(input: String, output: String)]

    /// Expected MLMultiArray shape for each state input (keyed by input name).
    /// Used to zero-init the streaming state directly from the model's
    /// description without depending on a sidecar manifest file.
    let stateInputShapes: [String: [Int]]

    /// Expected MLMultiArray dtype for state inputs (uniform across the
    /// model — all FP32 for v1, all FP16 for v2).
    let stateInputDataType: MLMultiArrayDataType

    /// Expected MLMultiArray dtype for the `latent` input (matches state
    /// dtype in practice, but kept separate for safety).
    let latentDataType: MLMultiArrayDataType

    /// Names of all state inputs (ordered).
    var stateInputNames: [String] { stateMapping.map { $0.input } }

    enum DiscoveryError: Error, LocalizedError {
        case missingAudioOutput(modelName: String, candidates: [String])
        case missingStateOutput(inputName: String, expectedOutput: String)

        var errorDescription: String? {
            switch self {
            case .missingAudioOutput(let modelName, let candidates):
                return
                    "PocketTTS \(modelName): could not find audio output (no `audio` output and no v1 fallback hit). Outputs seen: \(candidates)"
            case .missingStateOutput(let input, let expected):
                return
                    "PocketTTS mimi schema: state input `\(input)` has no matching output `\(expected)`"
            }
        }
    }

    /// Discover the Mimi schema from a loaded MLModel.
    static func discover(from model: MLModel) throws -> PocketTtsMimiSchema {
        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName

        // State inputs = every input except `latent`.
        let stateInputNames = inputs.keys.filter { $0 != "latent" }

        // ── Path A: v2 semantic schema ──────────────────────────────────
        if outputs["audio"] != nil {
            var mapping: [(String, String)] = []
            var shapes: [String: [Int]] = [:]
            var stateDtype: MLMultiArrayDataType = .float32
            for inputName in stateInputNames {
                // Pass-through state outputs (the `*_first` scalars and
                // zero-length `res*_conv1_prev` tensors) share an SSA value
                // with their input parameter and cannot be safely renamed at
                // conversion time — they keep the bare input name. All other
                // state outputs follow the `<input>_out` convention.
                let suffixed = "\(inputName)_out"
                let outputName: String
                if outputs[suffixed] != nil {
                    outputName = suffixed
                } else if outputs[inputName] != nil {
                    outputName = inputName  // pass-through alias
                } else {
                    throw DiscoveryError.missingStateOutput(
                        inputName: inputName, expectedOutput: suffixed)
                }
                mapping.append((inputName, outputName))
                if let constraint = inputs[inputName]?.multiArrayConstraint {
                    shapes[inputName] = constraint.shape.map { $0.intValue }
                    stateDtype = constraint.dataType
                }
            }
            let latentDtype = inputs["latent"]?.multiArrayConstraint?.dataType ?? .float32
            // Stable order: by input name (deterministic across runs).
            mapping.sort { $0.0 < $1.0 }
            return PocketTtsMimiSchema(
                audioOutputName: "audio", stateMapping: mapping, stateInputShapes: shapes,
                stateInputDataType: stateDtype, latentDataType: latentDtype)
        }

        // ── Path B: v1 legacy schema (HF FluidInference current English) ─
        // Pre-shipped models use auto-generated var_NNN / cast_NN names with
        // a known mapping baked at conversion time.
        if let v1 = legacyV1Schema, outputs[v1.audioOutputName] != nil {
            // Verify all expected state outputs exist; if any are missing,
            // fall through to error so we surface schema drift loudly.
            var verified: [(String, String)] = []
            var shapes: [String: [Int]] = [:]
            var stateDtype: MLMultiArrayDataType = .float32
            for (inp, out) in v1.stateMapping {
                guard inputs[inp] != nil else { continue }  // input dropped
                guard outputs[out] != nil else {
                    throw DiscoveryError.missingStateOutput(inputName: inp, expectedOutput: out)
                }
                verified.append((inp, out))
                if let constraint = inputs[inp]?.multiArrayConstraint {
                    shapes[inp] = constraint.shape.map { $0.intValue }
                    stateDtype = constraint.dataType
                }
            }
            let latentDtype = inputs["latent"]?.multiArrayConstraint?.dataType ?? .float32
            return PocketTtsMimiSchema(
                audioOutputName: v1.audioOutputName, stateMapping: verified,
                stateInputShapes: shapes, stateInputDataType: stateDtype,
                latentDataType: latentDtype)
        }

        throw DiscoveryError.missingAudioOutput(
            modelName: "mimi_decoder", candidates: Array(outputs.keys).sorted())
    }

    // MARK: - Legacy v1 fallback (current HF English pack)

    /// Legacy v1 mapping kept for backward compatibility with the
    /// FluidInference/pocket-tts-coreml English pack on HF (which still uses
    /// `var_NNN`/`cast_NN` output names from before semantic renaming was
    /// added to the converter).
    ///
    /// Once the v2 (FP16, semantic) pack is uploaded and the cache invalidated,
    /// this fallback can be removed.
    private static let legacyV1Schema: PocketTtsMimiSchema? = PocketTtsMimiSchema(
        audioOutputName: "var_821",
        stateMapping: [
            ("upsample_partial", "var_82"),
            ("attn0_cache", "var_262"),
            ("attn0_offset", "var_840"),
            ("attn0_end_offset", "new_end_offset_1"),
            ("attn1_cache", "var_479"),
            ("attn1_offset", "var_843"),
            ("attn1_end_offset", "new_end_offset"),
            ("conv0_prev", "var_607"),
            ("conv0_first", "conv0_first"),
            ("convtr0_partial", "var_634"),
            ("res0_conv0_prev", "var_660"),
            ("res0_conv0_first", "res0_conv0_first"),
            ("res0_conv1_prev", "res0_conv1_prev"),
            ("res0_conv1_first", "res0_conv1_first"),
            ("convtr1_partial", "var_700"),
            ("res1_conv0_prev", "var_726"),
            ("res1_conv0_first", "res1_conv0_first"),
            ("res1_conv1_prev", "res1_conv1_prev"),
            ("res1_conv1_first", "res1_conv1_first"),
            ("convtr2_partial", "var_766"),
            ("res2_conv0_prev", "var_792"),
            ("res2_conv0_first", "res2_conv0_first"),
            ("res2_conv1_prev", "res2_conv1_prev"),
            ("res2_conv1_first", "res2_conv1_first"),
            ("conv_final_prev", "var_824"),
            ("conv_final_first", "conv_final_first"),
        ],
        stateInputShapes: [:],  // populated from model description in `discover`
        stateInputDataType: .float32,
        latentDataType: .float32
    )
}
