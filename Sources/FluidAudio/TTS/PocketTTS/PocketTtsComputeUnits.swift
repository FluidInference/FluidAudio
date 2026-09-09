import CoreML
import Foundation

/// Per-stage Core ML compute-unit overrides for PocketTTS (#881).
///
/// Every field defaults to `nil`, which keeps the measured-fastest routing
/// for that stage and placement (documented in `PocketTtsModelStore`). Set a
/// field to move a single stage. Use `avoidNeuralEngine` when the ANE rejects
/// a stage on a given machine/OS — e.g. `flow_decoder_fused` aborting with
/// `ANEProgramProcessRequestDirect status=0x16` on an M1 Max under macOS 26.6,
/// where the default `.all` places it on the ANE and no `placement` avoids it.
public struct PocketTtsComputeUnits: Sendable, Hashable {
    /// `cond_prefill` (or `cond_prefill_ane`). Default `.all`.
    public var conditioner: MLComputeUnits?
    /// `flowlm_step` / `flowlm_stepv2` / `flowlm_step_ane`. Default `.all`,
    /// or `.cpuAndNeuralEngine` under `.ane` placement. Under `.aneState`
    /// this governs the fused `pocket_state` prefill/generate functions
    /// (default `.cpuAndNeuralEngine`), which subsume the flow decoder.
    public var flowLM: MLComputeUnits?
    /// `flow_decoder_fused`. Default `.all` (the one stage that is 100% ANE).
    public var flowDecoder: MLComputeUnits?
    /// `mimi_decoder`. Default `.cpuOnly`; keep it off the ANE — its fp16
    /// streaming-state feedback compounds into audible beeps there.
    public var mimiDecoder: MLComputeUnits?
    /// `mimi_encoder` (voice cloning only). Default `.cpuAndGPU`.
    public var mimiEncoder: MLComputeUnits?

    public init(
        conditioner: MLComputeUnits? = nil,
        flowLM: MLComputeUnits? = nil,
        flowDecoder: MLComputeUnits? = nil,
        mimiDecoder: MLComputeUnits? = nil,
        mimiEncoder: MLComputeUnits? = nil
    ) {
        self.conditioner = conditioner
        self.flowLM = flowLM
        self.flowDecoder = flowDecoder
        self.mimiDecoder = mimiDecoder
        self.mimiEncoder = mimiEncoder
    }

    /// The measured-fastest routing for every stage.
    public static let `default` = PocketTtsComputeUnits()

    /// Every stage on the same units. Prefer `avoidNeuralEngine` for the
    /// #881 case: it keeps `mimi_decoder` on the CPU, where it is fastest.
    public static func uniform(_ units: MLComputeUnits) -> PocketTtsComputeUnits {
        PocketTtsComputeUnits(
            conditioner: units, flowLM: units, flowDecoder: units,
            mimiDecoder: units, mimiEncoder: units)
    }

    /// No stage may be scheduled on the Neural Engine: GPU for the
    /// transformer stages, CPU for the mimi decoder. Pair with `.gpu`
    /// placement; `.aneState` is ANE-resident by design.
    public static let avoidNeuralEngine = PocketTtsComputeUnits(
        conditioner: .cpuAndGPU, flowLM: .cpuAndGPU, flowDecoder: .cpuAndGPU,
        mimiDecoder: .cpuOnly, mimiEncoder: .cpuAndGPU)

    /// Fully resolved units for the four IO-pipeline models.
    struct Resolved: Hashable {
        let conditioner: MLComputeUnits
        let flowLM: MLComputeUnits
        let flowDecoder: MLComputeUnits
        let mimiDecoder: MLComputeUnits
    }

    /// Overrides applied over the defaults for `placement` (`.gpu` / `.ane`).
    func resolved(for placement: PocketTtsModelPlacement) -> Resolved {
        Resolved(
            conditioner: conditioner ?? .all,
            flowLM: flowLM ?? (placement == .ane ? .cpuAndNeuralEngine : .all),
            flowDecoder: flowDecoder ?? .all,
            mimiDecoder: mimiDecoder ?? .cpuOnly
        )
    }

    /// Units for the `.aneState` multifunction package (prefill + generate).
    var resolvedStatePipeline: MLComputeUnits { flowLM ?? .cpuAndNeuralEngine }

    var resolvedMimiEncoder: MLComputeUnits { mimiEncoder ?? .cpuAndGPU }
}

extension MLComputeUnits {
    /// Stable log label (the raw value is an opaque integer).
    var pocketTtsLabel: String {
        switch self {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .all: return "all"
        @unknown default: return "unknown(\(rawValue))"
        }
    }
}
