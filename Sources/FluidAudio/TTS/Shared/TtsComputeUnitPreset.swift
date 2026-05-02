@preconcurrency import CoreML
import Foundation

/// Generic compute-unit preset shared across TTS backends.
///
/// Each backend keeps its own per-stage `<Backend>ComputeUnits` struct
/// because stage names differ (Kokoro ANE has 7 stages, PocketTTS has 4
/// CoreML models, StyleTTS2 has 4 models, etc.). This preset is the
/// uniform knob the benchmarking harness flips so a single CLI flag
/// (`--compute-units default|all-ane|cpu-and-gpu|cpu-only`) maps to a
/// sensible per-stage assignment on every backend.
///
/// Backends opt in by adding `init(preset: TtsComputeUnitPreset)` to
/// their compute-units struct (see `KokoroAneComputeUnits` for the
/// reference implementation).
public enum TtsComputeUnitPreset: String, Sendable, CaseIterable {

    /// The backend's empirically-tuned default — typically a mix of
    /// ANE-friendly and CPU+GPU stages chosen by the conversion author.
    case `default`

    /// Force every stage to `.cpuAndNeuralEngine`. Worst case for stages
    /// that fall back to CPU on ANE-incompatible ops, but the most
    /// energy-efficient when ops are ANE-clean.
    case allAne

    /// Force every stage to `.cpuAndGPU`. Skips the ANE entirely;
    /// useful as a latency baseline when the ANE compile cache is cold
    /// (no `anecompilerservice` time on first call).
    case cpuAndGpu

    /// Force every stage to `.cpuOnly`. Fallback / debugging baseline;
    /// every backend should at least run here, however slowly.
    case cpuOnly

    /// Concrete `MLComputeUnits` for "force every stage to X" presets.
    /// Returns `nil` for `.default`, which means "let the backend keep
    /// its empirical mapping".
    public var uniformUnits: MLComputeUnits? {
        switch self {
        case .default: return nil
        case .allAne: return .cpuAndNeuralEngine
        case .cpuAndGpu: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        }
    }

    /// Parse the CLI flag value (`default`, `all-ane`, `cpu-and-gpu`,
    /// `cpu-only`). Returns `nil` for unrecognised values so callers
    /// can surface a usage error.
    public init?(cliValue: String) {
        switch cliValue.lowercased() {
        case "default": self = .default
        case "all-ane", "ane", "neural-engine": self = .allAne
        case "cpu-and-gpu", "cpuandgpu", "gpu": self = .cpuAndGpu
        case "cpu-only", "cpu", "cpuonly": self = .cpuOnly
        default: return nil
        }
    }

    /// Canonical kebab-case form, matching the CLI flag values the
    /// `init?(cliValue:)` parser accepts. Use this for log lines and
    /// JSON reports so values round-trip back through the parser.
    public var cliValue: String {
        switch self {
        case .default: return "default"
        case .allAne: return "all-ane"
        case .cpuAndGpu: return "cpu-and-gpu"
        case .cpuOnly: return "cpu-only"
        }
    }
}
