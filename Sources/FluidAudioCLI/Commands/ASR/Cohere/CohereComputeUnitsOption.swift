#if os(macOS)
import CoreML
import FluidAudio
import Foundation

/// `--compute-units`, `--encoder-compute-units` and `--decoder-compute-units`
/// parsing shared by `cohere-transcribe` and `cohere-benchmark`.
///
/// Unset flags stay `nil` so `CoherePipeline.loadModels` applies its own
/// per-model defaults (encoder and decoder on `.cpuAndNeuralEngine`, see
/// FluidAudio #1045).
struct CohereComputeUnitsOption {
    var shared: MLComputeUnits?
    var encoder: MLComputeUnits?
    var decoder: MLComputeUnits?

    static let valueHelp = "all | cpu-only | cpu-gpu | ane"

    /// Parse one flag value. Returns `nil` for an unknown value.
    static func parse(_ raw: String) -> MLComputeUnits? {
        switch raw.lowercased() {
        case "all": return .all
        case "cpu-only", "cpu", "cpuonly": return .cpuOnly
        case "cpu-gpu", "cpu-and-gpu", "cpuandgpu", "gpu": return .cpuAndGPU
        case "ane", "cpu-ane", "cpu-and-ane", "cpuandneuralengine", "neural-engine": return .cpuAndNeuralEngine
        default: return nil
        }
    }

    /// Consume `arguments[index]` (and its value) if it is a compute-unit
    /// flag. Returns `false` when the flag is not one of ours so the caller
    /// can keep matching. Exits on an unknown value.
    mutating func consume(_ arguments: [String], at index: inout Int, logger: AppLogger) -> Bool {
        let flag = arguments[index]
        switch flag {
        case "--cpu-only":
            shared = .cpuOnly
            return true
        case "--cpu-gpu":
            shared = .cpuAndGPU
            return true
        case "--compute-units", "--encoder-compute-units", "--decoder-compute-units":
            guard index + 1 < arguments.count, let units = Self.parse(arguments[index + 1]) else {
                logger.error("\(flag) must be one of: \(Self.valueHelp)")
                exit(1)
            }
            index += 1
            switch flag {
            case "--encoder-compute-units": encoder = units
            case "--decoder-compute-units": decoder = units
            default: shared = units
            }
            return true
        default:
            return false
        }
    }

    static let usage = """
        Compute units:
            --compute-units <u>             Both models: \(valueHelp)
            --encoder-compute-units <u>     Encoder only (overrides --compute-units)
            --decoder-compute-units <u>     Decoder only (overrides --compute-units)
            --cpu-only / --cpu-gpu          Shorthands for --compute-units cpu-only / cpu-gpu
            (default: ane for both. With `all` Core ML may put the INT8 encoder on
             the GPU, which costs ~10 GB of extra memory; see Documentation/ASR/Cohere.md)
        """
}
#endif
