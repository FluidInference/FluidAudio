import CoreML
import Foundation

/// Which LocalVQE checkpoint to run.
///
/// Both are joint acoustic-echo-cancellation + noise-suppression +
/// dereverberation models from [localai-org/LocalVQE](https://github.com/localai-org/LocalVQE)
/// (Apache-2.0). Every variant needs a far-end **reference** signal (what the
/// loudspeaker played) alongside the microphone signal.
public enum LocalVqeVariant: String, CaseIterable, Sendable {
    /// v1.3, 4.8M params. Best joint quality; upstream default.
    case v13 = "v1.3"
    /// v1.2, 1.3M params. ~1/4 the per-hop cost of v1.3 for tight CPU budgets.
    case v12 = "v1.2"

    var fileStem: String {
        switch self {
        case .v13: return "localvqe-v1.3-4.8M"
        case .v12: return "localvqe-v1.2-1.3M"
        }
    }
}

/// How many 16 ms hops each Core ML call consumes.
///
/// The model is causal either way and produces bit-identical audio; the chunk
/// size only trades per-call overhead against latency.
public enum LocalVqeChunk: String, CaseIterable, Sendable {
    /// One 256-sample hop per call (16 ms). Use for live capture.
    case realtime16ms = "16ms"
    /// Sixteen hops per call (256 ms). ~2.5x the throughput of `realtime16ms`; use for files.
    case batch256ms = "256ms"

    /// Number of 256-sample hops consumed per call.
    public var framesPerCall: Int {
        switch self {
        case .realtime16ms: return 1
        case .batch256ms: return 16
        }
    }

    /// Samples consumed (and produced) per call.
    public var samplesPerCall: Int { framesPerCall * LocalVqeManager.hopSize }
}

public struct LocalVqeConfig: Sendable {
    public var variant: LocalVqeVariant
    public var chunk: LocalVqeChunk
    /// Defaults to `.cpuOnly`: the shipped models are fp32 (fp16 loses ~70 dB
    /// of parity in the recurrent bottleneck) and the per-hop graph is too
    /// small for ANE dispatch to pay off. `.cpuAndGPU` is ~20% faster on the
    /// 256 ms chunk on Apple Silicon Macs.
    public var computeUnits: MLComputeUnits

    public static let `default` = LocalVqeConfig()

    public init(
        variant: LocalVqeVariant = .v13,
        chunk: LocalVqeChunk = .batch256ms,
        computeUnits: MLComputeUnits = .cpuOnly
    ) {
        self.variant = variant
        self.chunk = chunk
        self.computeUnits = computeUnits
    }
}

public enum LocalVqeError: Error, LocalizedError {
    case notInitialized
    case modelLoadingFailed(String)
    case lengthMismatch(mic: Int, reference: Int)
    case modelProcessingFailed(String)

    public var errorDescription: String? {
        switch self {
        case .notInitialized:
            return "LocalVQE not initialized"
        case .modelLoadingFailed(let message):
            return "Failed to load LocalVQE model: \(message)"
        case .lengthMismatch(let mic, let reference):
            return "LocalVQE mic (\(mic)) and reference (\(reference)) sample counts differ"
        case .modelProcessingFailed(let message):
            return "LocalVQE processing failed: \(message)"
        }
    }
}
