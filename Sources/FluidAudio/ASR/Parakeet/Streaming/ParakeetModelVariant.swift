@preconcurrency import CoreML
import Foundation

/// Catalogues all available true streaming ASR model variants with cache-aware encoders.
///
/// These are models with native streaming architectures that maintain encoder cache states
/// across chunks. This does **not** include Parakeet TDT, which uses an offline encoder
/// in a sliding-window pseudo-streaming mode (use `AsrModelVersion` + `SlidingWindowAsrManager`
/// directly for TDT).
///
/// Call `createManager()` to instantiate the appropriate streaming ASR manager.
///
/// Following the `CtcModelVariant` pattern for consistency.
public enum StreamingModelVariant: String, CaseIterable, Sendable {
    // MARK: - Parakeet EOU (cache-aware streaming encoder, 120M params)

    /// Parakeet EOU 120M with 160ms chunks (lowest latency)
    case parakeetEou160ms = "parakeet-eou-160ms"
    /// Parakeet EOU 120M with 320ms chunks (balanced)
    case parakeetEou320ms = "parakeet-eou-320ms"
    /// Parakeet EOU 120M with 1280ms chunks (highest throughput)
    case parakeetEou1280ms = "parakeet-eou-1280ms"

    // MARK: - Nemotron Speech Streaming (cache-aware streaming, 0.6B params)

    /// Nemotron 0.6B with 2240ms chunks (default; highest throughput, +B1 fused decode)
    case nemotron2240ms = "nemotron-2240ms"
    /// Nemotron 0.6B with 1120ms chunks (lower latency)
    case nemotron1120ms = "nemotron-1120ms"
    /// Nemotron 0.6B with 560ms chunks (lowest-latency tier)
    case nemotron560ms = "nemotron-560ms"

    // MARK: - Parakeet Unified (chunked-attention streaming, 0.6B params)

    /// Parakeet Unified 0.6B, 2080ms latency (1.04s chunk + 1.04s right context).
    /// Stateless encoder re-run per chunk — streamed output matches offline quality.
    case parakeetUnified2080ms = "parakeet-unified-2080ms"
    /// Parakeet Unified 0.6B offline batch: full-attention 15s windows with 2s
    /// overlap, merged on the seams. Best WER; transcribes only at finish().
    case parakeetUnifiedOffline15s = "parakeet-unified-offline-15s"

    /// Human-readable display name
    public var displayName: String {
        switch self {
        case .parakeetEou160ms: return "Parakeet EOU 120M (160ms)"
        case .parakeetEou320ms: return "Parakeet EOU 120M (320ms)"
        case .parakeetEou1280ms: return "Parakeet EOU 120M (1280ms)"
        case .nemotron2240ms: return "Nemotron 0.6B (2240ms)"
        case .nemotron1120ms: return "Nemotron 0.6B (1120ms)"
        case .nemotron560ms: return "Nemotron 0.6B (560ms)"
        case .parakeetUnified2080ms: return "Parakeet Unified 0.6B (2080ms)"
        case .parakeetUnifiedOffline15s: return "Parakeet Unified 0.6B (offline 15s batch)"
        }
    }

    /// The HuggingFace repo for this variant's CoreML models
    public var repo: Repo {
        switch self {
        case .parakeetEou160ms: return .parakeetEou160
        case .parakeetEou320ms: return .parakeetEou320
        case .parakeetEou1280ms: return .parakeetEou1280
        case .nemotron2240ms: return .nemotronStreaming2240
        case .nemotron1120ms: return .nemotronStreaming1120
        case .nemotron560ms: return .nemotronStreaming560
        case .parakeetUnified2080ms, .parakeetUnifiedOffline15s: return .parakeetUnified
        }
    }

    /// Engine family grouping for factory dispatch
    public var engineFamily: EngineFamily {
        switch self {
        case .parakeetEou160ms, .parakeetEou320ms, .parakeetEou1280ms:
            return .parakeetEou
        case .nemotron2240ms, .nemotron1120ms, .nemotron560ms:
            return .nemotron
        case .parakeetUnified2080ms, .parakeetUnifiedOffline15s:
            return .parakeetUnified
        }
    }

    /// The streaming chunk size for EOU variants (nil for non-EOU)
    public var eouChunkSize: StreamingChunkSize? {
        switch self {
        case .parakeetEou160ms: return .ms160
        case .parakeetEou320ms: return .ms320
        case .parakeetEou1280ms: return .ms1280
        default: return nil
        }
    }

    /// The streaming chunk size for Nemotron variants (nil for non-Nemotron)
    public var nemotronChunkSize: NemotronChunkSize? {
        switch self {
        case .nemotron2240ms: return .ms2240
        case .nemotron1120ms: return .ms1120
        case .nemotron560ms: return .ms560
        default: return nil
        }
    }

    /// Create a streaming ASR manager for this variant.
    ///
    /// The returned manager is not yet loaded — call `loadModels()` before use.
    ///
    /// - Parameter configuration: Optional `MLModelConfiguration` override.
    /// - Returns: A streaming ASR manager conforming to `StreamingAsrManager`.
    public func createManager(
        configuration: sending MLModelConfiguration? = nil
    ) -> any StreamingAsrManager {
        // `sending` so the (non-Sendable) configuration transfers into the
        // actor manager inits without a data-race diagnostic. Xcode 16's iOS
        // build enforces this region check more strictly than the macOS 6.1
        // toolchain, which accepts the unannotated form.
        let mlConfig = configuration ?? MLModelConfiguration()
        switch engineFamily {
        case .parakeetEou:
            let chunkSize = eouChunkSize ?? .ms160
            return StreamingEouAsrManager(configuration: mlConfig, chunkSize: chunkSize)
        case .nemotron:
            let chunkSize = nemotronChunkSize ?? .ms2240
            return StreamingNemotronAsrManager(configuration: mlConfig, requestedChunkSize: chunkSize)
        case .parakeetUnified:
            if self == .parakeetUnifiedOffline15s {
                return UnifiedAsrManager(configuration: mlConfig)
            }
            return StreamingUnifiedAsrManager(configuration: mlConfig)
        }
    }

    /// Engine family types for true streaming models
    public enum EngineFamily: String, Sendable {
        /// Parakeet EOU: cache-aware streaming with end-of-utterance detection
        case parakeetEou = "parakeet-eou"
        /// Nemotron: cache-aware streaming with encoder cache states
        case nemotron = "nemotron"
        /// Parakeet Unified: chunked-attention streaming (stateless encoder re-runs)
        case parakeetUnified = "parakeet-unified"
    }
}
